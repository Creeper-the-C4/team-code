"""Sparse Gauss-Newton MAP smoother for one segment under the WNOJ prior.

Unknowns: per frame (d_phi, d_w, d_a) = 9 numbers, so 9T in total. Factors:
- WNOJ prior between consecutive frames (whitened, see wnoj_prior);
- gyro factor (w_k - w_meas_k) / sigma_g on frames with a measurement;
- orientation-type factors e_R = Log(R_target^T R_k) / sigma on chosen frames.
  This covers the segment's own quaternion and the joint factors that tie a
  withheld segment to a neighbour, R_target = R_neighbour @ R_rel;
- optional link-acceleration factors (T3-M3, accel_factor.AccelFactor), IRLS-Huber
  weighted, registered only with accel_unit='g' (P0 verdict) for sandwiched segments.
The normal matrix J^T J is block-tridiagonal (bandwidth 17), so each solve is
O(T). The default solver accumulates the 9x9 blocks directly and uses a banded
Cholesky (scipy.linalg.solveh_banded); 'sparse' (scipy.sparse spsolve) and
'dense' paths build the same matrix and serve as cross-checks. Acceleration is
read only when accel_unit='g'; with 'none'/'disabled' (default) no acceleration
factor is registered and the accelerometer channels are never read.
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solveh_banded
from scipy.sparse.linalg import spsolve

from .accel_factor import ACCEL_G_TO_SI, ACCEL_SCHEMA  # noqa: F401  (re-exported)
from .accel_factor import AccelFactor, irls_weights
from .accel_factor import jacobian as accel_jacobian
from .accel_factor import residual as accel_residual
from .so3_lie import exp_so3
from .wnoj_prior import orientation_residual, residual_jacobians, whitener

ACCEL_UNITS = ('none', 'disabled', 'g', 'mps2')
# Factor types this module can register. 'accel_link' exists only for accel_unit='g'
# (P0 verdict); in the default mode it is never registered (second guard after the CLI).
FACTOR_TYPES = ('wnoj_prior', 'gyro', 'orientation', 'accel_link')
ACCEL_FACTOR_REGISTERED = False      # default-mode value; 'g' runs report their own flag
SOLVER_LOG = []          # per-process record of smooth() calls (iterations, convergence)
ROT_TOL = 1e-7           # rad: converged when the largest orientation update is below this


def check_accel_unit(accel_unit):
    if accel_unit not in ACCEL_UNITS:
        raise ValueError(f'accel_unit must be one of {ACCEL_UNITS}')
    if accel_unit == 'mps2':
        raise NotImplementedError(
            'The P0 diag_accel_unit verdict is "g" (gravity-compensated, in g); '
            'accel_unit="mps2" contradicts it and is refused.')


@dataclass
class OrientationFactor:
    target: np.ndarray            # (T, 3, 3) target rotations
    sigma: object                 # rad, scalar or per-axis (3,)
    mask: np.ndarray = None       # (T,) bool, frames where the factor applies


@dataclass
class SegmentProblem:
    dt: float
    qc: float                                     # WNOJ power spectral density (rad^2/s^5)
    gyro: np.ndarray = None                       # (T, 3) world-frame rad/s, or None
    sigma_gyro: float = 0.05
    orientation: list = field(default_factory=list)
    accel: list = field(default_factory=list)     # AccelFactor list (accel_unit='g' only)


class _Builder:
    """Accumulates sparse Jacobian blocks (COO) and stacked residuals."""

    def __init__(self):
        self.rows, self.cols, self.vals, self.res = [], [], [], []
        self.n_rows = 0

    def _blocks(self, blocks, row0, col0):
        k, nr, nc = blocks.shape
        rr = np.broadcast_to(row0[:, None, None] + np.arange(nr)[None, :, None], (k, nr, nc))
        cc = np.broadcast_to(np.asarray(col0)[:, None, None] + np.arange(nc)[None, None, :],
                             (k, nr, nc))
        self.rows.append(rr.ravel())
        self.cols.append(cc.ravel())
        self.vals.append(np.ascontiguousarray(blocks).ravel())

    def add(self, blocks, col0, resid, blocks2=None, col2=None):
        """Residual rows with one column block, or two blocks sharing the rows."""
        k, nr, _ = blocks.shape
        row0 = self.n_rows + np.arange(k) * nr
        self._blocks(blocks, row0, col0)
        if blocks2 is not None:
            self._blocks(blocks2, row0, col2)
        self.res.append(np.asarray(resid).ravel())
        self.n_rows += k * nr

    def matrix(self, n_cols):
        j = sp.csr_matrix((np.concatenate(self.vals),
                           (np.concatenate(self.rows), np.concatenate(self.cols))),
                          shape=(self.n_rows, n_cols))
        return j, np.concatenate(self.res)


def linear_system(problem, r, w, a):
    """Whitened Jacobian J (sparse) and residual vector r at the current state."""
    t = r.shape[0]
    b = _Builder()
    wmat = whitener(problem.dt, problem.qc)
    if t > 1:
        e, j0, j1 = residual_jacobians(r[:-1], w[:-1], a[:-1], r[1:], w[1:], a[1:], problem.dt)
        b.add(wmat @ j0, 9 * np.arange(t - 1), e @ wmat.T,
              blocks2=wmat @ j1, col2=9 * np.arange(1, t))
    if problem.gyro is not None:
        ok = np.all(np.isfinite(problem.gyro), axis=1)
        idx = np.nonzero(ok)[0]
        blk = np.broadcast_to(np.eye(3) / problem.sigma_gyro, (idx.size, 3, 3))
        b.add(blk, 9 * idx + 3, (w[idx] - problem.gyro[idx]) / problem.sigma_gyro)
    for f in problem.orientation:
        mask = np.ones(t, bool) if f.mask is None else np.asarray(f.mask, bool)
        idx = np.nonzero(mask)[0]
        e, jac = orientation_residual(f.target[idx], r[idx])
        sig = np.broadcast_to(np.asarray(f.sigma, dtype=float), (3,))
        b.add(jac / sig[None, :, None], 9 * idx, e / sig)
    for f in problem.accel:
        e = accel_residual(r, w, a, f)
        sw = np.sqrt(irls_weights(e, f)) / f.sigma
        idx = np.nonzero(sw > 0)[0]
        b.add(accel_jacobian(r[idx], w[idx], a[idx], f) * sw[idx, None, None], 9 * idx,
              e[idx] * sw[idx, None])
    return b.matrix(9 * t)


def retract(r, w, a, delta):
    d = delta.reshape(-1, 9)
    return exp_so3(d[:, 0:3]) @ r, w + d[:, 3:6], a + d[:, 6:9]


def normal_blocks(problem, r, w, a):
    """Block-tridiagonal normal equations: diagonal D (T,9,9), upper O (T-1,9,9) = H_{k,k+1},
    gradient g (T,9) = J^T r, and cost r^T r. Same matrix as J^T J from linear_system."""
    t = r.shape[0]
    dg = np.zeros((t, 9, 9))
    off = np.zeros((max(t - 1, 0), 9, 9))
    g = np.zeros((t, 9))
    cost = 0.0
    if t > 1:
        wmat = whitener(problem.dt, problem.qc)
        e, j0, j1 = residual_jacobians(r[:-1], w[:-1], a[:-1], r[1:], w[1:], a[1:], problem.dt)
        a_, b_, res = wmat @ j0, wmat @ j1, e @ wmat.T
        at = np.swapaxes(a_, 1, 2)
        bt = np.swapaxes(b_, 1, 2)
        dg[:-1] += at @ a_
        dg[1:] += bt @ b_
        off += at @ b_
        g[:-1] += np.einsum('kij,kj->ki', at, res)
        g[1:] += np.einsum('kij,kj->ki', bt, res)
        cost += float(np.sum(res ** 2))
    if problem.gyro is not None:
        ok = np.all(np.isfinite(problem.gyro), axis=1)
        res = (w[ok] - problem.gyro[ok]) / problem.sigma_gyro
        dg[ok, 3:6, 3:6] += np.eye(3) / problem.sigma_gyro ** 2
        g[ok, 3:6] += res / problem.sigma_gyro
        cost += float(np.sum(res ** 2))
    for f in problem.orientation:
        mask = np.ones(t, bool) if f.mask is None else np.asarray(f.mask, bool)
        e, jac = orientation_residual(f.target[mask], r[mask])
        sig = np.broadcast_to(np.asarray(f.sigma, dtype=float), (3,))
        jw = jac / sig[None, :, None]
        rw = e / sig
        jt = np.swapaxes(jw, 1, 2)
        dg[mask, 0:3, 0:3] += jt @ jw
        g[mask, 0:3] += np.einsum('kij,kj->ki', jt, rw)
        cost += float(np.sum(rw ** 2))
    for f in problem.accel:                       # IRLS: weights from the current residual
        e = accel_residual(r, w, a, f)
        sw = np.sqrt(irls_weights(e, f)) / f.sigma
        jw = accel_jacobian(r, w, a, f) * sw[:, None, None]
        rw = np.where(sw[:, None] > 0, e * sw[:, None], 0.0)
        jt = np.swapaxes(jw, 1, 2)
        dg += jt @ jw
        g += np.einsum('kij,kj->ki', jt, rw)
        cost += float(np.sum(rw ** 2))
    return dg, off, g, cost


_LOWER = np.tril_indices(9)


def blocks_to_banded(dg, off):
    """Lower banded storage (18, 9T) of the symmetric block-tridiagonal matrix."""
    t = dg.shape[0]
    ab = np.zeros((18, 9 * t))
    ii, jj = _LOWER
    cols = 9 * np.arange(t)[:, None] + jj[None, :]
    ab[(ii - jj)[None, :].repeat(t, 0), cols] = dg[:, ii, jj]
    if t > 1:
        i2, j2 = np.meshgrid(np.arange(9), np.arange(9), indexing='ij')
        i2, j2 = i2.ravel(), j2.ravel()
        lower = np.swapaxes(off, 1, 2)            # H_{k+1,k}
        cols = 9 * np.arange(t - 1)[:, None] + j2[None, :]
        ab[(9 + i2 - j2)[None, :].repeat(t - 1, 0), cols] = lower[:, i2, j2]
    return ab


def blocks_to_sparse(dg, off):
    t = dg.shape[0]
    rows, cols, vals = [], [], []
    base = 9 * np.arange(t)
    r_ = (base[:, None, None] + np.arange(9)[None, :, None]).repeat(9, 2)
    c_ = (base[:, None, None] + np.arange(9)[None, None, :]).repeat(9, 1)
    rows.append(r_.ravel()); cols.append(c_.ravel()); vals.append(dg.ravel())
    if t > 1:
        ro = r_[:-1].ravel()
        co = (c_[:-1] + 9).ravel()
        rows += [ro, co]; cols += [co, ro]; vals += [off.ravel(), off.ravel()]
    return sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(9 * t, 9 * t))


def gauss_newton_step(problem, r, w, a, damping=1e-9, solver='banded'):
    """One GN step. The normal matrix is Jacobi-equilibrated (D H D, D = diag(H)^-1/2),
    because angles, rates and accelerations differ by orders of magnitude.
    solver: 'banded' (default), 'sparse' or 'dense'. Returns (delta, info dict)."""
    dg, off, g, cost = normal_blocks(problem, r, w, a)
    diag = np.diagonal(dg, axis1=1, axis2=2).ravel()
    d = 1.0 / np.sqrt(diag + 1e-300)
    gs = d * g.ravel()
    n = d.size
    if solver == 'banded':
        ab = blocks_to_banded(dg, off)
        ab *= d[None, :] * np.stack([np.concatenate([d[l:], np.zeros(l)]) for l in range(18)])
        ab[0] += damping
        y = -solveh_banded(ab, gs, lower=True, check_finite=False)
        nnz = int(np.count_nonzero(ab))
    else:
        h = blocks_to_sparse(dg, off)
        dm = sp.diags(d, format='csc')
        hs = (dm @ h @ dm).tocsc() + damping * sp.identity(n, format='csc')
        y = -(np.linalg.solve(hs.toarray(), gs) if solver == 'dense' else spsolve(hs, gs))
        nnz = int(h.nnz)
    return d * y, {'cost': cost, 'nnz_normal_matrix': nnz}


def registered_factors(problem):
    """Factor types present in a problem (accel_link only when accel factors were added)."""
    out = ['wnoj_prior']
    if problem.gyro is not None:
        out.append('gyro')
    out += ['orientation'] * len(problem.orientation)
    out += ['accel_link'] * len(problem.accel)
    assert set(out) <= set(FACTOR_TYPES)
    return out


def smooth(problem, r0, w0=None, a0=None, max_iter=15, tol=ROT_TOL, accel_unit='none',
           solver='banded'):
    """Gauss-Newton MAP estimate; returns (R, w, a, info).

    Converged when the largest orientation update falls below `tol` (rad) within
    max_iter; failed when the state becomes non-finite. Each call is appended to
    SOLVER_LOG.
    """
    check_accel_unit(accel_unit)
    t = r0.shape[0]
    r = np.array(r0, dtype=float, copy=True)
    w = np.zeros((t, 3)) if w0 is None else np.array(w0, dtype=float, copy=True)
    a = np.zeros((t, 3)) if a0 is None else np.array(a0, dtype=float, copy=True)
    costs, nnz, converged, last = [], 0, False, np.inf
    for it in range(max_iter):
        delta, info = gauss_newton_step(problem, r, w, a, solver=solver)
        costs.append(info['cost'])
        nnz = info['nnz_normal_matrix']
        r, w, a = retract(r, w, a, delta)
        last = float(np.max(np.abs(delta.reshape(-1, 9)[:, 0:3])))
        if last < tol:
            converged = True
            break
    costs.append(normal_blocks(problem, r, w, a)[3])
    failed = not (np.all(np.isfinite(r)) and np.all(np.isfinite(w)) and np.isfinite(costs[-1]))
    SOLVER_LOG.append((it + 1, converged, failed, last))
    return r, w, a, {'iterations': it + 1, 'converged': converged, 'failed': failed,
                     'final_rotation_update_rad': last,
                     'cost': costs, 'nnz_normal_matrix': nnz, 'unknowns': 9 * t,
                     'factors': registered_factors(problem)}


def initial_rates(r, dt):
    """World angular velocity and acceleration from finite differences of R."""
    from .so3_lie import log_so3
    t = r.shape[0]
    w = np.zeros((t, 3))
    if t > 1:
        step = log_so3(r[1:] @ np.swapaxes(r[:-1], -1, -2)) / dt
        w[1:-1] = 0.5 * (step[1:] + step[:-1])
        w[0], w[-1] = step[0], step[-1]
    a = np.gradient(w, dt, axis=0) if t > 2 else np.zeros_like(w)
    return w, a
