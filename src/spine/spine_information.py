"""Measurement information vs prior-regularized posterior curvature.

I_meas = J^T R^-1 J is built from sensor rows only. H_post = I_meas + H_prior
adds the anatomical range-of-motion prior. Nuisance parameters (pelvis world
orientation and, for trajectories, pelvis translation) are removed by a
Schur complement, leaving information on the joint angles. Only I_meas
quantifies what a sensor contributes; H_post shows what becomes solvable
once the prior is added and must not be reported as a sensor gain.

Conventions:
- Orientation residual e_R(q) = Log(R_meas^T R_pred(q))^vee, evaluated at
  the linearisation point (R_meas = R_pred(q_bar)), so J = d e_R / d q.
- Perturbations: pelvis orientation uses the right retraction
  R <- R Exp(d); joint angles are rotation vectors with an additive chart
  theta <- theta + d (the analytic Jacobian carries the SO(3) right Jacobian).
- The Log branch near pi is never reached (spine rotations are well below
  60 deg; so3.log raises if it is).
- States are dimensionless radians. In the trajectory analysis lengths are
  in spine-span units and time in repetition units (kappa dimensionless).
- J^T R^-1 J is the Gauss-Newton PSD approximation of the Hessian (exact at
  zero residual; second-order residual terms are dropped).
"""
import numpy as np

from .spine_kinematics import FE, LUMBAR_LEVELS, deadlift_pose
from .spine_observation import (config_sensors, frame_position_jacobians, frame_rows,
                                state_dim)

REL_TOL = 1e-9
PRECISION_DEG = 5.0


def n_precise_directions(eig, std_deg=PRECISION_DEG):
    """Measurement-only 5-deg-precision eigen-directions: eigenvalues of I_meas with
    1/sqrt(lambda) < std_deg. Distinct from the numerical rank (structural directions)
    and from posterior standard deviations (which include the prior)."""
    return int(np.sum(np.asarray(eig) > 1.0 / np.deg2rad(std_deg) ** 2))

QOIS = {
    'lumbar_flexion_L1_S1': ('L5/S1',) + LUMBAR_LEVELS,
    'lumbar_flexion_L1_L5': LUMBAR_LEVELS,
    'lumbosacral_flexion_L5_S1': ('L5/S1',),
    'thoracolumbar_flexion_T12_L1': ('T12/L1',),
}


def schur(info, keep, drop):
    """Information on `keep` after marginalizing `drop` (flat prior on drop)."""
    a = info[np.ix_(keep, keep)]
    if len(drop) == 0:
        return a
    b = info[np.ix_(keep, drop)]
    c = info[np.ix_(drop, drop)]
    m = a - b @ np.linalg.pinv(c, rcond=1e-12, hermitian=True) @ b.T
    return 0.5 * (m + m.T)


def spectrum(m, rel_tol=REL_TOL):
    eig = np.clip(np.linalg.eigvalsh(m)[::-1], 0.0, None)
    tol = rel_tol * max(eig[0], 1e-300)
    pos = eig[eig > tol]
    return {
        'eigenvalues': eig,
        'rank': int(pos.size),
        'dim': int(m.shape[0]),
        'logdet_plus': float(np.sum(np.log(pos))) if pos.size else float('-inf'),
        'lambda_min_plus': float(pos[-1]) if pos.size else 0.0,
        'cond_plus': float(pos[0] / pos[-1]) if pos.size else float('inf'),
        'meas_5deg_precision_eigendirections': n_precise_directions(eig),
    }


def qoi_std(info, c, rel_tol=REL_TOL):
    """sqrt(c^T info^+ c), or inf when c has a component in the null space."""
    w, v = np.linalg.eigh(info)
    tol = rel_tol * max(w.max(), 1e-300)
    null = v[:, w <= tol]
    if null.size and np.linalg.norm(null.T @ c) > 1e-6 * np.linalg.norm(c):
        return float('inf')
    pos = w > tol
    return float(np.sqrt(np.sum((v[:, pos].T @ c) ** 2 / w[pos])))


def static_information(model, config, phase=0.5, sigma_ori_deg=2.0, sigma_bend_deg=2.0,
                       prior_scale=1.0):
    """Single-frame joint information at a deadlift posture (orientation/bend only).

    prior_scale multiplies the ROM-derived nominal prior standard deviations.
    """
    sensors = config_sensors(model, config)
    phi_p, theta = deadlift_pose(model, phase)
    rows = frame_rows(model, sensors, phi_p, theta, np.deg2rad(sigma_ori_deg),
                      np.deg2rad(sigma_bend_deg))
    d = state_dim(model)
    rows = rows[:, :d - 3]                    # pelvis translation is not seen by orientations
    info = rows.T @ rows
    keep = list(range(3, d - 3))
    prior = np.linalg.inv(model.prior_cov() * prior_scale ** 2)
    i_meas = schur(info, keep, [0, 1, 2])
    full_post = info.copy()
    full_post[3:, 3:] += prior
    h_post = schur(full_post, keep, [0, 1, 2])
    return i_meas, h_post, prior


def rank_metadata(model, config, i_meas):
    """Model-specific dimensions and the analytic rank bound of I_meas on joints.

    IMUs on the same rigid segment span the same three directions. L5 is the
    relative-orientation reference node; the pelvis (root) world orientation is
    a nuisance/reference variable (gauge, 3) eliminated by the Schur complement.
    The L5 IMU alone cannot separate the pelvis orientation from the internal
    L5/S1 joint angle, which is why that joint stays unobserved without a sacrum
    IMU. Hence rank <= min(3n, 3 * (distinct IMU segments - 1) + bend channels).
    """
    sensors = config_sensors(model, config)
    imus = [s for s in sensors if s.kind == 'imu']
    n_bend = sum(s.kind == 'bend' for s in sensors)
    distinct = len({s.segment for s in imus})
    eig = np.clip(np.linalg.eigvalsh(i_meas), 0.0, None)
    tol = REL_TOL * max(eig.max(), 1e-300)
    return {
        'state_dimension': 3 + 3 * model.n_joints,
        'measurement_dimension': 3 * len(imus) + n_bend,
        'gauge_dimension': 3,
        'theoretical_rank_upper_bound': min(3 * model.n_joints, 3 * (distinct - 1) + n_bend),
        'numerical_rank': int(np.sum(eig > tol)),
        'rank_tolerance': float(tol),
        'rank_tolerance_rule': f'{REL_TOL:g} x largest eigenvalue',
    }


def analyze_static(model, config, **kw):
    i_meas, h_post, prior = static_information(model, config, **kw)
    cov_prior = model.prior_cov() * kw.get('prior_scale', 1.0) ** 2
    out = {'model': model.name, 'config': config, 'metadata': rank_metadata(model, config, i_meas),
           'meas': spectrum(i_meas), 'post': spectrum(h_post), 'qoi': {}}
    for name, levels in QOIS.items():
        c = model.qoi_vector(levels, FE)
        if c is None:
            out['qoi'][name] = None
            continue
        out['qoi'][name] = {
            'prior_std_deg': float(np.rad2deg(np.sqrt(c @ cov_prior @ c))),
            'meas_only_std_deg': float(np.rad2deg(qoi_std(i_meas, c))),
            'posterior_std_deg': float(np.rad2deg(np.sqrt(c @ np.linalg.solve(h_post, c)))),
        }
    return out


def second_difference(n_frames, dt):
    d2 = np.zeros((n_frames - 2, n_frames))
    for t in range(n_frames - 2):
        d2[t, t:t + 3] = [1.0, -2.0, 1.0]
    return d2 / dt**2


def trajectory_information(model, config, n_frames=61, sigma_ori_deg=2.0,
                           sigma_bend_deg=2.0, kappa=None):
    """Joint-trajectory information over one synthetic deadlift repetition.

    Lengths are in spine-span units and time in repetition units, so kappa is
    the dimensionless acceleration noise sigma_a * tau^2 / L. kappa=None means
    orientation/bend rows only.
    """
    sensors = config_sensors(model, config)
    d = state_dim(model)
    big = np.zeros((d * n_frames, d * n_frames))
    times = np.linspace(0.0, 1.0, n_frames)
    phases = 0.5 * (1.0 - np.cos(2.0 * np.pi * times))   # down and back up
    pos_jacs = []
    for t, s in enumerate(phases):
        phi_p, theta = deadlift_pose(model, s)
        rows = frame_rows(model, sensors, phi_p, theta, np.deg2rad(sigma_ori_deg),
                          np.deg2rad(sigma_bend_deg))
        sl = slice(t * d, (t + 1) * d)
        big[sl, sl] += rows.T @ rows
        pos_jacs.append(frame_position_jacobians(model, sensors, phi_p, theta))
    if kappa is not None:
        d2 = second_difference(n_frames, times[1] - times[0])
        for k in range(len(pos_jacs[0])):
            acc = np.zeros((3 * (n_frames - 2), d * n_frames))
            for t in range(n_frames):
                col = d2[:, t]
                nz = np.nonzero(col)[0]
                for r in nz:
                    acc[3 * r:3 * r + 3, t * d:(t + 1) * d] += col[r] * pos_jacs[t][k]
            acc /= kappa
            big += acc.T @ acc
    joint_idx, nuisance_idx = [], []
    for t in range(n_frames):
        base = t * d
        nuisance_idx += list(range(base, base + 3)) + list(range(base + d - 3, base + d))
        joint_idx += list(range(base + 3, base + d - 3))
    return schur(big, joint_idx, nuisance_idx)


BASIS_DEFINITION = {
    'basis_definition': 'DCT-II: b_k(t_n) = cos(pi k t_n), t_n = (n + 0.5) / N, '
                        'n = 0..N-1, k = 0..K-1 (k = 0 is the constant mode)',
    'basis_normalization': 'each column scaled to unit Euclidean norm over the N frames '
                           '(orthonormal columns)',
    'boundary_condition': 'even (Neumann, zero-slope) extension at both ends, implicit in DCT-II',
    'time_interval': 'dimensionless repetition time [0, 1]',
}


def smooth_basis(n_frames, n_modes):
    """Orthonormal DCT-II cosine basis (n_frames x n_modes) for slow joint-angle changes."""
    t = (np.arange(n_frames) + 0.5) / n_frames
    b = np.stack([np.cos(np.pi * k * t) for k in range(n_modes)], axis=1)
    return b / np.linalg.norm(b, axis=0)


def project_smooth(info, n_frames, n_joint_dofs, n_modes):
    """Information on smooth joint trajectories theta(t) = sum_k c_k b_k(t).

    The second difference amplifies frame-to-frame jitter, so the full-space
    spectrum is dominated by high-frequency directions; slow posture changes
    (the deadlift-relevant ones) are what this projection isolates.
    """
    basis = np.kron(smooth_basis(n_frames, n_modes), np.eye(n_joint_dofs))
    m = basis.T @ info @ basis
    return 0.5 * (m + m.T)
