"""Withheld-sensor reconstruction of segment orientations and its evaluation.

Reference type: 'withheld_sensor_reference'. The recorded IMU of a sensor
hidden from the model is the reference; it is never an independent optical
ground truth.

Protocol for one chunk of one trial (no chunk crosses a trial boundary):
1. Each observed sensor is smoothed on its own (WNOJ + its own gyro and/or
   quaternion). No cross-sensor information enters this step.
2. Withheld segments are rebuilt in tree order from their available
   neighbours (a parent that is observed or already rebuilt, and observed
   children) through joint factors R_target = R_neighbour @ mean relative
   rotation, with WNOJ smoothing. The joint statistics (chordal mean and
   per-axis spread) come from TRAINING participants only.
3. Baselines: copy-parent (R_w = R_parent) and mean-relative
   (R_w = R_parent @ mean relative rotation).
4. Metrics against the withheld sensor's own recording, used only here:
   V1 geodesic angle (deg), V2 angular-velocity RMSE against its gyro
   (rad/s), V3 forward-kinematics end-point displacement (m) from a standing
   calibration. V3 is a derived quantity, not a measured position.
"""
from dataclasses import dataclass

import numpy as np

from .kinematic_tree import CHILDREN, END_EFFECTORS, PARENT, forward_kinematics, tree_order
from .smoother import OrientationFactor, SegmentProblem, initial_rates, smooth
from .so3_lie import chordal_mean, geodesic_deg, log_so3, quat_xyzw_to_rot

FS = 60.0
EDGES = [(p, c) for c, p in PARENT.items() if p is not None]
METHODS = ('wnoj', 'mean_relative', 'copy_parent')


@dataclass
class Params:
    qc: float = 1e3              # WNOJ jerk power spectral density (rad^2/s^5)
    joint_scale: float = 1.0     # lambda_j: multiplies the training joint spread
    sigma_gyro: float = 0.05     # rad/s, nominal
    sigma_ori_deg: float = 2.0   # own-quaternion factor, nominal
    factors: str = 'both'        # 'gyro' | 'orientation' | 'both' for observed segments
    max_iter: int = 8


def rotations(values, names):
    """sensor -> (T, 3, 3) from (T, S, 10) xyzw quaternions (already sign-continuous)."""
    return {s: quat_xyzw_to_rot(values[:, i, 6:10]) for i, s in enumerate(names)}


def joint_statistics(rot_samples):
    """Training-only joint statistics per tree edge.

    rot_samples: list of dicts sensor -> (n, 3, 3) from training participants.
    Returns {(parent, child): (mean R_p^T R_c, per-axis std of the residual)}.
    """
    out = {}
    for p, c in EDGES:
        rel = np.concatenate([np.swapaxes(d[p], -1, -2) @ d[c] for d in rot_samples
                              if p in d and c in d])
        mean = chordal_mean(rel)
        res = log_so3(mean.T @ rel)
        out[(p, c)] = (mean, np.maximum(res.std(axis=0), np.deg2rad(0.5)))
    return out


def smooth_observed(values, names, sensors, params, dt=1 / FS):
    """Smooth each listed observed sensor independently (own channels only)."""
    out = {}
    for s in sensors:
        i = names.index(s)
        r_meas = quat_xyzw_to_rot(values[:, i, 6:10])
        gyro = values[:, i, 3:6].astype(float)
        t = len(r_meas)
        factors = []
        if params.factors in ('orientation', 'both'):
            factors.append(OrientationFactor(r_meas, np.deg2rad(params.sigma_ori_deg)))
        else:                                     # gyro only: anchor the first frame
            mask = np.zeros(t, bool)
            mask[0] = True
            factors.append(OrientationFactor(r_meas, np.deg2rad(params.sigma_ori_deg), mask))
        prob = SegmentProblem(dt=dt, qc=params.qc, sigma_gyro=params.sigma_gyro,
                              gyro=None if params.factors == 'orientation' else gyro,
                              orientation=factors)
        if params.factors == 'gyro':
            r0 = np.repeat(r_meas[:1], t, axis=0)
            w0 = gyro
        else:
            r0, w0 = r_meas, initial_rates(r_meas, dt)[0]
        r, w, _, _ = smooth(prob, r0, w0, np.gradient(w0, dt, axis=0), max_iter=params.max_iter)
        out[s] = (r, w)
    return out


def _neighbour_targets(w, est, stats):
    """(neighbour rotation, relative mean, per-axis sigma, kind) for available neighbours.

    Parent: R_w ~ R_p @ M_pw. Child: R_w ~ R_c @ M_wc^T. A neighbour is available
    when it is observed or has already been rebuilt.
    """
    targets = []
    p = PARENT[w]
    if p is not None and p in est:
        mean, sd = stats[(p, w)]
        targets.append((est[p][0], mean, sd, 'parent'))
    for c in CHILDREN[w]:
        if c in est:
            mean, sd = stats[(w, c)]
            targets.append((est[c][0], mean.T, sd, 'child'))
    return targets


def rebuild_order(withheld, observed):
    """Withheld segments ordered so each has an observed or already-rebuilt neighbour."""
    done, order, todo = set(observed), [], [w for w in tree_order() if w in withheld]
    while todo:
        progress = False
        for w in list(todo):
            p = PARENT[w]
            if (p is not None and p in done) or any(c in done for c in CHILDREN[w]):
                order.append(w)
                done.add(w)
                todo.remove(w)
                progress = True
        if not progress:
            raise ValueError(f'No observed neighbour reachable for {todo}')
    return order


def reconstruct(smoothed, observed, withheld, stats, params, dt=1 / FS):
    """Rebuild withheld segments. Returns method -> sensor -> (R, w) for withheld sensors."""
    out = {m: {} for m in METHODS}
    for m in METHODS:
        est = {s: smoothed[s] for s in observed if s in smoothed}
        for w in rebuild_order(withheld, observed):
            tg = _neighbour_targets(w, est, stats)
            parent_like = [x for x in tg if x[3] == 'parent'] or tg
            r_nb, mean, sd, kind = parent_like[0]
            if m == 'copy_parent':
                r = r_nb.copy()
            else:
                r = r_nb @ mean
            if m == 'wnoj':
                factors = [OrientationFactor(rn @ mn, params.joint_scale * s_)
                           for rn, mn, s_, _ in tg]
                w0, a0 = initial_rates(r, dt)
                prob = SegmentProblem(dt=dt, qc=params.qc, orientation=factors)
                r, wv, _, _ = smooth(prob, r, w0, a0, max_iter=params.max_iter)
            else:
                wv = initial_rates(r, dt)[0]
            est[w] = (r, wv)
            out[m][w] = (r, wv)
    return out


def needed_neighbours(observed, withheld):
    """Observed sensors adjacent to any withheld segment (the only ones smoothing must cover)."""
    return sorted({s for w in withheld for s in [PARENT[w]] + CHILDREN[w]
                   if s is not None and s in observed})


def evaluate_chunk(values, labels, names, observed, withheld, stats, params, bones=None,
                   smoothed=None):
    """Per-frame metrics for one chunk; `values` holds all sensors but inference reads
    only `observed` columns (withheld columns are used for the reference only)."""
    obs_idx = [names.index(s) for s in observed]
    obs_values = values[:, obs_idx, :]
    if smoothed is None:
        smoothed = smooth_observed(obs_values, list(observed),
                                   needed_neighbours(observed, withheld), params)
    else:   # a shared cache may hold more sensors; only observed ones may be used
        smoothed = {s: smoothed[s] for s in observed if s in smoothed}
    rec = reconstruct(smoothed, list(observed), withheld, stats, params)
    ref = rotations(values, names)
    out = {'labels': np.asarray(labels), 'v1': {}, 'v2': {}, 'v3': {}}
    for m in METHODS:
        for w in withheld:
            r_hat, w_hat = rec[m][w]
            out['v1'][(m, w)] = geodesic_deg(ref[w], r_hat).astype(np.float32)
            gyro = values[:, names.index(w), 3:6].astype(float)
            out['v2'][(m, w)] = np.linalg.norm(w_hat - gyro, axis=1).astype(np.float32)
        if bones is not None:
            rots = {s: (smoothed[s][0] if s in smoothed else ref[s]) for s in observed}
            rots.update({w: rec[m][w][0] for w in withheld})
            fk_hat = forward_kinematics(rots, bones)
            fk_ref = forward_kinematics(ref, bones)
            for e in END_EFFECTORS:
                out['v3'][(m, f'EE:{e}')] = np.linalg.norm(fk_hat[e] - fk_ref[e],
                                                           axis=1).astype(np.float32)
    return out


def summarise(errors):
    e = np.asarray(errors, dtype=float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {'n': 0, 'rmse': None, 'median': None, 'p95': None}
    return {'n': int(e.size), 'rmse': float(np.sqrt(np.mean(e**2))),
            'median': float(np.median(e)), 'p95': float(np.percentile(e, 95))}
