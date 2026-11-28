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
from .accel_factor import AccelFactor, link_target
from .smoother import OrientationFactor, SegmentProblem, initial_rates, smooth
from .so3_lie import chordal_mean, geodesic_deg, log_so3, quat_xyzw_to_rot

FS = 60.0
EDGES = [(p, c) for c, p in PARENT.items() if p is not None]
METHODS = ('wnoj', 'mean_relative', 'copy_parent')
ACCEL_METHOD = 'wnoj_acc'
# Task 3 model identifiers (distinct from the Task 4 spine models M1-M3)
MODEL_IDS = {'copy_parent': 'T3-M0', 'mean_relative': 'T3-M1', 'wnoj': 'T3-M2',
             'wnoj_acc': 'T3-M3'}


def methods_for(params):
    """T3-M0..M2 always; T3-M3 only when acceleration is enabled (accel_unit='g')."""
    return METHODS + ((ACCEL_METHOD,) if params.accel_unit == 'g' else ())


@dataclass
class Params:
    qc: float = 1e3              # WNOJ jerk power spectral density (rad^2/s^5)
    joint_scale: float = 1.0     # lambda_j: multiplies the training joint spread
    sigma_gyro: float = 0.05     # rad/s, nominal
    sigma_ori_deg: float = 2.0   # own-quaternion factor, nominal
    factors: str = 'both'        # 'gyro' | 'orientation' | 'both' for observed segments
    max_iter: int = 12
    max_iter_withheld: int = 30  # joint-factor problems converge linearly (large residuals)
    accel_unit: str = 'none'     # 'g' enables T3-M3 (P0 verdict); 'none'/'disabled' never reads Acc
    sigma_acc: float = 2.0       # m/s^2, link-acceleration factor scale (chosen on validation)
    huber_k: float = 1.345


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
        r, w, a, _ = smooth(prob, r0, w0, np.gradient(w0, dt, axis=0), max_iter=params.max_iter)
        out[s] = (r, w, a)                # rotation, world rate, angular acceleration
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


def _accel_factor(w, observed, smoothed, accel):
    """T3-M3 link factor for a sandwiched withheld segment, or None (skipped safely)."""
    if accel is None or accel.get('bones') is None:
        return None
    p = PARENT[w]
    kids = [c for c in CHILDREN[w] if c in observed and c in smoothed and c in accel['acc']]
    if p is None or p not in observed or p not in smoothed or p not in accel['acc'] or not kids:
        return None
    bones = accel['bones']
    c = kids[0]
    if any(k not in bones for k in (p, w, c)):
        return None
    rp, wp, ap = smoothed[p]
    rc, wc, ac = smoothed[c]
    target, ok = link_target(accel['acc'][p], accel['acc'][c], rp, wp, ap, rc, wc, ac,
                             bones[p], bones[c])
    return AccelFactor(target, bones[w], accel['sigma'], accel['huber_k'], ok)


def reconstruct(smoothed, observed, withheld, stats, params, dt=1 / FS, accel=None):
    """Rebuild withheld segments. Returns method -> sensor -> (R, w) for withheld sensors.

    accel (T3-M3 only): {'acc': observed sensor -> raw (T,3) g, 'bones': sensor-frame bone
    vectors, 'sigma': m/s^2, 'huber_k'}.
    """
    methods = methods_for(params)
    out = {m: {} for m in methods}
    for m in methods:
        est = {s: smoothed[s] for s in observed if s in smoothed}
        for w in rebuild_order(withheld, observed):
            tg = _neighbour_targets(w, est, stats)
            parent_like = [x for x in tg if x[3] == 'parent'] or tg
            r_nb, mean, sd, kind = parent_like[0]
            if m == 'copy_parent':
                r = r_nb.copy()
            else:
                r = r_nb @ mean
            if m in ('wnoj', ACCEL_METHOD):
                factors = [OrientationFactor(rn @ mn, params.joint_scale * s_)
                           for rn, mn, s_, _ in tg]
                if len(tg) > 1:     # start at the precision-weighted chordal mean of the targets
                    acc = sum((rn @ mn) / np.mean(s_ ** 2) for rn, mn, s_, _ in tg)
                    u, _, vt = np.linalg.svd(acc)
                    fix = np.ones((len(acc), 3))
                    fix[:, 2] = np.sign(np.linalg.det(u @ vt))
                    r = (u * fix[:, None, :]) @ vt
                w0, a0 = initial_rates(r, dt)
                prob = SegmentProblem(dt=dt, qc=params.qc, orientation=factors)
                if m == ACCEL_METHOD:
                    af = _accel_factor(w, observed, smoothed, accel)
                    if af is not None:
                        prob.accel.append(af)
                r, wv, _, _ = smooth(prob, r, w0, a0, max_iter=params.max_iter_withheld)
            else:
                wv = initial_rates(r, dt)[0]
            est[w] = (r, wv)
            out[m][w] = (r, wv)
    return out


def smooth_with_external_prior(r_prior_dict, gyro_obs_or_none=None, dt=1 / FS, qc=1e3,
                               sigma_prior_deg=4.0, sigma_gyro=0.05, max_iter=30):
    """WNOJ post-smoothing of external per-frame rotation predictions (e.g. a BiGRU).

    r_prior_dict: sensor -> (T, 3, 3) predicted world rotations. gyro_obs_or_none:
    sensor -> (T, 3) world-frame rad/s for sensors whose gyro is observed (never pass a
    withheld sensor's gyro). Returns sensor -> (R, w) with R smoothed under the WNOJ prior
    and an orientation factor of sigma_prior_deg to the prediction (model T3-M5).
    """
    gyro_obs_or_none = gyro_obs_or_none or {}
    out = {}
    for s, r_prior in r_prior_dict.items():
        r_prior = np.asarray(r_prior, dtype=float)
        g = gyro_obs_or_none.get(s)
        prob = SegmentProblem(dt=dt, qc=qc, sigma_gyro=sigma_gyro,
                              gyro=None if g is None else np.asarray(g, dtype=float),
                              orientation=[OrientationFactor(r_prior, np.deg2rad(sigma_prior_deg))])
        w0, a0 = initial_rates(r_prior, dt)
        r, w, _, _ = smooth(prob, r_prior, w0, a0, max_iter=max_iter)
        out[s] = (r, w)
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
    accel = None
    if params.accel_unit == 'g':      # only then are accelerometer channels read
        need = needed_neighbours(observed, withheld)
        accel = {'acc': {s: obs_values[:, list(observed).index(s), 0:3].astype(float)
                         for s in need},
                 'bones': bones, 'sigma': params.sigma_acc, 'huber_k': params.huber_k}
    rec = reconstruct(smoothed, list(observed), withheld, stats, params, accel=accel)
    ref = rotations(values, names)
    out = {'labels': np.asarray(labels), 'v1': {}, 'v2': {}, 'v3': {}}
    for m in methods_for(params):
        for w in withheld:
            r_hat, w_hat = rec[m][w]
            out['v1'][(m, w)] = geodesic_deg(ref[w], r_hat).astype(np.float32)
            gyro = values[:, names.index(w), 3:6].astype(float)
            out['v2'][(m, w)] = np.linalg.norm(w_hat - gyro, axis=1).astype(np.float32)
        if bones is not None and all(k in bones for k in PARENT):    # V3 needs every bone
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
