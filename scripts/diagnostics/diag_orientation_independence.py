"""P0 diagnostic: are orientations (and gyros) per-sensor outputs or skeleton-solved?

T5 hinge test (mounting-independent): for an ideal hinge, the relative angular
velocity expressed in the proximal segment frame, w_rel = R_p^T (w_c - w_p),
always lies along one fixed axis. Its second-moment matrix then has
sqrt(l2/l1) ~ 0. The ratio is computed twice, from the gyro and from
quaternion differences. Pre-registered flag 'skeleton_constrained_suspected':
median quaternion ratio < 0.05 AND median gyro ratio > 2 x quaternion ratio.

Residual analysis: e = w_gyro - w_quat (world frame) per sensor. Report the
gyro-vs-quaternion fit, the residual autocorrelation at lags 1/3/6 frames, and
the thigh-shank residual cross-correlation. Strongly correlated residuals
mean the 'gyro + orientation' factors must not be treated as independent.
"""
import argparse

import numpy as np

from common import (GYR, QUAT, SENSORS, SubjectData, available_subjects, batch_of,
                    label_segments, quat_to_rot, robust, world_rate_from_quat, write_report)

PAIRS = {'knee': ('Thigh', 'Shank'), 'elbow': ('UpperArm', 'ForeArm')}
LABELS = (4, 5, 6, 10, 12)
MIN_RATE = 0.2          # rad/s, ignore near-still frames in the hinge PCA
FLAG_QUAT, FLAG_FACTOR = 0.05, 2.0


def hinge_ratio(w_rel):
    keep = np.linalg.norm(w_rel, axis=1) > MIN_RATE
    if keep.sum() < 30:
        return np.nan
    lam = np.sort(np.linalg.eigvalsh(w_rel[keep].T @ w_rel[keep]))[::-1]
    return float(np.sqrt(max(lam[1], 0) / lam[0]))


def autocorr(x, lag):
    x = x - x.mean(axis=0)
    den = np.sum(x * x)
    return float(np.sum(x[lag:] * x[:-lag]) / den) if den > 0 else np.nan


def analyse_segment(v):
    out = {'hinge': {}, 'resid': {}, 'cross': {}}
    w_g = v[:, :, GYR].astype(float)
    w_q = np.stack([world_rate_from_quat(v[:, i, QUAT]) for i in range(len(SENSORS))], axis=1)
    rots = quat_to_rot(v[:, :, QUAT])
    sl = slice(2, -2)
    for joint, (p, c) in PAIRS.items():
        for side in ('Left', 'Right'):
            ip, ic = SENSORS.index(side + p), SENSORS.index(side + c)
            rel_g = np.einsum('nji,nj->ni', rots[sl, ip], (w_g[sl, ic] - w_g[sl, ip]))
            rel_q = np.einsum('nji,nj->ni', rots[sl, ip], (w_q[sl, ic] - w_q[sl, ip]))
            out['hinge'][f'{side}_{joint}'] = (hinge_ratio(rel_g), hinge_ratio(rel_q))
    resid = w_g[sl] - w_q[sl]
    for i, s in enumerate(SENSORS):
        g, q = w_g[sl, i].ravel(), w_q[sl, i].ravel()
        scale = float(g @ q / (q @ q)) if q @ q > 0 else np.nan
        r2 = 1 - np.var(g - scale * q) / np.var(g) if np.var(g) > 0 else np.nan
        out['resid'][s] = {'scale': scale, 'r2': float(r2),
                           **{f'acf_lag{k}': autocorr(resid[:, i], k) for k in (1, 3, 6)}}
    for side in ('Left', 'Right'):
        a = resid[:, SENSORS.index(side + 'Thigh')].ravel()
        b = resid[:, SENSORS.index(side + 'Shank')].ravel()
        out['cross'][f'{side}_thigh_shank'] = float(np.corrcoef(a, b)[0, 1])
    return out


def run(root, subjects=None, max_trials=2):
    subjects = subjects or available_subjects(root)
    hinge, resid, cross, by_batch = {}, {}, {}, {}
    for subj in subjects:
        d = SubjectData(root, subj)
        for label in LABELS:
            for _, v in label_segments(d, label, SENSORS, max_trials=max_trials):
                seg = analyse_segment(v)
                for k, (rg, rq) in seg['hinge'].items():
                    hinge.setdefault(k, {'gyro': [], 'quat': []})
                    hinge[k]['gyro'].append(rg)
                    hinge[k]['quat'].append(rq)
                    b = by_batch.setdefault(batch_of(subj) if subj <= 30 else 'other',
                                            {'gyro': [], 'quat': []})
                    if 'knee' in k:
                        b['gyro'].append(rg)
                        b['quat'].append(rq)
                for s, r in seg['resid'].items():
                    for kk, vv in r.items():
                        resid.setdefault(s, {}).setdefault(kk, []).append(vv)
                for k, c in seg['cross'].items():
                    cross.setdefault(k, []).append(c)
    hinge_sum = {}
    for k, v in hinge.items():
        g, q = robust(v['gyro']), robust(v['quat'])
        flag = (q['median'] is not None and g['median'] is not None
                and q['median'] < FLAG_QUAT and g['median'] > FLAG_FACTOR * q['median'])
        hinge_sum[k] = {'gyro_ratio': g, 'quat_ratio': q, 'skeleton_constrained_suspected': bool(flag)}
    return {
        'rule': {'flag': f'quat ratio < {FLAG_QUAT} and gyro ratio > {FLAG_FACTOR} x quat ratio',
                 'labels': list(LABELS), 'min_rate_rad_s': MIN_RATE},
        'subjects': list(subjects),
        'hinge': hinge_sum,
        'knee_hinge_by_batch': {k: {'gyro_ratio': robust(v['gyro']), 'quat_ratio': robust(v['quat'])}
                                for k, v in by_batch.items()},
        'gyro_vs_quat_residual': {s: {kk: robust(vv) for kk, vv in r.items()} for s, r in resid.items()},
        'thigh_shank_residual_correlation': {k: robust(v) for k, v in cross.items()},
    }


def markdown(rep):
    lines = ['# diag_orientation_independence', '', f"Flag rule: {rep['rule']['flag']}", '',
             '| joint | gyro ratio (median) | quat ratio (median) | skeleton-constrained suspected |',
             '|---|---:|---:|---|']
    for k, v in rep['hinge'].items():
        lines.append(f"| {k} | {v['gyro_ratio']['median']} | {v['quat_ratio']['median']} | "
                     f"{v['skeleton_constrained_suspected']} |")
    lines += ['', '| sensor | gyro~quat scale | R^2 | residual acf lag1 | lag6 |', '|---|---:|---:|---:|---:|']
    for s, r in rep['gyro_vs_quat_residual'].items():
        lines.append(f"| {s} | {r['scale']['median']} | {r['r2']['median']} | "
                     f"{r['acf_lag1']['median']} | {r['acf_lag6']['median']} |")
    lines += ['', f"Thigh-shank residual correlation: {rep['thigh_shank_residual_correlation']}"]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', default='results/p0_diagnostics')
    ap.add_argument('--subjects', type=int, nargs='*')
    ap.add_argument('--max-trials', type=int, default=2)
    args = ap.parse_args(argv)
    rep = run(args.data_root, args.subjects, args.max_trials)
    write_report(args.out, 'diag_orientation_independence', rep, markdown(rep))
    print(markdown(rep))
    return rep


if __name__ == '__main__':
    main()
