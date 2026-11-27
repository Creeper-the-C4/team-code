"""P0 diagnostic: is the stored free acceleration in g or in m/s^2?

Decision rule, fixed BEFORE running on real data:
- T1 jump flight plateau (whole-body ballistic frames, torso sensors, median +
  MAD): g if in [-1.25, -0.75]; m/s^2 if in [-12, -7.5].
- T2 foot zero-velocity stride integration (stance from the gyro, whose rad/s
  unit is already confirmed): a hypothesis passes when the ascending-stairs
  stride rise is in [0.20, 0.50] m AND the level-walking stride length is in
  [0.80, 2.00] m.
- Decided only when T1 and T2 single out the same unit; otherwise 'unresolved'.
T3 (knee joint-centre lever arms), T4 (static residual as tilt) and T6
(gyro range, impact-spike widths) are reported as supporting evidence only.
"""
import argparse
from pathlib import Path

import numpy as np

from common import (ACC, FS, G, GYR, QUAT, SENSORS, SubjectData, available_subjects,
                    label_segments, quat_to_rot, robust, write_report)

RULE = {
    'T1': {'g': (-1.25, -0.75), 'mps2': (-12.0, -7.5)},
    'T2_stairs_rise_m': (0.20, 0.50),
    'T2_walk_stride_m': (0.80, 2.00),
}
SCALE = {'g': G, 'mps2': 1.0}
TORSO = ('LowerBack', 'UpperBack', 'Head', 'LeftThigh', 'RightThigh')
STANCE_GYRO = 0.6          # rad/s
MIN_STANCE = 5             # frames


def runs(mask):
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def t1_flight_plateaus(values):
    idx = [SENSORS.index(s) for s in TORSO]
    a = values[:, idx, 1].astype(float)
    m = np.median(a, axis=1)
    m = np.convolve(m, np.ones(3) / 3, mode='same')
    spread = a.max(axis=1) - a.min(axis=1)
    q05 = np.percentile(m, 5)
    if q05 >= 0:
        return []
    out = []
    for s, e in runs(m < 0.5 * q05):
        if e - s < 4 or s < 12 or e + 12 > len(m):
            continue
        if not (m[s - 12:s].max() > 0 and m[e:e + 12].max() > 0):
            continue                                   # needs take-off and landing
        level = float(np.median(m[s:e]))
        if np.median(spread[s:e]) > 0.5 * abs(level):
            continue                                   # not a common whole-body acceleration
        out.append(level)
    return out


def stance_midpoints(gyro):
    g = np.linalg.norm(gyro, axis=1)
    g = np.maximum.reduce([np.roll(g, k) for k in (-2, -1, 0, 1, 2)])
    return [(s + e) // 2 for s, e in runs(g < STANCE_GYRO) if e - s >= MIN_STANCE]


def stride_displacements(values, foot):
    """Raw (source-unit * s^2) displacement between consecutive stance midpoints."""
    i = SENSORS.index(foot)
    a = values[:, i, ACC].astype(float)
    mids = stance_midpoints(values[:, i, GYR].astype(float))
    out = []
    dt = 1.0 / FS
    for s, e in zip(mids[:-1], mids[1:]):
        if not 0.6 * FS <= e - s <= 2.0 * FS:
            continue
        v = np.cumsum(a[s:e], axis=0) * dt
        v -= np.outer(np.arange(1, e - s + 1) / (e - s), v[-1])   # zero velocity at both ends
        out.append(np.sum(v, axis=0) * dt)
    return out


def hat_many(v):
    out = np.zeros(v.shape[:-1] + (3, 3))
    out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
    out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
    return out


def knee_lever_arms(values, side):
    """Joint-centre regression a_th - a_sh = -G_th R_th r1 + G_sh R_sh r2 (vector form)."""
    th, sh = SENSORS.index(f'{side}Thigh'), SENSORS.index(f'{side}Shank')
    blocks, rhs = [], []
    for k, sign in ((th, -1.0), (sh, 1.0)):
        w = values[:, k, GYR].astype(float)
        alpha = np.gradient(w, axis=0) * FS
        r = quat_to_rot(values[:, k, QUAT])
        wx, ax = hat_many(w), hat_many(alpha)
        gam = ax + wx @ wx
        blocks.append(sign * np.einsum('nij,njk->nik', gam, r))
    a = np.concatenate(blocks, axis=2)[2:-2].reshape(-1, 6)
    b = (values[:, th, ACC] - values[:, sh, ACC]).astype(float)[2:-2].reshape(-1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    resid = b - a @ sol
    r2 = 1 - resid.var() / b.var() if b.var() > 0 else np.nan
    r1, r2v = sol[:3], sol[3:]
    norm_form = None
    try:
        from scipy.optimize import least_squares
        a_th = values[:, th, ACC].astype(float)[2:-2]
        a_sh = values[:, sh, ACC].astype(float)[2:-2]
        g_th = blocks[0][2:-2] * -1
        g_sh = blocks[1][2:-2]

        def fun(x):
            return (np.linalg.norm(a_th + np.einsum('nij,j->ni', g_th, x[:3]), axis=1)
                    - np.linalg.norm(a_sh + np.einsum('nij,j->ni', g_sh, x[3:]), axis=1))
        res = least_squares(fun, sol, max_nfev=200)
        norm_form = {'r1_norm_raw': float(np.linalg.norm(res.x[:3])),
                     'r2_norm_raw': float(np.linalg.norm(res.x[3:]))}
    except ImportError:
        pass
    return {'r1_norm_raw': float(np.linalg.norm(r1)), 'r2_norm_raw': float(np.linalg.norm(r2v)),
            'r2_fit': float(r2), 'norm_form': norm_form}


def in_range(x, lo_hi):
    return x is not None and lo_hi[0] <= x <= lo_hi[1]


def decide(t1_level, stairs_raw, walk_raw):
    t1 = [u for u, rng in RULE['T1'].items() if in_range(t1_level, rng)]
    t2 = [u for u, s in SCALE.items()
          if stairs_raw is not None and walk_raw is not None
          and in_range(stairs_raw * s, RULE['T2_stairs_rise_m'])
          and in_range(walk_raw * s, RULE['T2_walk_stride_m'])]
    t1u = t1[0] if len(t1) == 1 else None
    t2u = t2[0] if len(t2) == 1 else None
    verdict = t1u if t1u is not None and t1u == t2u else 'unresolved'
    return {'T1_unit': t1u, 'T2_unit': t2u, 'verdict': verdict}


def run(root, subjects=None, max_trials=3):
    subjects = subjects or available_subjects(root)
    t1, stairs_up, stairs_down, walk, knees, static, gyro_over, spikes = [], [], [], [], [], {}, {}, {}
    per_subject = {}
    for subj in subjects:
        d = SubjectData(root, subj)
        ps = {'t1': [], 'stairs_up': [], 'walk': []}
        for _, v in label_segments(d, 11, SENSORS, max_trials=max_trials):
            ps['t1'] += t1_flight_plateaus(v)
        for label, store in ((7, 'stairs_up'), (8, None), (4, 'walk'), (5, 'walk'), (6, 'walk')):
            for _, v in label_segments(d, label, SENSORS, max_trials=max_trials):
                for foot in ('LeftFoot', 'RightFoot'):
                    for disp in stride_displacements(v, foot):
                        if label == 7:
                            ps['stairs_up'].append(disp[1])
                        elif label == 8:
                            stairs_down.append(disp[1])
                        else:
                            ps['walk'].append(float(np.hypot(disp[0], disp[2])))
                if label in (4, 5, 6) and store == 'walk':
                    for side in ('Left', 'Right'):
                        knees.append(knee_lever_arms(v, side))
        for _, v in label_segments(d, 3, SENSORS, max_trials=max_trials, prepare_q=False):
            for i, s in enumerate(SENSORS):
                static.setdefault(s, []).append(float(np.median(np.linalg.norm(v[:, i, ACC], axis=1))))
        for tr in list(d.trials.trial)[:]:
            v, _ = d.read_trial(int(tr), SENSORS, prepare_q=False)
            gm = np.linalg.norm(v[:, :, GYR], axis=2)
            for i, s in enumerate(SENSORS):
                gyro_over[s] = gyro_over.get(s, 0) + int(np.sum(gm[:, i] > 34.9))
                if 'Foot' in s or 'Shank' in s:
                    am = np.linalg.norm(v[:, i, ACC], axis=1)
                    if am.size and am.max() > 0:
                        pk = int(np.argmax(am))
                        half = am >= 0.5 * am[pk]
                        lo = pk
                        while lo > 0 and half[lo - 1]:
                            lo -= 1
                        hi = pk
                        while hi < len(am) - 1 and half[hi + 1]:
                            hi += 1
                        spikes.setdefault(s, []).append((float(am[pk]), hi - lo + 1))
        t1 += ps['t1']
        stairs_up += ps['stairs_up']
        walk += ps['walk']
        per_subject[subj] = {'T1_plateau': robust(ps['t1']), 'T2_stairs_rise_raw': robust(ps['stairs_up']),
                             'T2_walk_stride_raw': robust(ps['walk'])}
    t1s, up, wk = robust(t1), robust(stairs_up), robust(walk)
    verdict = decide(t1s['median'], up['median'], wk['median'])
    static_med = {s: float(np.median(v)) for s, v in static.items()}
    report = {
        'rule': RULE, 'subjects': list(subjects), 'decision': verdict,
        'T1_flight_plateau': t1s,
        'T2_stairs_up_rise_raw': up, 'T2_stairs_down_rise_raw': robust(stairs_down),
        'T2_walk_stride_raw': wk,
        'T2_implied_m': {u: {'stairs_rise': None if up['median'] is None else up['median'] * s,
                             'walk_stride': None if wk['median'] is None else wk['median'] * s}
                         for u, s in SCALE.items()},
        'T3_knee_lever_arms_raw': {
            'r1': robust([k['r1_norm_raw'] for k in knees]),
            'r2': robust([k['r2_norm_raw'] for k in knees]),
            'fit_r2': robust([k['r2_fit'] for k in knees]),
            'norm_form_r1': robust([k['norm_form']['r1_norm_raw'] for k in knees if k['norm_form']]),
            'note': 'the fit returns r_true / scale, so metres = raw x scale '
                    '(g: raw x 9.81; m/s^2: raw). Supporting evidence only.',
        },
        'T4_static_norm_median': static_med,
        'T4_implied_tilt_deg': {u: {s: float(np.degrees(np.arcsin(min(1.0, v * sc / G))))
                                    for s, v in static_med.items()} for u, sc in SCALE.items()},
        'T6_gyro_frames_over_2000dps': gyro_over,
        'T6_foot_shank_peak_halfwidth_frames': {s: robust([w for _, w in v]) for s, v in spikes.items()},
        'per_subject': per_subject,
    }
    return report


def markdown(rep):
    d = rep['decision']
    lines = ['# diag_accel_unit', '', f"**Verdict: {d['verdict']}**  "
             f"(T1 -> {d['T1_unit']}, T2 -> {d['T2_unit']})", '',
             'Pre-registered rule: T1 plateau in [-1.25,-0.75] => g, in [-12,-7.5] => m/s^2; '
             'T2 stairs stride rise in [0.20,0.50] m and walking stride in [0.80,2.00] m.', '',
             '| test | n | median | MAD |', '|---|---:|---:|---:|']
    for k in ('T1_flight_plateau', 'T2_stairs_up_rise_raw', 'T2_stairs_down_rise_raw',
              'T2_walk_stride_raw'):
        v = rep[k]
        lines.append(f"| {k} | {v['n']} | {v['median']} | {v['mad']} |")
    lines += ['', f"T2 implied metres: {rep['T2_implied_m']}", '',
              f"T3 knee lever arms (raw units; metres = raw x scale): "
              f"r1 {rep['T3_knee_lever_arms_raw']['r1']}, r2 {rep['T3_knee_lever_arms_raw']['r2']}"]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', default='results/p0_diagnostics')
    ap.add_argument('--subjects', type=int, nargs='*')
    ap.add_argument('--max-trials', type=int, default=3)
    args = ap.parse_args(argv)
    rep = run(args.data_root, args.subjects, args.max_trials)
    write_report(args.out, 'diag_accel_unit', rep, markdown(rep))
    print(markdown(rep))
    return rep


if __name__ == '__main__':
    main()
