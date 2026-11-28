"""Calibrate the WP3 dimensionless acceleration noise kappa from static (standing) data.

Read-only. For each participant, the standing (class 3) runs of the LowerBack and
UpperBack sensors are read through scripts/preprocessing/independent_data_access.py.
Each run is de-biased (its median is subtracted per axis). The per-axis robust
standard deviation is 1.4826 x MAD; the 3 x 3 covariance is computed after
winsorising at 5 robust standard deviations. Units follow the P0 verdict: stored
values are gravity-compensated free acceleration in g. SI conversion scales a
standard deviation by 9.80665 and a covariance by 9.80665^2.

kappa = sigma_a * tau^2 / L, with sigma_a the per-axis RMS of the robust covariance
(m/s^2), L the participant's torso_length_m (proxy for the L5 -> upper-back spine
span) and tau in {1.0, 1.5, 2.0} s by default.

Boundary: this is the STATIC measurement noise floor. Dynamic motion, soft-tissue
artefact and lever-arm errors are larger and are not included, so kappa here is
a lower bound for the deadlift trajectory analysis.

Outputs (population aggregates only, no per-participant rows), written to --out:
calibration_summary.json, calibration_table.csv and README.md. Optionally the WP3
slow-mode counts are recomputed at the calibrated kappa values as a supplement;
results/spine_offline_v1/ is never overwritten.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts' / 'preprocessing'))
sys.path.insert(0, str(REPO / 'src'))
from independent_data_access import SENSORS, SubjectData  # noqa: E402

G_TO_SI = 9.80665
SPINE_SENSORS = ('LowerBack', 'UpperBack')
STANDING = 3
FS = 60.0


def runs(labels, label):
    m = np.concatenate([[False], np.asarray(labels) == label, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def robust_stats(seg):
    """Per-axis robust std (g) and winsorised 3x3 covariance (g^2) of one de-biased run."""
    x = seg - np.median(seg, axis=0)
    sd = 1.4826 * np.median(np.abs(x), axis=0)
    lim = 5.0 * np.maximum(sd, 1e-12)
    xw = np.clip(x, -lim, lim)
    return sd, np.cov(xw.T)


def subject_noise(root, subject, max_runs, min_seconds):
    """{sensor: (per-axis sd (3,), cov (3,3))} averaged over standing runs, in g."""
    d = SubjectData(root, subject)
    idx = [SENSORS.index(s) for s in SPINE_SENSORS]
    acc = {s: ([], []) for s in SPINE_SENSORS}
    used = 0
    for tr in d.trials.itertuples():
        if getattr(tr, f'label_count_{STANDING}') < min_seconds * FS:
            continue
        values, labels = d.read_trial(int(tr.trial), list(SPINE_SENSORS), prepare_q=False)
        for s0, e0 in runs(labels, STANDING):
            if e0 - s0 < min_seconds * FS:
                continue
            for k, s in enumerate(SPINE_SENSORS):
                sd, cov = robust_stats(values[s0:e0, k, 0:3].astype(float))
                acc[s][0].append(sd)
                acc[s][1].append(cov)
            used += 1
            if used >= max_runs:
                break
        if used >= max_runs:
            break
    if used == 0:
        return None
    return {s: (np.median(np.array(v[0]), axis=0), np.mean(np.array(v[1]), axis=0))
            for s, v in acc.items()}


def agg(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {'n': 0, 'median': None, 'q25': None, 'q75': None, 'mean': None}
    return {'n': int(v.size), 'median': float(np.median(v)), 'q25': float(np.percentile(v, 25)),
            'q75': float(np.percentile(v, 75)), 'mean': float(np.mean(v))}


def supplementary(kappas, n_frames=41, n_modes=6):
    """WP3 M3 slow-mode 5-deg-precision counts at calibrated kappa (supplement only)."""
    from spine.spine_information import (n_precise_directions, project_smooth,
                                         trajectory_information)
    from spine.spine_kinematics import MODELS
    model = MODELS['M3']()
    out = {}
    base = {}
    for cfg in ('C0', 'C1'):
        info = trajectory_information(model, cfg, n_frames=n_frames)
        base[cfg] = n_precise_directions(np.linalg.eigvalsh(
            project_smooth(info, n_frames, 3 * model.n_joints, n_modes)))
    for k in sorted(set(round(float(x), 3) for x in kappas if x and x > 0)):
        info = trajectory_information(model, 'C0', n_frames=n_frames, kappa=k)
        out[f'{k:g}'] = n_precise_directions(np.linalg.eigvalsh(
            project_smooth(info, n_frames, 3 * model.n_joints, n_modes)))
    return {'model': 'M3', 'smooth_modes': n_modes, 'frames': n_frames,
            'C0_orientation_only': base['C0'], 'C1_orientation_only': base['C1'],
            'C0_plus_acc_by_kappa': out,
            'note': 'Model-based prediction; supplements results/spine_offline_v1 '
                    '(kappa grid 0.1/0.5/2.0), which is not overwritten.'}


def run(root, body_dims, subjects=None, taus=(1.0, 1.5, 2.0), max_runs=3, min_seconds=5.0,
        with_supplement=True):
    bd = pd.read_csv(body_dims)
    torso = dict(zip(bd.subject.astype(int), bd.torso_length_m.astype(float)))
    if subjects is None:
        subjects = sorted(int(p.name[1:]) for p in (Path(root) / 'subjects').glob('P*')
                          if (p / 'complete.json').exists())
    per = {s: subject_noise(root, s, max_runs, min_seconds) for s in subjects}
    per = {s: v for s, v in per.items() if v is not None}
    table, summary = [], {'sensors': {}}
    for sensor in SPINE_SENSORS:
        sd_g = np.array([per[s][sensor][0] for s in per])               # (n, 3)
        cov_g = np.array([per[s][sensor][1] for s in per])              # (n, 3, 3)
        sigma_si = np.sqrt(np.trace(cov_g, axis1=1, axis2=2) / 3.0) * G_TO_SI
        entry = {'robust_sd_g': {ax: agg(sd_g[:, i]) for i, ax in enumerate('XYZ')},
                 'robust_sd_si': {ax: agg(sd_g[:, i] * G_TO_SI) for i, ax in enumerate('XYZ')},
                 'cov_g2_mean': np.mean(cov_g, axis=0).tolist(),
                 'cov_si2_mean': (np.mean(cov_g, axis=0) * G_TO_SI ** 2).tolist(),
                 'sigma_a_si': agg(sigma_si)}
        summary['sensors'][sensor] = entry
        for i, ax in enumerate('XYZ'):
            table.append({'quantity': 'robust_sd', 'sensor': sensor, 'axis': ax, 'unit': 'm/s^2',
                          **agg(sd_g[:, i] * G_TO_SI)})
        table.append({'quantity': 'sigma_a', 'sensor': sensor, 'axis': 'rms', 'unit': 'm/s^2',
                      **agg(sigma_si)})
    kappa = {}
    for tau in taus:
        vals = []
        for s in per:
            if s not in torso:
                continue
            cov = 0.5 * (per[s]['LowerBack'][1] + per[s]['UpperBack'][1])
            sig = np.sqrt(np.trace(cov) / 3.0) * G_TO_SI
            vals.append(sig * tau ** 2 / torso[s])
        kappa[f'{tau:g}'] = agg(vals)
        table.append({'quantity': 'kappa_nominal', 'sensor': 'LowerBack+UpperBack',
                      'axis': f'tau={tau:g}s', 'unit': '1', **agg(vals)})
    summary.update({
        'definition': 'kappa = sigma_a * tau^2 / L; sigma_a = per-axis RMS of the robust '
                      'covariance (m/s^2); L = torso_length_m',
        'unit_rule': 'stored values are gravity-compensated free acceleration in g; '
                     'sd x 9.80665, covariance x 9.80665^2',
        'boundary': 'static measurement noise only (lower bound); dynamic motion, soft-tissue '
                    'and lever-arm errors are not included',
        'n_subjects': len(per), 'taus_s': list(taus), 'kappa_nominal': kappa,
        'original_grid': [0.1, 0.5, 2.0],
        'privacy': 'population aggregates only; no per-participant rows'})
    meds = [v['median'] for v in kappa.values() if v['median'] is not None]
    summary['outside_original_grid'] = bool(meds) and (min(meds) < 0.1 or max(meds) > 2.0)
    if with_supplement and meds:
        qs = [kappa[k][q] for k in kappa for q in ('q25', 'median', 'q75')
              if kappa[k][q] is not None]
        summary['supplementary_wp3'] = supplementary(qs)
    return summary, table


def write(out, summary, table):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'calibration_summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    with open(out / 'calibration_table.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    k = summary['kappa_nominal']
    lines = ['# WP3 kappa calibration from static P0 noise', '',
             f"Participants: {summary['n_subjects']} (population aggregates only).", '',
             f"Definition: {summary['definition']}.", f"Units: {summary['unit_rule']}.", '',
             f"**Boundary:** {summary['boundary']}.", '',
             '| tau (s) | kappa median | IQR |', '|---:|---:|---|']
    for tau, v in k.items():
        lines.append(f"| {tau} | {v['median']:.3g} | {v['q25']:.3g}-{v['q75']:.3g} |")
    if 'supplementary_wp3' in summary:
        s = summary['supplementary_wp3']
        lines += ['', 'Supplementary WP3 counts (M3, slow modes K=6, measurement-only '
                  '5-deg-precision directions; model-based prediction):', '',
                  f"C0 orientation only: {s['C0_orientation_only']}; "
                  f"C1 orientation only: {s['C1_orientation_only']}; "
                  f"C0 + acceleration by kappa: {s['C0_plus_acc_by_kappa']}"]
    lines += ['', 'results/spine_offline_v1/ is unchanged; this directory only adds the '
              'calibrated scenarios.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--body-dims', required=True)
    ap.add_argument('--out', default='results/spine_p0_calibration')
    ap.add_argument('--subjects', type=int, nargs='*')
    ap.add_argument('--taus', type=float, nargs='*', default=[1.0, 1.5, 2.0])
    ap.add_argument('--max-runs', type=int, default=3)
    ap.add_argument('--min-seconds', type=float, default=5.0)
    ap.add_argument('--no-supplement', action='store_true')
    args = ap.parse_args(argv)
    summary, table = run(args.data_root, args.body_dims, args.subjects, tuple(args.taus),
                         args.max_runs, args.min_seconds, not args.no_supplement)
    write(args.out, summary, table)
    print((Path(args.out) / 'README.md').read_text(encoding='utf-8'))
    return summary


if __name__ == '__main__':
    main()
