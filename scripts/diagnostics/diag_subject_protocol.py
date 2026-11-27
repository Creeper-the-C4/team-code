"""P0 diagnostic: subject/protocol heterogeneity, split audit, durations, walking cadence.

1. Trial structure per subject and per batch (P01-P24 / P25 / P26-P30):
   trial count, duration, labels per trial, within-trial label changes per
   minute, Others share, and the share of mixed 2 s windows.
2. Split audit: train/validation/test disjoint in every fold, every subject
   tested exactly once, and per-fold test-set class coverage.
3. Within-trial label run durations (seconds) per class. The mean run length
   gives tau for the sticky HMM, p_stay = exp(-dt/tau).
4. Walking cadence per subject and speed class from the thigh gyro in the
   thigh's own frame (principal axis, so heading- and mounting-invariant).
   Autocorrelation period with parabolic interpolation on 8 s chunks; thigh
   excursion per stride; dimensionless step frequency f*sqrt(L/g) when leg
   lengths are given. Checks whether slow < moderate < brisk holds per subject.
5. Kinematic walking speed v_hat = 2 L sin(dtheta_th/2) * f_step, checked
   against the fixed treadmill speeds of the protocol: slow/moderate/brisk =
   0.9/1.2/1.5 m/s (Feng et al. 2026). dtheta_th is the full peak-to-peak
   thigh excursion over a stride; f_step is steps per second (two per gait
   cycle); L approximates the effective hip-to-ankle leg length as thigh +
   shank length. It is a simplified kinematic (inverted-pendulum) estimate.
   The comparison is in the belt frame: the world-frame pelvis is ~stationary
   on a treadmill, and the belt speed is the protocol reference.
6. Optional: where misclassified windows of one class sit inside their trials,
   given an out-of-fold prediction table (--oof).
"""
import argparse
import json

import numpy as np
import pandas as pd

from common import (ACTIVITY, FS, G, GYR, QUAT, SENSORS, SubjectData, autocorr_period,
                    available_subjects, batch_of, label_runs, label_segments, quat_to_rot,
                    robust, write_report)

WALK = (4, 5, 6)
TREADMILL_MPS = {4: 0.9, 5: 1.2, 6: 1.5}      # protocol belt speeds (Feng et al. 2026)
CHUNK = int(8 * FS)


def trial_structure(d, subject, root):
    t = d.trials
    counts = t[[f'label_count_{k}' for k in range(13)]].to_numpy()
    labels_per_trial = (counts >= FS).sum(axis=1)
    changes = []
    for tr in t.itertuples():
        lab = np.asarray(d.labels[tr.source_start:tr.source_stop_exclusive])
        changes.append(np.sum(lab[1:] != lab[:-1]) / (len(lab) / FS / 60.0))
    w = pd.read_parquet(f'{root}/subjects/P{subject:02d}/windows_2s.parquet',
                        columns=['mixed_labels'])
    return {'batch': batch_of(subject) if subject <= 30 else 'other',
            'trials': int(len(t)), 'frames': int(t.frames.sum()),
            'duration_s': robust(t.duration_s), 'labels_per_trial': robust(labels_per_trial),
            'label_changes_per_min': robust(changes),
            'others_share': float(counts[:, 0].sum() / counts.sum()),
            'mixed_2s_window_share': float(w.mixed_labels.mean())}


def run_durations(d):
    out = {}
    for tr in d.trials.itertuples():
        lab = np.asarray(d.labels[tr.source_start:tr.source_stop_exclusive])
        for k in np.unique(lab):
            out.setdefault(int(k), []).extend((e - s) / FS for s, e in label_runs(lab, k))
    return out


def audit_splits(splits, subjects, class_frames):
    folds = splits['folds']
    problems, tested = [], []
    for f in folds:
        tr, va, te = set(f['train']), set(f['validation']), set(f['test'])
        if tr & va or tr & te or va & te:
            problems.append(f"fold {f['fold']}: overlapping roles")
        tested += list(te)
    if sorted(tested) != sorted(set(tested)):
        problems.append('a subject is tested more than once')
    missing = sorted(set(subjects) - set(tested))
    if missing:
        problems.append(f'never tested: {missing}')
    coverage = {}
    for f in folds:
        frames = sum((class_frames.get(s, np.zeros(13)) for s in f['test']), np.zeros(13))
        coverage[f['fold']] = {'missing_classes': [ACTIVITY[k] for k in range(13) if frames[k] == 0],
                               'test_frames': int(frames.sum())}
    return {'ok': not problems, 'problems': problems, 'test_coverage': coverage}


def thigh_cadence(v):
    """Step frequency (Hz) and thigh excursion (deg) from both thighs of one chunk."""
    freqs, excursions = [], []
    for side in ('Left', 'Right'):
        i = SENSORS.index(side + 'Thigh')
        r = quat_to_rot(v[:, i, QUAT])
        wb = np.einsum('nji,nj->ni', r, v[:, i, GYR].astype(float))   # body frame
        u, s_, vt = np.linalg.svd(wb - wb.mean(axis=0), full_matrices=False)
        sig = wb @ vt[0]
        period, strength = autocorr_period(sig, min_s=0.6, max_s=2.0)
        if np.isfinite(period) and strength > 0.3:
            freqs.append(2.0 / period)
            ang = np.cumsum(sig) / FS
            n = int(round(period * FS))
            chunks = [ang[k:k + n] for k in range(0, len(ang) - n + 1, n)]
            excursions += [np.degrees(c.max() - c.min()) for c in chunks]
    return (np.median(freqs) if freqs else np.nan,
            np.median(excursions) if excursions else np.nan)


def walking_cadence(d, max_trials):
    out = {}
    for label in WALK:
        f, ex = [], []
        for _, v in label_segments(d, label, SENSORS, min_frames=CHUNK, max_trials=max_trials):
            for k in range(0, len(v) - CHUNK + 1, CHUNK):
                fk, ek = thigh_cadence(v[k:k + CHUNK])
                f.append(fk)
                ex.append(ek)
        out[ACTIVITY[label]] = {'step_freq_hz': robust(f), 'thigh_excursion_deg': robust(ex)}
    return out


def oof_localisation(oof_path, root, true_class, pred_class):
    oof = pd.read_parquet(oof_path)
    cols = {c.lower(): c for c in oof.columns}
    need = [next((cols[c] for c in cands if c in cols), None) for cands in
            (('subject',), ('trial',), ('source_start',), ('source_stop_exclusive',),
             ('y_true', 'true', 'label', 'majority_label'), ('y_pred', 'pred', 'prediction'))]
    if None in need:
        return {'skipped': f'columns not recognised: {list(oof.columns)}'}
    sub, tri, st, sp, yt, yp = need
    bad = oof[(oof[yt] == true_class) & (oof[yp] == pred_class)]
    pos = []
    for subj, g in bad.groupby(sub):
        trials = pd.read_parquet(f'{root}/subjects/P{int(subj):02d}/trials.parquet').set_index('trial')
        for row in g.itertuples(index=False):
            r = row._asdict()
            tr = trials.loc[int(r[tri])]
            centre = (r[st] + r[sp]) / 2
            pos.append((centre - tr.source_start) / max(tr.frames, 1))
    pos = np.asarray(pos)
    return {'n': int(pos.size),
            'share_first_10pct': float(np.mean(pos < 0.1)) if pos.size else None,
            'share_last_10pct': float(np.mean(pos > 0.9)) if pos.size else None,
            'share_middle': float(np.mean((pos >= 0.1) & (pos <= 0.9))) if pos.size else None}


def run(root, splits_path, subjects=None, body_dims=None, oof=None, max_trials=3):
    subjects = subjects or available_subjects(root)
    legs = {}
    if body_dims:
        bd = pd.read_csv(body_dims)
        legs = dict(zip(bd.subject, bd.thigh_length_m + bd.shank_length_m))
    structure, durations, cadence, class_frames = {}, {}, {}, {}
    for subj in subjects:
        d = SubjectData(root, subj)
        structure[subj] = trial_structure(d, subj, root)
        class_frames[subj] = d.trials[[f'label_count_{k}' for k in range(13)]].sum().to_numpy()
        for k, v in run_durations(d).items():
            durations.setdefault(k, []).extend(v)
        cad = walking_cadence(d, max_trials)
        f = [cad[ACTIVITY[k]]['step_freq_hz']['median'] for k in WALK]
        cad['monotonic_slow_mod_brisk'] = (None if any(x is None for x in f)
                                           else bool(f[0] < f[1] < f[2]))
        if subj in legs and all(x is not None for x in f):
            cad['dimensionless_step_freq'] = [x * np.sqrt(legs[subj] / G) for x in f]
            speeds = {}
            for k in WALK:
                c = cad[ACTIVITY[k]]
                fk, ex = c['step_freq_hz']['median'], c['thigh_excursion_deg']['median']
                if fk is not None and ex is not None:
                    v_hat = 2 * legs[subj] * np.sin(np.radians(ex) / 2) * fk
                    speeds[ACTIVITY[k]] = {'v_hat_mps': float(v_hat),
                                           'belt_mps': TREADMILL_MPS[k],
                                           'ratio': float(v_hat / TREADMILL_MPS[k])}
            cad['kinematic_speed'] = speeds
        cadence[subj] = cad
    batches = {}
    for subj, s in structure.items():
        b = batches.setdefault(s['batch'], {'subjects': [], 'trials': [], 'mixed': [], 'others': [],
                                             'changes': []})
        b['subjects'].append(subj)
        b['trials'].append(s['trials'])
        b['mixed'].append(s['mixed_2s_window_share'])
        b['others'].append(s['others_share'])
        b['changes'].append(s['label_changes_per_min']['median'])
    splits = json.loads(open(splits_path, encoding='utf-8').read())
    report = {
        'subjects': list(subjects),
        'structure': structure,
        'batches': {k: {'subjects': v['subjects'], 'trials': robust(v['trials']),
                        'mixed_2s_window_share': robust(v['mixed']),
                        'others_share': robust(v['others']),
                        'label_changes_per_min': robust(v['changes'])} for k, v in batches.items()},
        'split_audit': audit_splits(splits, subjects, class_frames),
        'run_durations_s': {ACTIVITY[k]: {**robust(v), 'mean': float(np.mean(v))}
                            for k, v in sorted(durations.items())},
        'walking_cadence': cadence,
        'non_monotonic_subjects': [s for s, c in cadence.items()
                                   if c['monotonic_slow_mod_brisk'] is False],
        'treadmill_speed_check': {
            ACTIVITY[k]: {'belt_mps': TREADMILL_MPS[k],
                          'v_hat_mps': robust([c['kinematic_speed'][ACTIVITY[k]]['v_hat_mps']
                                               for c in cadence.values()
                                               if ACTIVITY[k] in c.get('kinematic_speed', {})]),
                          'ratio': robust([c['kinematic_speed'][ACTIVITY[k]]['ratio']
                                           for c in cadence.values()
                                           if ACTIVITY[k] in c.get('kinematic_speed', {})])}
            for k in WALK} if legs else None,
    }
    if oof:
        report['rowing_to_sitting_position'] = oof_localisation(oof, root, 12, 2)
    return report


def markdown(rep):
    lines = ['# diag_subject_protocol', '', '## Batches', '',
             '| batch | subjects | trials (median) | mixed 2s share | Others share | label changes/min |',
             '|---|---:|---:|---:|---:|---:|']
    for k, v in rep['batches'].items():
        lines.append(f"| {k} | {len(v['subjects'])} | {v['trials']['median']} | "
                     f"{v['mixed_2s_window_share']['median']:.4f} | {v['others_share']['median']:.4f} | "
                     f"{v['label_changes_per_min']['median']:.3f} |")
    a = rep['split_audit']
    lines += ['', f"## Split audit: {'OK' if a['ok'] else 'PROBLEMS'} {a['problems']}", '',
              '## Walking step frequency (Hz) per subject', '',
              '| subject | slow | moderate | brisk | monotonic |', '|---|---:|---:|---:|---|']
    for s, c in rep['walking_cadence'].items():
        f = [c[ACTIVITY[k]]['step_freq_hz']['median'] for k in WALK]
        lines.append(f"| {s} | " + ' | '.join('-' if x is None else f'{x:.3f}' for x in f)
                     + f" | {c['monotonic_slow_mod_brisk']} |")
    lines += ['', f"Non-monotonic subjects: {rep['non_monotonic_subjects']}"]
    if rep.get('treadmill_speed_check'):
        lines += ['', '## Kinematic speed vs treadmill belt speed', '',
                  '| class | belt (m/s) | v_hat median (m/s) | ratio median | ratio MAD |',
                  '|---|---:|---:|---:|---:|']
        for k, v in rep['treadmill_speed_check'].items():
            lines.append(f"| {k} | {v['belt_mps']} | {v['v_hat_mps']['median']} | "
                         f"{v['ratio']['median']} | {v['ratio']['mad']} |")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--splits', required=True)
    ap.add_argument('--body-dims')
    ap.add_argument('--oof')
    ap.add_argument('--out', default='results/p0_diagnostics')
    ap.add_argument('--subjects', type=int, nargs='*')
    ap.add_argument('--max-trials', type=int, default=3)
    args = ap.parse_args(argv)
    rep = run(args.data_root, args.splits, args.subjects, args.body_dims, args.oof, args.max_trials)
    write_report(args.out, 'diag_subject_protocol', rep, markdown(rep))
    print(markdown(rep))
    return rep


if __name__ == '__main__':
    main()
