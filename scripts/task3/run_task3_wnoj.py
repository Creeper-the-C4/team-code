"""Task 3 cloud entry point: WNOJ smoothing + withheld-sensor reconstruction evaluation.

Reads the independent_v1 cache through scripts/preprocessing/independent_data_access.py
and uses the six subject folds (20 train / 5 validation / 5 test).

Per fold:
- joint statistics come from TRAINING participants only;
- the two hyperparameters (qc, joint_scale) are chosen on that fold's VALIDATION
  participants (grid, skipped with --no-calibrate);
- metrics are computed once on that fold's TEST participants.

Schemes:
- H1: withhold one sensor at a time (--h1-sensors);
- H2: a fixed observed layout (--observed-sensors or --withhold);
- H3: the per-fold Task 2 layout from --fold-subsets-json (no cross-fold leakage).

Ablation: observed segments smoothed from gyro only, orientation only, or both,
over 2/5/10 s horizons. In gyro-only mode the first frame of each horizon is
anchored to the measured quaternion.

The reference type is 'withheld_sensor_reference' (not an optical ground truth).
Acceleration factors stay gated: --accel-unit none|disabled (default); g and mps2 are
refused until the P0 verdict.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')          # one BLAS thread per worker process
import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts' / 'preprocessing'))

from independent_data_access import SENSORS, SubjectData  # noqa: E402
from task3.kinematic_tree import bone_lengths, calibrate_bones  # noqa: E402
from task3.smoother import check_accel_unit  # noqa: E402
from task3.so3_lie import chordal_mean  # noqa: E402
from task3.withheld_eval import (METHODS, Params, evaluate_chunk, joint_statistics,  # noqa: E402
                                 needed_neighbours, rotations, smooth_observed, summarise)

ACTIVITY = {0: 'Others', 1: 'lying', 2: 'sitting', 3: 'standing', 4: 'slow_walking',
            5: 'moderate_walking', 6: 'brisk_walking', 7: 'ascending_stairs',
            8: 'descending_stairs', 9: 'cycling', 10: 'running', 11: 'jumping', 12: 'rowing'}
DEFAULT_H1 = ['Head', 'UpperBack', 'RightForeArm', 'LeftUpperArm', 'RightShank', 'LeftThigh']
QC_GRID = [1e2, 1e3, 1e4]
LAMBDA_GRID = [0.5, 1.0, 2.0]
REFERENCE = 'withheld_sensor_reference'


# ----------------------------------------------------------------------------- data access
def subject_chunks(root, subject, max_trials, max_seconds, chunk_seconds, first_chunk_only=False):
    """Yield (activity, values, labels) chunks inside official trials."""
    d = SubjectData(root, subject)
    t = d.trials
    per_act = {}
    for row in t.itertuples():
        act = int(row.majority_label)
        if max_trials and per_act.get(act, 0) >= max_trials:
            continue
        per_act[act] = per_act.get(act, 0) + 1
        values, labels = d.read_trial(int(row.trial), SENSORS)
        n = len(labels) if not max_seconds else min(len(labels), int(max_seconds * 60))
        step = int(chunk_seconds * 60)
        for s in range(0, n, step):
            e = min(s + step, n)
            if e - s < 120:
                break
            yield act, values[s:e], labels[s:e]
            if first_chunk_only:
                break


def training_samples(root, subjects, max_trials, stride=10):
    out = []
    for s in subjects:
        for _, v, _ in subject_chunks(root, s, max_trials, 60, 60):
            out.append({k: r[::stride] for k, r in rotations(v, SENSORS).items()})
    return out


def subject_bones(root, subject, dims):
    if dims is None or subject not in dims:
        return None
    rots = {}
    for act, v, _ in subject_chunks(root, subject, 1, 20, 20):
        if act == 3:
            rots = {k: chordal_mean(r) for k, r in rotations(v, SENSORS).items()}
            break
    return calibrate_bones(rots, bone_lengths(dims[subject]))


# ----------------------------------------------------------------------------- layouts
def parse_fold_subsets(path, fold, ks):
    """Observed layouts for one fold from a Task 2 output (flexible schema)."""
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    entry = None
    if isinstance(obj, dict) and 'folds' in obj:
        folds = obj['folds']
        if isinstance(folds, list):
            entry = next((f for f in folds if int(f.get('fold', -1)) == fold), None)
        else:
            entry = folds.get(str(fold))
    elif isinstance(obj, dict):
        entry = obj.get(str(fold), obj.get(fold))
    elif isinstance(obj, list):
        entry = next((f for f in obj if int(f.get('fold', -1)) == fold), None)
    if entry is None:
        raise ValueError(f'fold {fold} not found in {path}')

    def as_layout(x):
        if isinstance(x, list) and all(isinstance(s, str) for s in x):
            return x
        if isinstance(x, dict):
            for key in ('observed_sensors', 'selected', 'sensors', 'subset', 'recommended'):
                if key in x and as_layout(x[key]):
                    return as_layout(x[key])
        return None

    layouts = {}
    direct = as_layout(entry)
    if direct:
        layouts[len(direct)] = direct
    by_k = entry.get('by_k', entry) if isinstance(entry, dict) else {}
    for key, val in (by_k.items() if isinstance(by_k, dict) else []):
        if str(key).isdigit() and as_layout(val):
            layouts[int(key)] = as_layout(val)
    if ks:
        layouts = {k: v for k, v in layouts.items() if k in ks}
    if not layouts:
        raise ValueError(f'no usable layout for fold {fold} in {path}')
    return layouts


def build_schemes(args, fold):
    schemes = {}
    if 'H1' in args.schemes:
        for w in args.h1_sensors:
            schemes[f'H1:{w}'] = ([s for s in SENSORS if s != w], [w])
    if args.observed_sensors or args.withhold:
        obs = args.observed_sensors or [s for s in SENSORS if s not in args.withhold]
        schemes['H2:' + '+'.join(obs)] = (obs, [s for s in SENSORS if s not in obs])
    if args.fold_subsets_json:
        for k, obs in parse_fold_subsets(args.fold_subsets_json, fold, args.subset_k).items():
            schemes[f'H3:k{k}'] = (list(obs), [s for s in SENSORS if s not in obs])
    return schemes


# ----------------------------------------------------------------------------- workers
def eval_subject(job):
    root, subject, schemes, stats, params, bones, cfg = job
    rec = {}
    for act, v, lab in subject_chunks(root, subject, cfg['max_trials'], cfg['max_seconds'],
                                      cfg['chunk_seconds']):
        # each sensor is smoothed from its own channels only, so one cache serves all schemes
        need = sorted({n for obs, wh in schemes.values() for n in needed_neighbours(obs, wh)})
        cache = smooth_observed(v[:, [SENSORS.index(n) for n in need], :], need, need, params)
        for name, (obs, wh) in schemes.items():
            out = evaluate_chunk(v, lab, SENSORS, obs, wh, stats, params, bones, smoothed=cache)
            for key in ('v1', 'v2', 'v3'):
                for (m, s), arr in out[key].items():
                    rec.setdefault((name, key, m, s), []).append(arr)
            rec.setdefault((name, 'labels'), []).append(out['labels'])
    return subject, {k: np.concatenate(v) for k, v in rec.items()}


def calib_subject(job):
    root, subject, stats, sensors, cfg = job
    scores = {}
    chunks = list(subject_chunks(root, subject, 1, cfg['chunk_seconds'], cfg['chunk_seconds'],
                                 first_chunk_only=True))
    for qc in QC_GRID:
        caches = []
        for _, v, lab in chunks:
            need = sorted({n for w in sensors
                           for n in needed_neighbours([s for s in SENSORS if s != w], [w])})
            caches.append(smooth_observed(v[:, [SENSORS.index(n) for n in need], :], need, need,
                                          Params(qc=qc)))
        for lam in LAMBDA_GRID:
            p = Params(qc=qc, joint_scale=lam)
            meds = []
            for (_, v, lab), cache in zip(chunks, caches):
                for w in sensors:
                    out = evaluate_chunk(v, lab, SENSORS, [s for s in SENSORS if s != w], [w],
                                         stats, p, smoothed=cache)
                    meds.append(np.median(out['v1'][('wnoj', w)]))
            scores[(qc, lam)] = meds
    return scores


def ablation_subject(job):
    """Observed-segment factor ablation over 2/5/10 s horizons (first chunk per trial)."""
    from task3.kinematic_tree import PARENT
    from task3.so3_lie import geodesic_deg
    root, subject, stats, sensors, cfg = job
    rows = {}
    for horizon in (2, 5, 10):
        chunks = list(subject_chunks(root, subject, 1, horizon, horizon, first_chunk_only=True))
        for mode in ('gyro', 'orientation', 'both'):
            p = Params(qc=cfg['qc'], joint_scale=cfg['joint_scale'], factors=mode)
            for _, v, lab in chunks:
                need = sorted({n for w in sensors
                               for n in needed_neighbours([s for s in SENSORS if s != w], [w])})
                cache = smooth_observed(v[:, [SENSORS.index(n) for n in need], :], need, need, p)
                ref = rotations(v, SENSORS)
                for w in sensors:
                    out = evaluate_chunk(v, lab, SENSORS, [s for s in SENSORS if s != w], [w],
                                         stats, p, smoothed=cache)
                    key = (mode, horizon, w)
                    rows.setdefault(key, {'v1': [], 'self': []})
                    rows[key]['v1'].append(out['v1'][('wnoj', w)])
                    nb = PARENT[w] if PARENT[w] is not None else need[0]
                    rows[key]['self'].append(geodesic_deg(ref[nb], cache[nb][0]))
    return {k: {'v1': np.concatenate(v['v1']), 'self': np.concatenate(v['self'])}
            for k, v in rows.items()}


def pool_map(fn, jobs, workers):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, jobs))


# ----------------------------------------------------------------------------- outputs
def write_csv(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def fmt(x, d=3):
    return '-' if x is None else f'{x:.{d}f}'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--splits', required=True)
    ap.add_argument('--body-dims')
    ap.add_argument('--out', default='results/task3_v4')
    ap.add_argument('--fold-subsets-json')
    ap.add_argument('--subset-k', type=int, nargs='*', default=[3, 4, 5, 6])
    ap.add_argument('--observed-sensors', nargs='*')
    ap.add_argument('--withhold', nargs='*')
    ap.add_argument('--schemes', nargs='*', default=['H1'])
    ap.add_argument('--h1-sensors', nargs='*', default=DEFAULT_H1)
    ap.add_argument('--folds', type=int, nargs='*')
    ap.add_argument('--max-trials-per-activity', type=int, default=2)
    ap.add_argument('--max-seconds-per-trial', type=float, default=30.0)
    ap.add_argument('--chunk-seconds', type=float, default=10.0)
    ap.add_argument('--accel-unit', default='none', choices=['none', 'disabled', 'g', 'mps2'])
    ap.add_argument('--factors', default='both', choices=['gyro', 'orientation', 'both'])
    ap.add_argument('--qc', type=float, default=1e3)
    ap.add_argument('--joint-scale', type=float, default=1.0)
    ap.add_argument('--no-calibrate', action='store_true')
    ap.add_argument('--no-ablation', action='store_true')
    ap.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args(argv)
    check_accel_unit(args.accel_unit)
    if args.h1_sensors == ['all']:
        args.h1_sensors = list(SENSORS)
    out = Path(args.out)
    if (out / 'summary.json').exists() and not args.overwrite:
        print(f'{out}/summary.json exists; skipping (use --overwrite to recompute).')
        return 0
    parts = out / '_parts'
    parts.mkdir(parents=True, exist_ok=True)
    splits = json.loads(Path(args.splits).read_text(encoding='utf-8'))['folds']
    folds = [f for f in splits if args.folds is None or f['fold'] in args.folds]
    dims = None
    if args.body_dims:
        import pandas as pd
        bd = pd.read_csv(args.body_dims)
        dims = {int(r.subject): r._asdict() for r in bd.itertuples(index=False)}
    cfg = {'max_trials': args.max_trials_per_activity, 'max_seconds': args.max_seconds_per_trial,
           'chunk_seconds': args.chunk_seconds}
    t_start = time.time()
    fold_info, rows_subject, pooled, ablation = [], [], {}, {}
    for fold in folds:
        part = parts / f"fold{fold['fold']}.npz"
        meta = parts / f"fold{fold['fold']}.json"
        schemes = build_schemes(args, fold['fold'])
        if args.resume and part.exists() and meta.exists():
            info = json.loads(meta.read_text())
            data = dict(np.load(part, allow_pickle=True))
            subj_res = data['subj_res'].item()
            abl = data['ablation'].item()
        else:
            stats = joint_statistics(training_samples(args.data_root, fold['train'],
                                                      max(1, args.max_trials_per_activity)))
            qc, lam = args.qc, args.joint_scale
            calib = None
            if not args.no_calibrate:
                res = pool_map(calib_subject, [(args.data_root, s, stats, args.h1_sensors[:3], cfg)
                                               for s in fold['validation']], args.workers)
                score = {k: float(np.mean(np.concatenate([np.asarray(r[k]) for r in res])))
                         for k in res[0]}
                qc, lam = min(score, key=score.get)
                calib = {f'qc={k[0]:g},joint_scale={k[1]:g}': v for k, v in score.items()}
            params = Params(qc=qc, joint_scale=lam, factors=args.factors)
            jobs = [(args.data_root, s, schemes, stats, params,
                     subject_bones(args.data_root, s, dims), cfg) for s in fold['test']]
            subj_res = dict(pool_map(eval_subject, jobs, args.workers))
            abl = {}
            if not args.no_ablation:
                res = pool_map(ablation_subject, [(args.data_root, s, stats, args.h1_sensors[:3],
                                                   {'qc': qc, 'joint_scale': lam})
                                                  for s in fold['test']], args.workers)
                for r in res:
                    for k, v in r.items():
                        abl.setdefault(k, {'v1': [], 'self': []})
                        abl[k]['v1'].append(v['v1'])
                        abl[k]['self'].append(v['self'])
            info = {'fold': fold['fold'], 'qc': qc, 'joint_scale': lam, 'calibration': calib,
                    'train': fold['train'], 'validation': fold['validation'], 'test': fold['test'],
                    'schemes': {k: {'observed': v[0], 'withheld': v[1]} for k, v in schemes.items()}}
            np.savez(part, subj_res=np.array(subj_res, dtype=object),
                     ablation=np.array(abl, dtype=object))
            meta.write_text(json.dumps(info, indent=1))
        fold_info.append(info)
        for subj, rec in subj_res.items():
            for key, arr in rec.items():
                if key[1] == 'labels':
                    continue
                name, metric, m, s = key
                pooled.setdefault((name, metric, m, s), []).append(arr)
                if metric == 'v1':
                    pooled.setdefault((name, 'labels', m, s), []).append(rec[(name, 'labels')])
                    v2 = rec.get((name, 'v2', m, s))
                    rows_subject.append({'scheme': name, 'subject': subj, 'fold': info['fold'],
                                         'sensor': s, 'method': m, 'n_frames': int(arr.size),
                                         'v1_median_deg': float(np.median(arr)),
                                         'v1_rmse_deg': float(np.sqrt(np.mean(arr ** 2))),
                                         'v2_rmse_rad_s': float(np.sqrt(np.mean(v2 ** 2)))})
        for k, v in abl.items():
            ablation.setdefault(k, {'v1': [], 'self': []})
            ablation[k]['v1'] += list(v['v1'])
            ablation[k]['self'] += list(v['self'])

    summary, rows_activity = {}, []
    for (name, metric, m, s), arrs in pooled.items():
        if metric == 'labels':
            continue
        arr = np.concatenate(arrs)
        entry = summary.setdefault(name, {}).setdefault(s, {}).setdefault(m, {})
        entry[metric] = summarise(arr)
        if metric == 'v1':
            subj_meds = [r['v1_median_deg'] for r in rows_subject
                         if r['scheme'] == name and r['sensor'] == s and r['method'] == m]
            entry['v1']['subject_equal_median'] = float(np.mean(subj_meds))
            labels = np.concatenate(pooled[(name, 'labels', m, s)])
            v2 = np.concatenate(pooled[(name, 'v2', m, s)])
            for act in np.unique(labels):
                sel = labels == act
                rows_activity.append({'scheme': name, 'sensor': s, 'method': m,
                                      'activity': ACTIVITY.get(int(act), str(act)),
                                      'n_frames': int(sel.sum()),
                                      'v1_median_deg': float(np.median(arr[sel])),
                                      'v1_rmse_deg': float(np.sqrt(np.mean(arr[sel] ** 2))),
                                      'v1_p95_deg': float(np.percentile(arr[sel], 95)),
                                      'v2_rmse_rad_s': float(np.sqrt(np.mean(v2[sel] ** 2)))})
    rows_ablation = []
    for (mode, horizon, s), v in sorted(ablation.items()):
        v1 = np.concatenate(v['v1']) if v['v1'] else np.array([])
        sf = np.concatenate(v['self']) if v['self'] else np.array([])
        rows_ablation.append({'mode': mode, 'horizon_s': horizon, 'withheld_sensor': s,
                              'n_frames': int(v1.size),
                              'withheld_v1_median_deg': float(np.median(v1)) if v1.size else None,
                              'neighbour_self_median_deg': float(np.median(sf)) if sf.size else None})
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'per_activity_metrics.csv', rows_activity)
    write_csv(out / 'per_subject_metrics.csv', rows_subject)
    write_csv(out / 'ablation_summary.csv', rows_ablation)
    result = {'reference_type': REFERENCE,
              'note': 'Withheld-sensor reference: the hidden sensor\'s own recording, '
                      'not an independent optical ground truth.',
              'accel_unit': args.accel_unit, 'config': vars(args), 'folds': fold_info,
              'metrics': summary, 'runtime_s': round(time.time() - t_start, 1)}
    (out / 'summary.json').write_text(json.dumps(result, indent=1, default=str), encoding='utf-8')
    lines = ['# Task 3 WNOJ withheld-sensor reconstruction', '',
             f'Reference type: `{REFERENCE}` (not an optical ground truth). '
             f'Acceleration factors: `{args.accel_unit}`. Runtime {result["runtime_s"]} s.', '',
             '| scheme | withheld | method | V1 median (deg) | V1 subject-equal median | V1 P95 | '
             'V2 RMSE (rad/s) |', '|---|---|---|---:|---:|---:|---:|']
    for name, per_s in summary.items():
        for s, per_m in per_s.items():
            for m in METHODS:
                if m in per_m and 'v1' in per_m[m]:
                    e = per_m[m]
                    lines.append(f"| {name} | {s} | {m} | {fmt(e['v1']['median'], 2)} | "
                                 f"{fmt(e['v1']['subject_equal_median'], 2)} | "
                                 f"{fmt(e['v1']['p95'], 2)} | {fmt(e['v2']['rmse'])} |")
    from task3.kinematic_tree import PARENT

    def affected(scheme, ee):
        withheld = set(fold_info[0]['schemes'].get(scheme, {}).get('withheld', []))
        node = ee
        while node is not None:
            if node in withheld:
                return True
            node = PARENT[node]
        return False
    ee_rows = [(name, s, m, per_m[m]['v3']) for name, per_s in summary.items()
               for s, per_m in per_s.items() if s.startswith('EE:') and affected(name, s[3:])
               for m in METHODS if m in per_m]
    if ee_rows:
        lines += ['', 'V3 forward-kinematics end-point displacement (m; derived from a standing '
                  'calibration, not a measured position):', '',
                  '| scheme | end point | method | median | P95 |', '|---|---|---|---:|---:|']
        lines += [f"| {n} | {s[3:]} | {m} | {fmt(v['median'])} | {fmt(v['p95'])} |"
                  for n, s, m, v in ee_rows]
    lines += ['', 'Per-fold hyperparameters (chosen on validation participants): ' +
              ', '.join(f"fold {f['fold']}: qc={f['qc']:g}, joint_scale={f['joint_scale']:g}"
                        for f in fold_info),
              '', 'Files: summary.json, per_activity_metrics.csv, per_subject_metrics.csv, '
              'ablation_summary.csv.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
