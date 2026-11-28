"""Recount Task 2 predictions and export aggregate-only results for the team."""
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import hashlib
import json
import re
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path('/root/autodl-tmp/C_Problem')
SOURCE = PROJECT / 'results/task2_v4_20261128'
DEST = PROJECT / 'reports/task2_v4_20261128_public'
MAIN = 'F1H_backward'
CLASSES = ('Others', 'Lying', 'Sitting', 'Standing', 'Slow walking', 'Moderate walking',
           'Brisk walking', 'Ascending stairs', 'Descending stairs', 'Cycling', 'Running',
           'Jumping', 'Rowing')
ALLOW = ('README.md', 'config.json', 'pareto_curve.csv', 'per_fold_pareto.csv',
         'per_fold_selected_subsets.json', 'ranking_prefix_pareto_curve.csv',
         'sensor_rank_stability.csv', 'summary.json')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def f1(matrices):
    diagonal = np.diagonal(matrices, axis1=-2, axis2=-1)
    denominator = matrices.sum(axis=-2) + matrices.sum(axis=-1)
    return np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal, dtype=float),
                     where=denominator != 0)


def scores(by_subject):
    total = by_subject.sum(axis=0)
    return dict(pooled=float(f1(total).mean()), equal=float(f1(by_subject).mean()),
                accuracy=float(total.trace() / total.sum()))


def paired_ci(candidate, baseline):
    # Cluster bootstrap by participant conditional on the existing six fits.
    rng = np.random.default_rng(20261128)
    indices = rng.integers(0, len(baseline), size=(3000, len(baseline)))
    pooled = (f1(candidate[indices].sum(axis=1)).mean(axis=1)
              - f1(baseline[indices].sum(axis=1)).mean(axis=1))
    equal = (f1(candidate).mean(axis=1) - f1(baseline).mean(axis=1))[indices].mean(axis=1)
    return {'pooled_delta_ci95': np.quantile(pooled, [0.025, 0.975]).tolist(),
            'equal_delta_ci95': np.quantile(equal, [0.025, 0.975]).tolist()}


def recount(directory, matrix, expected_windows):
    seen = set()
    for path in sorted(directory.glob('fold_*/test_predictions.parquet')):
        fold = int(path.parent.name.split('_')[-1])
        mark = read(path.parent / 'complete.json')
        parts = [set(mark['participants'][role]) for role in ('train', 'validation', 'test')]
        assert [len(x) for x in parts] == [20, 5, 5]
        assert len(set.union(*parts)) == 30 and not seen.intersection(parts[2])
        seen.update(parts[2])
        columns = ['k', 'subject', 'trial', 'source_start', 'majority_label', 'y_pred', 'fold']
        df = pd.read_parquet(path, columns=columns)
        assert df['fold'].eq(fold).all() and set(df['subject']) == parts[2]
        assert set(df['k']) == set(range(1, 18))
        coords = ['subject', 'trial', 'source_start', 'majority_label']
        reference = df.loc[df['k'].eq(17), coords].reset_index(drop=True)
        assert not reference.duplicated(coords[:3]).any()
        for k, group in df.groupby('k', sort=False):
            assert reference.equals(group[coords].reset_index(drop=True))
            person, truth, predicted = (group[coln].to_numpy(dtype=np.int64)
                                       for coln in ('subject', 'majority_label', 'y_pred'))
            assert np.isin(truth, np.arange(13)).all() and np.isin(predicted, np.arange(13)).all()
            bins = ((person - 1) * 13 + truth) * 13 + predicted
            matrix[int(k) - 1] += np.bincount(bins, minlength=30 * 13 * 13).reshape((30, 13, 13))
    assert seen == set(range(1, 31))
    assert (matrix.sum(axis=(1, 2, 3)) == expected_windows).all()


def plot(curves, dest):
    labels = {'F1H_ranking_prefix': 'F1H gain ranking', 'F1H_backward': 'F1H backward (primary)',
              'F3_backward': 'F3 backward', 'F1H_forward': 'F1H forward'}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), layout='constrained', sharex=True)
    for ax, col, title in zip(axes, ('subject_equal_macro_f1_13', 'pooled_macro_f1_13'),
                              ('A   Participant-equal', 'B   Pooled windows')):
        for name in (MAIN, 'F1H_ranking_prefix', 'F3_backward', 'F1H_forward'):
            rows = curves[name]
            ax.plot(rows['k'], 100 * rows[col], label=labels[name],
                    linewidth=2.4 if name == MAIN else 1.35,
                    marker='o' if name == MAIN else None, markersize=3.5)
        ax.set(title=title, xlabel='Number of retained sensors', ylabel='13-class macro-F1 (%)',
               xticks=(1, 3, 5, 8, 10, 13, 15, 17), xlim=(0.8, 17.2))
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, loc='lower right', fontsize=8)
    fig.savefig(dest / 'sensor_count_tradeoff.png', dpi=200)
    plt.close(fig)


def main():
    if DEST.exists():
        raise FileExistsError(DEST)
    DEST.mkdir(parents=True)
    stage = read(SOURCE / 'stage_status.json')
    manifest = read(SOURCE / 'run_manifest.json')
    names = [exp['name'] for exp in manifest['experiments']]
    assert stage['status'] == 'complete' and stage['completed'] == names
    write(DEST / 'stage_status.json', {k: v for k, v in stage.items() if k != 'pid'})
    shutil.copyfile(SOURCE / 'run_manifest.json', DEST / 'run_manifest.json')
    scripts = DEST / 'execution'; scripts.mkdir()
    patch = subprocess.check_output(['git', '-C', str(PROJECT / 'team-code-task2-cuda'), 'diff', '--',
                                     *manifest['patched_source_sha256']])
    for name, digest in manifest['patched_source_sha256'].items():
        assert sha(PROJECT / 'team-code-task2-cuda' / name) == digest
    (scripts / 'task2_cuda.patch').write_bytes(patch)
    runner = PROJECT / 'scripts/run_task2_cuda.py'
    assert sha(runner) == manifest['launcher_sha256']
    shutil.copyfile(runner, scripts / runner.name)
    shutil.copyfile(Path(__file__), scripts / Path(__file__).name)

    # Choose operating sizes using each fold's saved validation scores only.
    primary = SOURCE / MAIN
    validation = {}
    members = {}
    for fold in range(6):
        marker = read(primary / f'fold_{fold}/complete.json')
        validation[fold] = marker['validation_subject_equal_macro_f1_13']
        members[fold] = np.array(marker['participants']['test']) - 1
    tolerances = (0.005, 0.01, 0.02)
    chosen = {t: {fold: min(int(k) for k, v in values.items() if v >= values['17'] - t - 1e-12)
                  for fold, values in validation.items()} for t in tolerances}

    rows = []
    class_rows = []
    timings = []
    comparison = {}
    matrices = {}
    curves = {}
    for spec in manifest['experiments']:
        name = spec['name']
        folder = SOURCE / name
        out = DEST / name; out.mkdir()
        for file in ALLOW:
            if (folder / file).is_file():
                shutil.copyfile(folder / file, out / file)
        val_records = []
        for fold in range(6):
            marker = read(folder / f'fold_{fold}/complete.json')
            values = marker['validation_subject_equal_macro_f1_13']
            for k, val in sorted(values.items(), key=lambda item: int(item[0])):
                val_records.append(dict(fold=fold, k=int(k), validation_subject_equal_macro_f1_13=val,
                                        validation_delta_vs_17=val - values['17']))
        pd.DataFrame(val_records).to_csv(out / 'validation_per_fold.csv', index=False)
        curve = pd.read_csv(folder / 'pareto_curve.csv')
        curves[name] = curve
        raw_summary = read(folder / 'summary.json')
        assert raw_summary['complete'] and raw_summary['config']['device_type'] == 'cuda'
        assert all(all(row['anti_leakage_audit'].values()) for row in raw_summary['pareto'])
        matrix = np.zeros((17, 30, 13, 13), dtype=np.int64)
        recount(folder, matrix, manifest['expected_oof_windows_per_k'])
        matrices[name] = matrix
        comparison[name] = {}
        baseline_f1 = f1(matrix[16].sum(axis=0))
        baseline = scores(matrix[16])
        for index, k in enumerate(range(1, 18)):
            score = scores(matrix[index])
            old = raw_summary['pareto'][index]
            assert old['k'] == k and np.isclose(score['pooled'], old['pooled_macro_f1_13'], atol=1e-12, rtol=0)
            assert np.isclose(score['equal'], old['subject_equal_macro_f1_13'], atol=1e-12, rtol=0)
            assert np.isclose(score['accuracy'], old['accuracy_13'], atol=1e-12, rtol=0)
            delta = {f'{key}_delta': score[key] - baseline[key] for key in score}
            rows.append(dict(experiment=name, k=k, **score, **delta))
            cm = matrix[index].sum(axis=0)
            class_f1 = f1(cm)
            for cid, activity in enumerate(CLASSES):
                class_rows.append(dict(experiment=name, k=k, class_id=cid, activity=activity,
                                       support=int(cm[cid].sum()), true_positive=int(cm[cid, cid]),
                                       predicted_count=int(cm[:, cid].sum()), f1=float(class_f1[cid]),
                                       delta_vs_17=float(class_f1[cid] - baseline_f1[cid])))
            if k in (1, 3, 4, 5, 6, 8, 10, 17):
                comparison[name][str(k)] = dict(**score, **delta, fold_sd_equal=old['subject_equal_macro_f1_13_fold_sd'],
                    **(paired_ci(matrix[index], matrix[16]) if name == MAIN else {}))
        log = SOURCE / 'logs' / f'{name}.log'
        times = [float(match[1]) for line in log.read_text().splitlines()
                 if (match := re.fullmatch(r'FIT_DONE k=\d+ seconds=(\d+\.\d+)', line))]
        assert len(times) == 6 * spec['fits_per_fold']
        timings.append(dict(experiment=name, fits=len(times), summed_fit_log_seconds=sum(times)))
        print('VERIFIED', name, len(times), flush=True)

    pd.DataFrame(rows).to_csv(DEST / 'all_curves.csv', index=False)
    class_table = pd.DataFrame(class_rows)
    class_table.to_csv(DEST / 'class_metrics.csv', index=False)
    pd.DataFrame(timings).to_csv(DEST / 'timing_summary.csv', index=False)
    matrix = matrices[MAIN]
    layouts = read(primary / 'per_fold_selected_subsets.json')['folds']
    canonical = read(primary / 'config.json')['parameters']['sensors']
    consensus = {}
    for k in range(1, 18):
        sets = [set(layouts[str(i)][str(k)]) for i in range(6)]
        frequencies = Counter(sensor for subset in sets for sensor in subset)
        jaccard = [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
        consensus[str(k)] = dict(frequency=dict(frequencies), mean_pairwise_jaccard=float(np.mean(jaccard)),
                                 identical_layouts=len({frozenset(s) for s in sets}) == 1)
    policy_results = []
    for tolerance, selected in chosen.items():
        heldout = np.zeros_like(matrix[16])
        for fold, k in selected.items():
            heldout[members[fold]] = matrix[k - 1, members[fold]]
        assert sum(heldout.flat) == manifest['expected_oof_windows_per_k']
        policy_results.append(dict(validation_loss_budget=tolerance, chosen_k_by_fold=selected,
                                   median_k=float(np.median(list(selected.values()))),
                                   **scores(heldout), **paired_ci(heldout, matrix[16])))
    primary_k = int(np.ceil(policy_results[1]['median_k']))
    counts = Counter(s for fold in layouts.values() for s in fold[str(primary_k)])
    # Earliest size at which a site has survived each backward path, then canonical order.
    persistence = {s: sum(min(int(k) for k, sites in fold.items() if s in sites) for fold in layouts.values())
                   for s in canonical}
    ordered = sorted(canonical, key=lambda s: (-counts[s], persistence[s], canonical.index(s)))
    recommendation = dict(k=primary_k, sites=ordered[:primary_k], selections_per_site={s: counts[s] for s in ordered},
        rule='Ceiling of the median of fold-local 1-point validation policies, then descending vote count at that size, persistence and canonical order for ties.',
        evidence_scope='Consensus prototype. Reported results refer to fold-selected layouts, not this single fixed prototype.')
    changes = {}
    for k in sorted({primary_k, 3, 5, 6, 8, 10}):
        table = class_table[(class_table.experiment == MAIN) & (class_table.k == k)]
        changes[str(k)] = table.nsmallest(3, 'delta_vs_17')[['activity', 'f1', 'delta_vs_17']].to_dict('records')
    write(DEST / 'analysis_summary.json', dict(comparison=comparison, runtime=timings, consensus=consensus,
          validation_policies=policy_results, prototype=recommendation, largest_class_drops=changes,
          started_utc=stage['started_utc'], completed_utc=stage['updated_utc'],
          duration_seconds=(datetime.fromisoformat(stage['updated_utc']) - datetime.fromisoformat(stage['started_utc'])).total_seconds(),
          recomputed=True, windows_per_count=manifest['expected_oof_windows_per_k'], participants=30,
          recovery_note='Each fold prediction coordinate and label matching against k=17 was checked.'))
    plot(curves, DEST)
    receipt = {p.relative_to(DEST).as_posix(): sha(p) for p in DEST.rglob('*') if p.is_file()}
    write(DEST / 'source_export_checksums.json', receipt)
    archive = DEST.with_suffix('.zip')
    with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
        for path in sorted(DEST.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(DEST).as_posix())
    print('AGGREGATE_EXPORT', archive, sha(archive), len(receipt) + 1, flush=True)


if __name__ == '__main__':
    main()
