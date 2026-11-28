"""Isolated, resumable Task 1 V4 run. Does not execute Task 2 or Task 3."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import datetime
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback

PROJECT = Path('/root/autodl-tmp/C_Problem')
REPO = PROJECT / 'team-code'
DATA = PROJECT / 'data/processed/independent_v1'
CACHE = PROJECT / 'data/processed/v4_features_cache'
ROOT = PROJECT / 'results/task1_v4_20261128'
PY = PROJECT / '.venv/bin/python'
EXPECTED = '46bc8a1056461a3980a75d51869c4a676d5dbfcc'
SPECS = [('F1H', 2), ('F1W', 2), ('F1B', 2), ('F2p', 2),
         ('F3', 2), ('F3', 4), ('F4', 2), ('F4', 4)]
FAMILIES = {2: ['F1H', 'F1W', 'F1B', 'F2', 'F2p', 'F3', 'F4'],
            4: ['F1H', 'F3', 'F4']}
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='2', PYTHONUNBUFFERED='1',
                  PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
sys.path.insert(0, str(REPO))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def extract(job):
    from src.v4_features.workflow import build_subject_cache
    subject, seconds = job
    started = time.monotonic()
    count = build_subject_cache(DATA, CACHE, subject, seconds, FAMILIES[seconds])
    return {'subject': subject, 'window_seconds': seconds, 'windows': count,
            'elapsed_seconds': time.monotonic() - started}


def preflight():
    import numpy as np
    import pandas as pd
    from src.v4_features.feature_families import feature_names
    from src.v4_features.workflow import load_folds
    from scripts.preprocessing.independent_data_access import SENSORS
    head = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    if head != EXPECTED:
        raise RuntimeError('Pinned source commit changed')
    if subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain',
                                '--untracked-files=no'], text=True).strip():
        raise RuntimeError('Tracked source changes detected')
    splits = DATA / 'subject_splits_6fold.json'
    load_folds(splits)
    if splits.read_bytes() != (REPO / 'results/preprocessing/independent_v1' / splits.name).read_bytes():
        raise RuntimeError('Repository/cache split mismatch')
    counts, inputs = {}, {}
    for subject in range(1, 31):
        folder = DATA / 'subjects' / f'P{subject:02d}'
        signals = np.load(folder / 'signals.npy', mmap_mode='r')
        if signals.ndim != 3 or signals.shape[1:] != (17, 10):
            raise RuntimeError('Signal shape mismatch')
        for seconds in FAMILIES:
            table = folder / f'windows_{seconds}s.parquet'
            counts[subject, seconds] = len(pd.read_parquet(table, columns=['subject']))
            inputs[str(table.relative_to(DATA))] = sha(table)
    estimated = sum(counts[s, ws] * sum(len(feature_names(f, SENSORS)) for f in families) * 4
                    for ws, families in FAMILIES.items() for s in range(1, 31))
    free = shutil.disk_usage(PROJECT).free
    existing_cache = sum(p.stat().st_size for p in CACHE.rglob('features.npy')) if CACHE.exists() else 0
    if free < max(0, estimated - existing_cache) + 2 * 1024**3:
        raise RuntimeError('Insufficient disk for caches plus 2 GiB working reserve')
    source_paths = ['src/v4_features/feature_families.py', 'src/v4_features/workflow.py',
                    'src/v4_features/sticky_hmm.py', 'scripts/task1_v4/train_task1_v4.py',
                    'scripts/task1_v4/extract_v4_features.py',
                    'scripts/preprocessing/independent_data_access.py']
    versions = {}
    for name in ['numpy', 'pandas', 'scipy', 'sklearn', 'lightgbm', 'pyarrow', 'pytest']:
        module = __import__(name)
        versions[name] = module.__version__
    manifest = {'started_utc': now(), 'source_commit': head, 'scope': 'Task 1 only',
                'specifications': [{'family': f, 'window_seconds': ws} for f, ws in SPECS],
                'data_root': str(DATA), 'feature_cache': str(CACHE), 'source_files_sha256':
                {name: sha(REPO / name) for name in source_paths},
                'runner_sha256': sha(__file__), 'model_saving_wrapper_sha256':
                sha(PROJECT / 'scripts/task1_v4_train_with_models.py'),
                'split_sha256': sha(splits), 'window_indices_sha256': inputs,
                'expected_oof_windows': {str(ws): sum(counts[s, ws] for s in range(1, 31))
                                         for ws in FAMILIES},
                'estimated_feature_bytes': estimated, 'initial_free_bytes': free,
                'extraction_workers': 6, 'training_jobs_concurrent': 1,
                'lightgbm_threads_per_model': 12, 'maximum_boosting_rounds': 300,
                'hmm_fit': 'inner training subjects only; offline trial-wise Viterbi',
                'selection': 'fixed eight upstream configurations; no test-driven tuning',
                'limitations': ['Fixed inner validation, not full nested CV.',
                    'OOF ablation comparison is exploratory; no untouched confirmation set.',
                    'Viterbi uses future windows and known trial boundaries; not online inference.',
                    '2s/4s grids differ; their pooled metrics are not paired window comparisons.'],
                'environment': {'python': sys.version, 'platform': platform.platform(), 'versions': versions,
                    'container_limits': {name: Path('/sys/fs/cgroup', name).read_text().strip()
                        for name in ['cpu.max', 'memory.max'] if Path('/sys/fs/cgroup', name).is_file()}},
                'precheck': {'pytest': '14 passed, including synthetic real-codec/LightGBM smoke',
                            'raw_signals_modified': False}}
    return manifest


def validate_cache():
    import numpy as np
    from src.v4_features.workflow import _cache_location, _cache_spec, _valid_cache, window_table
    from src.v4_features.feature_families import feature_names, _normalized, _rotation, _spectrum
    from scripts.preprocessing.independent_data_access import SENSORS
    rows = []
    for seconds, families in FAMILIES.items():
        for subject in range(1, 31):
            source, windows = window_table(DATA, subject, seconds)
            for family in families:
                directory = _cache_location(CACHE, family, subject, seconds)
                spec = _cache_spec(family, subject, seconds, source, len(windows))
                if not _valid_cache(directory, spec):
                    raise RuntimeError(f'Invalid feature cache: {directory}')
                values = np.load(directory / 'features.npy', mmap_mode='r').reshape(len(windows), -1)
                bad_columns = np.isnan(values).any(axis=0)
                names = feature_names(family, SENSORS)
                if np.isinf(values).any() or any(
                    'dominant_frequency_hz' not in names[i]
                    and not names[i].endswith('.kinematics.step_frequency_hz')
                    for i in np.flatnonzero(bad_columns)
                ):
                    raise RuntimeError(f'Unexpected nonfinite features: {directory}')
                if family == 'F4':
                    for thigh in ['LeftThigh', 'RightThigh']:
                        step_freq = values[:, names.index(thigh + '.kinematics.step_frequency_hz')]
                        missing = np.flatnonzero(np.isnan(step_freq))
                        if len(missing):
                            raw = np.load(DATA / 'subjects' / f'P{subject:02d}' / 'signals.npy',
                                          mmap_mode='r')
                            starts = windows['source_start'].to_numpy()
                            sensor = SENSORS.index(thigh)
                            for offset in range(0, len(missing), 128):
                                idx = missing[offset:offset + 128]
                                frames = starts[idx, None] + np.arange(seconds * 60)
                                batch = raw[frames, sensor]
                                rotation = _rotation(_normalized(batch[..., 6:].astype(np.float64)))
                                gyro = np.einsum('btij,bti->btj', rotation,
                                                 batch[..., 3:6].astype(np.float64))
                                recomputed = _spectrum(gyro[:, :, :1], 60)[:, 0, 0]
                                if not np.isnan(recomputed).all():
                                    raise RuntimeError(f'Unexpected missing F4 step frequency: {directory}')
                if family == 'F3' and seconds == 2:
                    old = np.load(DATA / 'subjects' / f'P{subject:02d}' / 'baseline_features_2s.npy',
                                  mmap_mode='r').reshape(len(windows), -1)
                    idx = np.linspace(0, len(windows) - 1, min(64, len(windows)), dtype=int)
                    if not np.array_equal(values[idx], old[idx], equal_nan=True):
                        raise RuntimeError(f'F3 baseline alignment failed: P{subject:02d}')
                rows.append({'subject': subject, 'window_seconds': seconds, 'family': family,
                             'windows': len(windows), 'features': values.shape[1],
                             'nan_count': int(np.isnan(values).sum())})
    write(ROOT / 'feature_cache_audit.json', {'complete': True, 'caches': rows,
          'f4_step_frequency_nan_check': 'all missing step frequencies recomputed from raw body-frame gyro',
          'f3_baseline_comparison': '64 evenly spaced windows per participant, exact equality including NaN'})


def validate_outputs(family, seconds, expected_windows):
    import numpy as np
    import pandas as pd
    from src.v4_features.workflow import load_folds, score_predictions
    out = ROOT / 'task1_v4' / f'{family}_{seconds}s'
    summary = json.loads((out / 'summary.json').read_text())
    if not summary.get('complete') or summary['config']['fast_smoke']:
        raise RuntimeError('Training incomplete or smoke result in formal output')
    frame = pd.read_parquet(out / 'oof_predictions.parquet')
    if len(frame) != expected_windows or set(frame.subject) != set(range(1, 31)):
        raise RuntimeError('Incomplete OOF population')
    if frame.duplicated(['subject', 'trial', 'source_start']).any():
        raise RuntimeError('Duplicate OOF predictions')
    folds = load_folds(DATA / 'subject_splits_6fold.json')
    for fold in folds:
        folder = out / f"fold_{fold['fold']}"
        params = json.loads((folder / 'hmm_parameters.json').read_text())
        if params['fitting_subjects'] != fold['train'] or not (folder / 'model.txt').is_file():
            raise RuntimeError('Fold HMM isolation or model artifact check failed')
        if set(frame.loc[frame.fold == fold['fold'], 'subject']) != set(fold['test']):
            raise RuntimeError('OOF fold membership mismatch')
    for mode, column in [('raw', 'y_pred'), ('hmm', 'y_pred_hmm')]:
        computed = score_predictions(frame, frame[column])[0]
        for key in ['subject_equal_macro_f1_13', 'pooled_macro_f1_13', 'accuracy_13', 'walk3_macro_f1']:
            if not np.isclose(computed[key], summary[mode][key], atol=1e-12, rtol=0):
                raise RuntimeError('Reported score mismatch')
    del frame
    gc.collect()
    return summary


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock_file = (ROOT / '.runner.lock').open('a')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (ROOT / 'logs').mkdir(exist_ok=True)
    manifest = preflight()
    old = ROOT / 'run_manifest.json'
    if old.exists():
        previous = json.loads(old.read_text())
        for key in ['source_commit', 'source_files_sha256', 'runner_sha256', 'split_sha256',
                    'window_indices_sha256', 'specifications']:
            if previous[key] != manifest[key]:
                raise RuntimeError('Existing run fingerprint differs: ' + key)
        manifest = previous
    else:
        write(old, manifest)
    started = time.monotonic()
    state = {'complete': False, 'updated_utc': now(), 'stage': 'feature_extraction',
             'finished_feature_jobs': [], 'experiments_complete': []}
    write(ROOT / 'stage_status.json', state)
    print('TASK1_V4_START ' + json.dumps({'source': EXPECTED,
          'feature_GiB': manifest['estimated_feature_bytes'] / 1024**3,
          'initial_free_GiB': manifest['initial_free_bytes'] / 1024**3}), flush=True)
    jobs = [(s, ws) for ws in FAMILIES for s in range(1, 31)]
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(extract, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            state['finished_feature_jobs'].append(row)
            state['updated_utc'] = now()
            write(ROOT / 'stage_status.json', state)
            print('FEATURE_DONE ' + json.dumps(row), flush=True)
    for seconds, families in FAMILIES.items():
        rows = [r for r in state['finished_feature_jobs'] if r['window_seconds'] == seconds]
        write(CACHE / f'{seconds}s' / 'complete.json', {'complete': True,
              'window_seconds': seconds, 'families': families, 'participants': rows,
              'total_windows': sum(r['windows'] for r in rows)})
    state['stage'] = 'feature_validation'
    write(ROOT / 'stage_status.json', state)
    validate_cache()
    for family, seconds in SPECS:
        state['stage'] = 'training'
        state['current_experiment'] = f'{family}_{seconds}s'
        state['updated_utc'] = now()
        write(ROOT / 'stage_status.json', state)
        if shutil.disk_usage(PROJECT).free < 1024**3:
            raise RuntimeError('Free disk below 1 GiB reserve')
        cmd = [PY, PROJECT / 'scripts/task1_v4_train_with_models.py', '--data-root', DATA,
               '--feature-cache', CACHE, '--splits', DATA / 'subject_splits_6fold.json',
               '--family', family, '--window-sec', seconds, '--enable-sticky-hmm',
               '--out', ROOT / 'task1_v4' / f'{family}_{seconds}s']
        log_path = ROOT / 'logs' / f'{family}_{seconds}s.log'
        print(f'EXPERIMENT_START {family}_{seconds}s', flush=True)
        before = time.monotonic()
        with log_path.open('a', encoding='utf-8', buffering=1) as log:
            subprocess.run(list(map(str, cmd)), cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = validate_outputs(family, seconds, manifest['expected_oof_windows'][str(seconds)])
        state['experiments_complete'].append({'experiment': f'{family}_{seconds}s',
            'elapsed_seconds': time.monotonic() - before, 'raw': result['raw'], 'hmm': result['hmm']})
        state['updated_utc'] = now()
        write(ROOT / 'stage_status.json', state)
        print(f'EXPERIMENT_DONE {family}_{seconds}s', flush=True)
    # Confirm all source window-index fingerprints stayed unchanged.
    if any(sha(DATA / name) != value for name, value in manifest['window_indices_sha256'].items()):
        raise RuntimeError('Input window index changed during execution')
    state.update(complete=True, stage='complete', updated_utc=now(),
                 elapsed_seconds=time.monotonic() - started)
    write(ROOT / 'stage_status.json', state)
    write(ROOT / 'comparison_summary.json', {'complete': True, 'source_commit': EXPECTED,
          'experiments': state['experiments_complete'], 'limitations': manifest['limitations']})
    print('TASK1_V4_COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        ROOT.mkdir(parents=True, exist_ok=True)
        write(ROOT / 'failure.json', {'failed_utc': now(), 'traceback': traceback.format_exc()})
        raise
