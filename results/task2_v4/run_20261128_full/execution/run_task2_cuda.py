"""Run the four V4 sensor-selection experiments on the isolated CUDA backend."""
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path('/root/autodl-tmp/C_Problem')
REPO = PROJECT / 'team-code-task2-cuda'
DATA = PROJECT / 'data/processed/independent_v1'
CACHE = PROJECT / 'data/processed/v4_features_cache'
OUTPUT = PROJECT / 'results/task2_v4_20261128'
CUDA_PACKAGE = PROJECT / 'benchmarks/task2_lightgbm_cuda_4_6_0/cuda_package'
CUDA_LIBRARY = CUDA_PACKAGE / 'lightgbm/lib/lib_lightgbm.so'
SOURCE_SHA = '46bc8a1056461a3980a75d51869c4a676d5dbfcc'
PLAN = (
    ('F1H_ranking_prefix', 'F1H', 'ranking_prefix', 17),
    ('F1H_backward', 'F1H', 'backward_greedy', 153),
    ('F3_backward', 'F3', 'backward_greedy', 153),
    ('F1H_forward', 'F1H', 'forward_greedy', 153),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    sys.path.insert(0, str(REPO))
    from src.v4_features.workflow import file_sha256, load_folds, read_json, write_json

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logs = OUTPUT / 'logs'
        logs.mkdir(exist_ok=True)
        state = dict(status='running', started_utc=now(), pid=os.getpid(), current=None, completed=[])
        status_file = OUTPUT / 'stage_status.json'

        def update(**changes):
            state.update(changes, updated_utc=now())
            write_json(status_file, state)

        update()
        try:
            head = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
            if head != SOURCE_SHA:
                raise RuntimeError('Unexpected base revision for Task 2')
            splits = DATA / 'subject_splits_6fold.json'
            folds = load_folds(splits)
            if len(folds) != 6:
                raise RuntimeError('Expected six participant folds')
            window_total = read_json(CACHE / '2s/complete.json')['total_windows']
            env = os.environ.copy()
            env.update(PYTHONPATH=str(CUDA_PACKAGE), LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6',
                       OMP_NUM_THREADS='12', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1')
            source_files = ('scripts/task2_v4/run_sensor_selection.py', 'src/v4_features/workflow.py')
            manifest = dict(base_commit=head, github_main_at_launch='17bb054bcefa691128d0ca3df80b3a923bcccd82',
                            backend='cuda', gpu='NVIDIA RTX 4090', lightgbm='4.6.0', n_estimators=80,
                            splits_sha256=file_sha256(splits), feature_cache=str(CACHE),
                            cuda_library_sha256=file_sha256(CUDA_LIBRARY),
                            launcher_sha256=file_sha256(Path(__file__)),
                            patched_source_sha256={name: file_sha256(REPO / name) for name in source_files},
                            n_folds=len(folds), expected_oof_windows_per_k=window_total,
                            experiments=[dict(name=name, family=family, strategy=strategy, fits_per_fold=count)
                                         for name, family, strategy, count in PLAN])
            manifest_path = OUTPUT / 'run_manifest.json'
            if manifest_path.exists() and manifest != read_json(manifest_path):
                raise RuntimeError('Task 2 continuation requires an identical run manifest')
            write_json(manifest_path, manifest)
            subprocess.run([sys.executable, '-c',
                            'from pathlib import Path; import lightgbm as l; '
                            'assert l.__version__ == "4.6.0"; '
                            'assert Path(l.basic._LIB._name).resolve() == Path(__import__("sys").argv[1])',
                            str(CUDA_LIBRARY)], check=True, env=env)
            for name, family, strategy, _ in PLAN:
                experiment = OUTPUT / name
                logfile = logs / f'{name}.log'
                update(current=name)
                print(f'EXPERIMENT_START {name}', flush=True)
                with logfile.open('a', encoding='utf-8') as log:
                    print(f'ATTEMPT_START {now()}', file=log, flush=True)
                    subprocess.run([
                        sys.executable, str(REPO / 'scripts/task2_v4/run_sensor_selection.py'),
                        '--data-root', str(DATA), '--feature-cache', str(CACHE), '--splits', str(splits),
                        '--family', family, '--strategy', strategy, '--window-sec', '2',
                        '--device-type', 'cuda', '--n-estimators', '80', '--out', str(experiment),
                    ], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                result = read_json(experiment / 'summary.json')
                layouts = read_json(experiment / 'per_fold_selected_subsets.json')['folds']
                assert result['complete'] and result['config']['device_type'] == 'cuda'
                assert set(layouts) == {str(fold['fold']) for fold in folds}
                sizes = set(range(1, 18))
                assert {r['k'] for r in result['pareto']} == sizes
                for fold_layouts in layouts.values():
                    assert set(map(int, fold_layouts)) == sizes
                    assert all(len(set(sensors)) == int(k) for k, sensors in fold_layouts.items())
                for row in result['pareto']:
                    assert row['n_windows'] == window_total and row['n_subjects'] == 30
                    assert len(row['anti_leakage_audit']) == len(folds) and all(row['anti_leakage_audit'].values())
                state['completed'].append(name)
                update()
                print(f'EXPERIMENT_DONE {name}', flush=True)
            update(status='complete', current=None)
            print('TASK2_CUDA_COMPLETE', flush=True)
        except BaseException as error:
            update(status='failed', error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    main()
