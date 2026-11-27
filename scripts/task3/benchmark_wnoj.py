"""O(T) scaling benchmark of the WNOJ segment smoother (synthetic rotation, 60 Hz).

Reports per-GN-iteration wall time (banded and sparse solvers), normal-matrix
non-zeros and unknowns for T in {120, 360, 720, 1800} frames by default.
Synthetic input is used only to time the solver.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from task3.smoother import (OrientationFactor, SegmentProblem, blocks_to_sparse,  # noqa: E402
                            gauss_newton_step, initial_rates, normal_blocks)
from task3.so3_lie import exp_so3  # noqa: E402


def problem(n, seed=0):
    rng = np.random.default_rng(seed)
    dt = 1 / 60
    t = np.arange(n) * dt
    r_true = exp_so3(np.stack([0.6 * np.sin(2 * np.pi * 0.8 * t), 0.3 * np.sin(t), 0.2 * t], 1))
    r_meas = exp_so3(rng.normal(0, np.deg2rad(2), (n, 3))) @ r_true
    w0, a0 = initial_rates(r_meas, dt)
    prob = SegmentProblem(dt=dt, qc=1e3, gyro=w0 + rng.normal(0, 0.05, (n, 3)),
                          orientation=[OrientationFactor(r_meas, np.deg2rad(2))])
    return prob, r_meas, w0, a0


def run(lengths=(120, 360, 720, 1800), repeats=5):
    rows = []
    for n in lengths:
        prob, r, w, a = problem(n)
        row = {'frames': n, 'unknowns': 9 * n}
        dg, off, _, _ = normal_blocks(prob, r, w, a)
        row['normal_matrix_nnz'] = int(blocks_to_sparse(dg, off).nnz)
        for solver in ('banded', 'sparse'):
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                gauss_newton_step(prob, r, w, a, solver=solver)
                times.append(time.perf_counter() - t0)
            row[f'{solver}_ms_per_iteration'] = 1000 * float(np.median(times))
        rows.append(row)
    base = rows[0]
    for row in rows:
        row['nnz_per_frame'] = row['normal_matrix_nnz'] / row['frames']
        row['banded_time_ratio_vs_T_ratio'] = (row['banded_ms_per_iteration'] /
                                               base['banded_ms_per_iteration']) / (row['frames'] /
                                                                                   base['frames'])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/task3_offline_v1')
    ap.add_argument('--frames', type=int, nargs='*', default=[120, 360, 720, 1800])
    args = ap.parse_args(argv)
    rows = run(args.frames)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'complexity_benchmark.json').write_text(json.dumps(rows, indent=1), encoding='utf-8')
    lines = ['| T (frames) | unknowns | H nnz | nnz / frame | banded ms/iter | sparse ms/iter | '
             '(time ratio)/(T ratio) |', '|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['frames']} | {r['unknowns']} | {r['normal_matrix_nnz']} | "
                     f"{r['nnz_per_frame']:.1f} | {r['banded_ms_per_iteration']:.2f} | "
                     f"{r['sparse_ms_per_iteration']:.2f} | {r['banded_time_ratio_vs_T_ratio']:.2f} |")
    (out / 'complexity_benchmark.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
