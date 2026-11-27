import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts' / 'diagnostics'))
sys.path.insert(0, str(REPO / 'scripts' / 'preprocessing'))
sys.path.insert(0, str(REPO / 'scripts' / 'task3'))

from task3.so3_lie import (exp_so3, geodesic_deg, hemisphere_continuity, jl, jl_inv,  # noqa: E402
                           jr, jr_inv, log_so3, quat_xyzw_to_rot, rot_to_quat_xyzw)
from task3.smoother import (OrientationFactor, SegmentProblem, blocks_to_sparse,  # noqa: E402
                            check_accel_unit, gauss_newton_step, initial_rates, linear_system,
                            normal_blocks, smooth)
from task3.wnoj_prior import (orientation_residual, phi_matrix, q_matrix, residual,  # noqa: E402
                              residual_jacobians, whitener)
from task3.withheld_eval import (Params, evaluate_chunk, joint_statistics,  # noqa: E402
                                 rebuild_order, rotations)

RNG = np.random.default_rng(3)
DT = 1 / 60


# ----------------------------------------------------------------------------- SO(3)
def test_exp_log_roundtrip_including_near_pi():
    axes = RNG.normal(size=(500, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    for angles in (RNG.uniform(0, 3.0, (500, 1)), RNG.uniform(np.pi - 1e-3, np.pi - 1e-7, (500, 1)),
                   RNG.uniform(0, 1e-6, (500, 1))):
        phi = axes * angles
        r = exp_so3(phi)
        assert np.allclose(r @ np.swapaxes(r, -1, -2), np.eye(3), atol=1e-12)
        assert np.max(np.abs(exp_so3(log_so3(r)) - r)) < 1e-8


@pytest.mark.parametrize('scale', [1e-8, 1e-5, 1e-2, 1.0, 2.5])
def test_right_left_jacobian_inverses(scale):
    phi = RNG.normal(size=(200, 3)) * scale
    assert np.max(np.abs(jr(phi) @ jr_inv(phi) - np.eye(3))) < 1e-10
    assert np.max(np.abs(jl(phi) @ jl_inv(phi) - np.eye(3))) < 1e-10


def test_right_jacobian_definition():
    phi = RNG.normal(size=(50, 3))
    d = RNG.normal(size=(50, 3)) * 1e-6
    lhs = exp_so3(phi + d)
    rhs = exp_so3(phi) @ exp_so3(np.einsum('nij,nj->ni', jr(phi), d))
    assert np.max(np.abs(lhs - rhs)) < 1e-10


def test_quaternion_roundtrip_and_continuity():
    r = exp_so3(RNG.normal(size=(300, 3)))
    q = rot_to_quat_xyzw(r)
    assert np.max(np.abs(quat_xyzw_to_rot(q) - r)) < 1e-12
    flipped = q * np.where(RNG.random(300) < 0.3, -1, 1)[:, None]
    cont = hemisphere_continuity(flipped)
    assert np.all(np.sum(cont[1:] * cont[:-1], axis=1) >= 0)
    assert np.max(np.abs(quat_xyzw_to_rot(cont) - r)) < 1e-12


# ----------------------------------------------------------------------------- WNOJ prior
def random_pair(k=6):
    r0 = exp_so3(RNG.normal(size=(k, 3)))
    r1 = exp_so3(RNG.normal(0, 0.05, (k, 3))) @ r0
    return (r0, *(RNG.normal(0, 2, (k, 3)) for _ in range(2)),
            r1, *(RNG.normal(0, 2, (k, 3)) for _ in range(2)))


def perturb(r, w, a, d):
    return exp_so3(d[:, 0:3]) @ r, w + d[:, 3:6], a + d[:, 6:9]


def test_wnoj_jacobians_match_finite_differences():
    r0, w0, a0, r1, w1, a1 = random_pair()
    _, j0, j1 = residual_jacobians(r0, w0, a0, r1, w1, a1, DT)
    h = 1e-6
    for which, jac in ((0, j0), (1, j1)):
        for c in range(9):
            d = np.zeros((r0.shape[0], 9))
            d[:, c] = h
            if which == 0:
                ep = residual(*perturb(r0, w0, a0, d), r1, w1, a1, DT)[0]
                em = residual(*perturb(r0, w0, a0, -d), r1, w1, a1, DT)[0]
            else:
                ep = residual(r0, w0, a0, *perturb(r1, w1, a1, d), DT)[0]
                em = residual(r0, w0, a0, *perturb(r1, w1, a1, -d), DT)[0]
            num = (ep - em) / (2 * h)
            assert np.max(np.abs(num - jac[:, :, c])) < 1e-8


def test_orientation_factor_jacobian_matches_finite_differences():
    rt = exp_so3(RNG.normal(size=(6, 3)))
    r = exp_so3(RNG.normal(0, 0.3, (6, 3))) @ rt
    _, jac = orientation_residual(rt, r)
    h = 1e-6
    for c in range(3):
        d = np.zeros((6, 3))
        d[:, c] = h
        num = (orientation_residual(rt, exp_so3(d) @ r)[0] -
               orientation_residual(rt, exp_so3(-d) @ r)[0]) / (2 * h)
        assert np.max(np.abs(num - jac[:, :, c])) < 1e-8


def test_wnoj_noise_model():
    q = q_matrix(DT, 7.0)
    assert np.all(np.linalg.eigvalsh(q) > 0)
    w = whitener(DT, 7.0)
    assert np.allclose(w @ q @ w.T, np.eye(9), atol=1e-8)
    p = phi_matrix(DT)
    assert np.allclose(p[0:3, 6:9], 0.5 * DT ** 2 * np.eye(3))


# ----------------------------------------------------------------------------- solver
def synthetic_problem(n=300, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * DT
    r_true = exp_so3(np.stack([0.6 * np.sin(2 * np.pi * 0.8 * t), 0.3 * np.sin(t), 0.2 * t], 1))
    r_meas = exp_so3(rng.normal(0, np.deg2rad(2), (n, 3))) @ r_true
    w0, a0 = initial_rates(r_meas, DT)
    prob = SegmentProblem(dt=DT, qc=1e3, gyro=initial_rates(r_true, DT)[0] +
                          rng.normal(0, 0.05, (n, 3)),
                          orientation=[OrientationFactor(r_meas, np.deg2rad(2))])
    return prob, r_true, r_meas, w0, a0


def test_block_normal_equations_equal_jtj():
    prob, _, r, w, a = synthetic_problem(120)
    j, res = linear_system(prob, r, w, a)
    dg, off, g, cost = normal_blocks(prob, r, w, a)
    h = (j.T @ j).toarray()
    assert np.max(np.abs(h - blocks_to_sparse(dg, off).toarray())) < 1e-12 * np.abs(h).max()
    assert np.max(np.abs(j.T @ res - g.ravel())) < 1e-10 * np.abs(g).max()
    assert abs(cost - res @ res) < 1e-10 * cost


def test_sparse_banded_and_dense_solutions_agree():
    prob, _, r, w, a = synthetic_problem(200)
    d = {s: gauss_newton_step(prob, r, w, a, solver=s)[0] for s in ('banded', 'sparse', 'dense')}
    for s in ('banded', 'sparse'):
        assert np.max(np.abs(d[s] - d['dense'])) < 1e-9
        assert np.linalg.norm(d[s] - d['dense']) / np.linalg.norm(d['dense']) < 1e-9


def test_linear_complexity_of_normal_matrix():
    nnz = []
    for n in (120, 360, 720, 1800):
        prob, _, r, w, a = synthetic_problem(n)
        dg, off, _, _ = normal_blocks(prob, r, w, a)
        nnz.append(blocks_to_sparse(dg, off).nnz)
    per_frame = np.array(nnz) / np.array([120, 360, 720, 1800])
    assert np.ptp(per_frame) / per_frame.mean() < 0.01      # nnz grows linearly in T


def test_smoother_reduces_orientation_noise():
    prob, r_true, r_meas, w0, a0 = synthetic_problem(400)
    r, w, a, info = smooth(prob, r_meas, w0, a0)
    assert info['cost'][-1] < info['cost'][0]
    assert np.median(geodesic_deg(r, r_true)) < 0.75 * np.median(geodesic_deg(r_meas, r_true))


def test_accel_unit_is_gated():
    check_accel_unit('none')
    check_accel_unit('disabled')
    for unit in ('g', 'mps2'):
        with pytest.raises(NotImplementedError):
            check_accel_unit(unit)
    with pytest.raises(ValueError):
        check_accel_unit('m/s2')
    prob, _, r, w, a = synthetic_problem(60)
    with pytest.raises(NotImplementedError):
        smooth(prob, r, w, a, accel_unit='g')


# ----------------------------------------------------------------------------- withheld
def chain_trial(n=600, seed=1):
    """Parent thigh, child shank with a flexing knee, observed foot: known truth."""
    from independent_data_access import SENSORS
    rng = np.random.default_rng(seed)
    t = np.arange(n) * DT
    rots = {s: np.repeat(np.eye(3)[None], n, 0) for s in SENSORS}
    rots['RightThigh'] = exp_so3(np.stack([0.4 * np.sin(2 * np.pi * t), 0 * t, 0 * t], 1))
    knee = 0.6 + 0.3 * np.sin(2 * np.pi * t + 0.5)
    rots['RightShank'] = rots['RightThigh'] @ exp_so3(np.stack([knee, 0 * t, 0 * t], 1))
    rots['RightFoot'] = rots['RightShank'] @ exp_so3(np.array([-0.5, 0, 0]))
    values = np.zeros((n, len(SENSORS), 10))
    for i, s in enumerate(SENSORS):
        noisy = exp_so3(rng.normal(0, np.deg2rad(1), (n, 3))) @ rots[s]
        values[:, i, 6:10] = hemisphere_continuity(rot_to_quat_xyzw(noisy))
        values[:, i, 3:6] = initial_rates(rots[s], DT)[0] + rng.normal(0, 0.02, (n, 3))
    return values, np.full(n, 5), SENSORS, rots


def test_withheld_reconstruction_beats_copy_parent_and_ignores_withheld_channels():
    values, labels, names, truth = chain_trial()
    train = [{k: v[::5] for k, v in rotations(chain_trial(seed=s)[0], names).items()}
             for s in (2, 3)]
    stats = joint_statistics(train)
    obs = [s for s in names if s != 'RightShank']
    out = evaluate_chunk(values, labels, names, obs, ['RightShank'], stats, Params())
    med = {m: np.median(out['v1'][(m, 'RightShank')]) for m in ('wnoj', 'copy_parent')}
    assert med['wnoj'] < 0.5 * med['copy_parent']
    corrupted = values.copy()
    corrupted[:, names.index('RightShank'), :] = np.nan        # withheld channels unusable
    out2 = evaluate_chunk(corrupted, labels, names, obs, ['RightShank'], stats, Params(),
                          bones=None)
    for m in ('wnoj', 'mean_relative', 'copy_parent'):
        # identical reconstruction: only the reference (now NaN) differs
        assert np.all(np.isnan(out2['v1'][(m, 'RightShank')]))
    from task3.withheld_eval import reconstruct, smooth_observed
    idx = [names.index(s) for s in obs]
    sm = smooth_observed(values[:, idx], obs, ['RightThigh', 'RightFoot'], Params())
    sm2 = smooth_observed(corrupted[:, idx], obs, ['RightThigh', 'RightFoot'], Params())
    a = reconstruct(sm, obs, ['RightShank'], stats, Params())
    b = reconstruct(sm2, obs, ['RightShank'], stats, Params())
    assert np.array_equal(a['wnoj']['RightShank'][0], b['wnoj']['RightShank'][0])


def test_rebuild_order_handles_withheld_chains():
    order = rebuild_order(['RightShoulder', 'RightUpperArm', 'RightForeArm'],
                          ['RightWrist', 'UpperBack', 'LowerBack'])
    assert set(order) == {'RightShoulder', 'RightUpperArm', 'RightForeArm'}


# ----------------------------------------------------------------------------- CLI
@pytest.fixture(scope='module')
def fixture_root(tmp_path_factory):
    from make_synthetic_fixture import build
    root = build(tmp_path_factory.mktemp('t3'), subjects=(1, 2, 3, 4), seed=9)
    cols = dict(torso_length_m=.5, neck_length_m=.1, head_height_m=.19, shoulder_width_m=.4,
                upper_arm_length_m=.28, forearm_length_m=.25, thigh_length_m=.46,
                shank_length_m=.41, foot_length_m=.26, bi_iliac_breadth_m=.26)
    lines = ['subject,' + ','.join(cols)] + [f'{s},' + ','.join(str(v) for v in cols.values())
                                            for s in (1, 2, 3, 4)]
    (root / 'dims.csv').write_text('\n'.join(lines) + '\n')
    return root


def test_fold_subset_parsing(tmp_path):
    import run_task3_wnoj as cli
    a = tmp_path / 'a.json'
    a.write_text(json.dumps({'folds': [{'fold': 0, 'selected': ['LowerBack', 'RightWrist']}]}))
    assert cli.parse_fold_subsets(a, 0, None) == {2: ['LowerBack', 'RightWrist']}
    b = tmp_path / 'b.json'
    b.write_text(json.dumps({'0': {'by_k': {'1': ['LowerBack'], '2': ['LowerBack', 'Head']}}}))
    assert cli.parse_fold_subsets(b, 0, [2]) == {2: ['LowerBack', 'Head']}
    # format written by scripts/task2_v4/run_sensor_selection.py
    c = tmp_path / 'c.json'
    c.write_text(json.dumps({'family': 'F1H', 'folds': {
        '0': {'1': ['LowerBack'], '3': ['LowerBack', 'Head', 'RightWrist'],
              '4': ['LowerBack', 'Head', 'RightWrist', 'LeftFoot']},
        '1': {'3': ['UpperBack', 'Head', 'LeftWrist']}}}))
    assert cli.parse_fold_subsets(c, 0, [3, 4]) == {3: ['LowerBack', 'Head', 'RightWrist'],
                                                  4: ['LowerBack', 'Head', 'RightWrist', 'LeftFoot']}
    assert cli.parse_fold_subsets(c, 1, [3, 4, 5]) == {3: ['UpperBack', 'Head', 'LeftWrist']}


def test_cli_end_to_end_and_resume(fixture_root, tmp_path):
    out = tmp_path / 'out'
    cmd = [sys.executable, str(REPO / 'scripts' / 'task3' / 'run_task3_wnoj.py'),
           '--data-root', str(fixture_root), '--splits', str(fixture_root / 'splits.json'),
           '--body-dims', str(fixture_root / 'dims.csv'), '--out', str(out), '--folds', '0',
           '--h1-sensors', 'RightShank', '--max-trials-per-activity', '1',
           '--max-seconds-per-trial', '10', '--workers', '1', '--observed-sensors',
           'LowerBack', 'RightThigh', 'RightFoot', 'UpperBack']
    subprocess.run(cmd, check=True, capture_output=True)
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['reference_type'] == 'withheld_sensor_reference'
    assert summary['accel_unit'] == 'none'
    assert 'H1:RightShank' in summary['metrics']
    assert any(k.startswith('H2:') for k in summary['metrics'])
    for f in ('per_activity_metrics.csv', 'per_subject_metrics.csv', 'ablation_summary.csv',
              'README.md'):
        assert (out / f).exists()
    again = subprocess.run(cmd, capture_output=True, text=True)
    assert again.returncode == 0 and 'skipping' in again.stdout
    bad = subprocess.run(cmd + ['--accel-unit', 'g', '--out', str(tmp_path / 'g')],
                         capture_output=True, text=True)
    assert bad.returncode != 0 and 'gated' in bad.stderr
