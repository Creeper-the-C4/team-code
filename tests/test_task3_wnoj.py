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
                            normal_blocks, registered_factors, smooth)
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


def test_accel_unit_gate_follows_p0_verdict():
    for unit in ('none', 'disabled', 'g'):          # P0 verdict 'g' unlocks T3-M3
        check_accel_unit(unit)
    with pytest.raises(NotImplementedError):
        check_accel_unit('mps2')                     # contradicts the P0 verdict
    with pytest.raises(ValueError):
        check_accel_unit('m/s2')
    prob, _, r, w, a = synthetic_problem(60)
    with pytest.raises(NotImplementedError):
        smooth(prob, r, w, a, accel_unit='mps2')
    assert 'accel_link' not in registered_factors(prob)


def test_accel_factor_jacobian_matches_finite_differences():
    from task3.accel_factor import AccelFactor, jacobian, residual as acc_res
    rng = np.random.default_rng(11)
    t = 7
    r = exp_so3(rng.normal(size=(t, 3)))
    w = rng.normal(0, 3, (t, 3))
    a = rng.normal(0, 20, (t, 3))
    f = AccelFactor(rng.normal(0, 5, (t, 3)), np.array([0.02, -0.4, 0.05]), 0.5)
    jac = jacobian(r, w, a, f)
    h = 1e-6
    for c in range(9):
        d = np.zeros((t, 9))
        d[:, c] = h
        num = (acc_res(*perturb(r, w, a, d), f) - acc_res(*perturb(r, w, a, -d), f)) / (2 * h)
        assert np.max(np.abs(num - jac[:, :, c])) < 1e-8


def test_solvers_agree_with_accel_factor():
    from task3.accel_factor import AccelFactor
    prob, _, r, w, a = synthetic_problem(150)
    rng = np.random.default_rng(12)
    prob.accel.append(AccelFactor(rng.normal(0, 2, (150, 3)), np.array([0, -0.4, 0.0]), 2.0))
    j, res = linear_system(prob, r, w, a)
    dg, off, g, cost = normal_blocks(prob, r, w, a)
    assert np.max(np.abs((j.T @ j).toarray() - blocks_to_sparse(dg, off).toarray())) < 1e-9 * dg.max()
    d = {s: gauss_newton_step(prob, r, w, a, solver=s)[0] for s in ('banded', 'sparse', 'dense')}
    for s in ('banded', 'sparse'):
        assert np.linalg.norm(d[s] - d['dense']) / np.linalg.norm(d['dense']) < 1e-9
    assert 'accel_link' in registered_factors(prob)


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


def test_acceleration_factor_is_never_registered_or_read():
    from task3 import smoother
    from task3.withheld_eval import smooth_observed
    assert smoother.ACCEL_FACTOR_REGISTERED is False
    assert 'acceleration' not in smoother.FACTOR_TYPES
    values, labels, names, _ = chain_trial(seed=4)
    train = [{k: v[::5] for k, v in rotations(chain_trial(seed=5)[0], names).items()}]
    stats = joint_statistics(train)
    obs = [n for n in names if n != 'RightShank']
    smoother.SOLVER_LOG.clear()
    a = evaluate_chunk(values, labels, names, obs, ['RightShank'], stats, Params())
    no_acc = values.copy()
    no_acc[:, :, 0:3] = np.nan                   # accelerometer channels unusable everywhere
    b = evaluate_chunk(no_acc, labels, names, obs, ['RightShank'], stats, Params())
    for key in a['v1']:
        assert np.array_equal(a['v1'][key], b['v1'][key])
        assert np.array_equal(a['v2'][key], b['v2'][key])
    assert smoother.SOLVER_LOG and all(not entry[2] for entry in smoother.SOLVER_LOG)
    sm = smooth_observed(values[:, [names.index('RightThigh')]], ['RightThigh'], ['RightThigh'],
                         Params())
    assert set(sm) == {'RightThigh'}


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
    assert summary['accel_factor_registered'] is False
    assert 0 <= summary['solver']['failure_rate'] <= summary['solver']['convergence_rate'] + 1
    assert summary['solver']['max_iterations'] >= 1
    assert summary['input_masks']['0']['H1:RightShank'] == [
        s for s in summary['folds'][0]['schemes']['H1:RightShank']['observed']]
    for f in ('per_sensor_metrics.csv', 'per_activity_metrics.csv', 'per_subject_metrics.csv',
              'observed_residuals.csv', 'ablation_summary.csv', 'README.md'):
        assert (out / f).exists()
    header = (out / 'per_sensor_metrics.csv').read_text().splitlines()[0]
    for col in ('rmse_deg', 'median_deg', 'p95_deg', 'input_mask'):
        assert col in header
    again = subprocess.run(cmd, capture_output=True, text=True)
    assert again.returncode == 0 and 'skipping' in again.stdout
    bad = subprocess.run(cmd + ['--accel-unit', 'mps2', '--out', str(tmp_path / 'm')],
                         capture_output=True, text=True)
    assert bad.returncode != 0 and 'verdict' in bad.stderr


def test_cli_h3_reads_only_its_own_fold_layout(fixture_root, tmp_path):
    from independent_data_access import SENSORS
    layouts = {'0': {'3': ['LowerBack', 'RightThigh', 'RightFoot'],
                     '4': ['LowerBack', 'RightThigh', 'RightFoot', 'Head']},
               '1': {'3': ['UpperBack', 'LeftThigh', 'LeftFoot']}}
    sub = tmp_path / 'per_fold_selected_subsets.json'
    sub.write_text(json.dumps({'family': 'F1H', 'folds': layouts}))
    out = tmp_path / 'h3'
    subprocess.run([sys.executable, str(REPO / 'scripts' / 'task3' / 'run_task3_wnoj.py'),
                    '--data-root', str(fixture_root), '--splits', str(fixture_root / 'splits.json'),
                    '--out', str(out), '--folds', '0', '1', '--schemes', 'H3',
                    '--fold-subsets-json', str(sub), '--h3-k', '3', '--no-calibrate',
                    '--no-ablation', '--max-trials-per-activity', '1',
                    '--max-seconds-per-trial', '10', '--workers', '1', '--accel-unit', 'disabled'],
                   check=True, capture_output=True)
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['accel_unit'] == 'none' and summary['accel_unit_requested'] == 'disabled'
    for fold in ('0', '1'):
        assert summary['input_masks'][fold] == {'H3:k3': layouts[fold]['3']}
        info = next(f for f in summary['folds'] if str(f['fold']) == fold)
        withheld = info['schemes']['H3:k3']['withheld']
        assert set(withheld) == set(SENSORS) - set(layouts[fold]['3'])
        assert not set(withheld) & set(info['schemes']['H3:k3']['observed'])


def test_wnoj_kinematic_consistency_against_ode():
    """Verify xi_dot = Jl_inv(xi) w = w - (xi x w)/2 + O(||xi||^2) via direct ODE integration."""
    from scipy.integrate import solve_ivp
    from task3 import so3_lie as so3
    from task3 import wnoj_prior as wnoj
    w0 = np.array([1.2, -0.7, 0.5])
    a0 = np.array([-0.8, 1.5, 2.0])
    r0 = so3.exp_so3(np.array([0.3, -0.4, 0.2]))
    t_end, h = 0.20, 1e-6
    sol = solve_ivp(lambda t, y: (so3.hat(w0 + a0 * t) @ y.reshape(3, 3)).ravel(),
                    [0.0, t_end + h], r0.ravel(), t_eval=[t_end - h, t_end, t_end + h],
                    rtol=1e-13, atol=1e-14)
    rm, r1, rp = [sol.y[:, i].reshape(3, 3) for i in range(3)]
    xi_m = so3.log_so3(rm @ r0.T)
    xi = so3.log_so3(r1 @ r0.T)
    xi_p = so3.log_so3(rp @ r0.T)
    xi_dot_exact = (xi_p - xi_m) / (2.0 * h)
    w1 = w0 + a0 * t_end
    # Exact Jl_inv(xi) @ w1 matches ODE derivative to integration precision
    assert np.linalg.norm(so3.jl_inv(xi) @ w1 - xi_dot_exact) < 1e-8
    # First-order approximation used in wnoj_prior.residual: w1 - 0.5 * (xi x w1)
    err_minus = np.linalg.norm((w1 - 0.5 * np.cross(xi, w1)) - xi_dot_exact)
    err_plus = np.linalg.norm((w1 + 0.5 * np.cross(xi, w1)) - xi_dot_exact)
    assert err_minus < 5e-3
    assert err_plus > 20.0 * err_minus
    # At 60 Hz (dt = 1/60), WNOJ residual e2 with exact ODE trajectory is O(dt^2)
    dt = 1.0 / 60.0
    sol_dt = solve_ivp(lambda t, y: (so3.hat(w0 + a0 * t) @ y.reshape(3, 3)).ravel(),
                       [0.0, dt], r0.ravel(), t_eval=[dt], rtol=1e-13, atol=1e-14)
    r_dt = sol_dt.y[:, 0].reshape(3, 3)
    w_dt = w0 + a0 * dt
    e, _ = wnoj.residual(r0[None], w0[None], a0[None], r_dt[None], w_dt[None], a0[None], dt)
    bound = 0.5 * np.linalg.norm(w0) * np.linalg.norm(a0) * dt ** 2      # O(dt^2) local term
    assert np.linalg.norm(e[0, 3:6]) < bound
    half = solve_ivp(lambda t, y: (so3.hat(w0 + a0 * t) @ y.reshape(3, 3)).ravel(),
                     [0.0, dt / 2], r0.ravel(), t_eval=[dt / 2], rtol=1e-13, atol=1e-14)
    e_half, _ = wnoj.residual(r0[None], w0[None], a0[None], half.y[:, 0].reshape(3, 3)[None],
                              (w0 + a0 * dt / 2)[None], a0[None], dt / 2)
    ratio = np.linalg.norm(e[0, 3:6]) / np.linalg.norm(e_half[0, 3:6])
    assert 3.6 < ratio < 4.4                                            # halving dt -> 1/4


def test_fold_subset_parsing_rejects_invalid_entries(tmp_path):
    import run_task3_wnoj as cli
    bad_unknown = tmp_path / 'unknown.json'
    bad_unknown.write_text(json.dumps({'folds': {'0': {'2': ['LowerBack', 'NotASensor']}}}))
    with pytest.raises(ValueError, match='unknown sensors'):
        cli.parse_fold_subsets(bad_unknown, 0, [2])
    bad_dup = tmp_path / 'dup.json'
    bad_dup.write_text(json.dumps({'folds': {'0': {'2': ['LowerBack', 'LowerBack']}}}))
    with pytest.raises(ValueError, match='duplicate'):
        cli.parse_fold_subsets(bad_dup, 0, [2])
    bad_len = tmp_path / 'len.json'
    bad_len.write_text(json.dumps({'folds': {'0': {'3': ['LowerBack', 'Head']}}}))
    with pytest.raises(ValueError, match='has 2 sensors'):
        cli.parse_fold_subsets(bad_len, 0, [3])


# ----------------------------------------------------------------------------- T3-M3 / T3-M5
def sandwich_chain(n=600, seed=21):
    """Thigh (observed) - shank (withheld) - foot (observed) with consistent free accelerations."""
    from independent_data_access import SENSORS
    from task3.accel_factor import ACCEL_G_TO_SI
    rng = np.random.default_rng(seed)
    t = np.arange(n) * DT
    z = 0 * t
    rp = exp_so3(np.stack([0.5 * np.sin(2 * np.pi * t), z, z], 1))
    rw = rp @ exp_so3(np.stack([0.8 + 0.6 * np.sin(2 * np.pi * t + 0.5), z, z], 1))
    rc = rw @ exp_so3(np.stack([0.4 * np.sin(2 * np.pi * 1.3 * t), z, z], 1))
    bones = {'RightThigh': np.array([0, -0.45, 0.0]), 'RightShank': np.array([0, -0.42, 0.0]),
             'RightFoot': np.array([0, 0, 0.20])}
    knee = np.einsum('nij,j->ni', rp, bones['RightThigh'])
    ankle = knee + np.einsum('nij,j->ni', rw, bones['RightShank'])
    pos = {'RightThigh': 0.5 * knee, 'RightShank': knee + 0.5 * (ankle - knee),
           'RightFoot': ankle + 0.5 * np.einsum('nij,j->ni', rc, bones['RightFoot'])}
    rots = {s: np.repeat(np.eye(3)[None], n, 0) for s in SENSORS}
    rots.update({'RightThigh': rp, 'RightShank': rw, 'RightFoot': rc})
    values = np.zeros((n, len(SENSORS), 10))
    for i, s in enumerate(SENSORS):
        p = pos.get(s, np.zeros((n, 3)))
        acc = np.gradient(np.gradient(p, DT, axis=0), DT, axis=0)
        values[:, i, 0:3] = (acc + rng.normal(0, 0.05, acc.shape)) / ACCEL_G_TO_SI
        values[:, i, 3:6] = initial_rates(rots[s], DT)[0] + rng.normal(0, 0.02, (n, 3))
        noisy = exp_so3(rng.normal(0, np.deg2rad(0.5), (n, 3))) @ rots[s]
        values[:, i, 6:10] = hemisphere_continuity(rot_to_quat_xyzw(noisy))
    return values, SENSORS, rots, bones


def test_t3m3_link_acceleration_corrects_a_biased_joint_prior():
    from task3.withheld_eval import EDGES, MODEL_IDS
    values, names, rots, bones = sandwich_chain()
    stats = {e: (np.eye(3), np.full(3, np.deg2rad(10.0))) for e in EDGES}
    stats[('RightThigh', 'RightShank')] = (exp_so3(np.array([0.3, 0, 0])), np.full(3, np.deg2rad(25)))
    stats[('RightShank', 'RightFoot')] = (exp_so3(np.array([-0.3, 0, 0])), np.full(3, np.deg2rad(25)))
    obs = [s for s in names if s != 'RightShank']
    labels = np.full(len(values), 5)
    out = evaluate_chunk(values, labels, names, obs, ['RightShank'], stats,
                         Params(accel_unit='g', sigma_acc=1.0), bones)
    m2 = np.median(out['v1'][('wnoj', 'RightShank')])
    m3 = np.median(out['v1'][('wnoj_acc', 'RightShank')])
    assert m3 < 0.7 * m2
    assert MODEL_IDS['wnoj_acc'] == 'T3-M3' and MODEL_IDS['wnoj'] == 'T3-M2'
    # default mode never reads acceleration: NaN accelerometer channels change nothing
    nan_acc = values.copy()
    nan_acc[:, :, 0:3] = np.nan
    a = evaluate_chunk(values, labels, names, obs, ['RightShank'], stats, Params(), bones)
    b = evaluate_chunk(nan_acc, labels, names, obs, ['RightShank'], stats, Params(), bones)
    assert ('wnoj_acc', 'RightShank') not in a['v1']
    assert np.array_equal(a['v1'][('wnoj', 'RightShank')], b['v1'][('wnoj', 'RightShank')])


def test_t3m3_skips_unsandwiched_segments_safely():
    values, names, _, bones = sandwich_chain(n=240)
    stats = {e: (np.eye(3), np.full(3, np.deg2rad(10.0))) for e in __import__(
        'task3.withheld_eval', fromlist=['EDGES']).EDGES}
    obs = [s for s in names if s not in ('RightShank', 'RightFoot')]   # no observed child
    out = evaluate_chunk(values, np.full(240, 5), names, obs, ['RightShank', 'RightFoot'],
                         stats, Params(accel_unit='g'), bones)
    assert np.array_equal(out['v1'][('wnoj', 'RightShank')], out['v1'][('wnoj_acc', 'RightShank')])
    out2 = evaluate_chunk(values, np.full(240, 5), names, [s for s in names if s != 'RightShank'],
                          ['RightShank'], stats, Params(accel_unit='g'), bones=None)
    assert np.array_equal(out2['v1'][('wnoj', 'RightShank')], out2['v1'][('wnoj_acc', 'RightShank')])


def test_smooth_with_external_prior_interface():
    from task3.withheld_eval import smooth_with_external_prior
    prob, r_true, r_meas, _, _ = synthetic_problem(300)
    out = smooth_with_external_prior({'RightShank': r_meas}, None, sigma_prior_deg=2.0)
    r, w = out['RightShank']
    assert r.shape == r_meas.shape and w.shape == (300, 3)
    assert np.median(geodesic_deg(r, r_true)) < np.median(geodesic_deg(r_meas, r_true))


def test_cli_accel_g_reports_t3m2_and_t3m3_side_by_side(fixture_root, tmp_path):
    out = tmp_path / 'g'
    subprocess.run([sys.executable, str(REPO / 'scripts' / 'task3' / 'run_task3_wnoj.py'),
                    '--data-root', str(fixture_root), '--splits', str(fixture_root / 'splits.json'),
                    '--body-dims', str(fixture_root / 'dims.csv'), '--out', str(out),
                    '--folds', '0', '--h1-sensors', 'RightShank', '--max-trials-per-activity', '1',
                    '--max-seconds-per-trial', '10', '--workers', '1', '--no-ablation',
                    '--accel-unit', 'g'], check=True, capture_output=True)
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['accel_unit'] == 'g' and summary['accel_factor_registered'] is True
    assert summary['accel_schema']['ACCEL_G_TO_SI'] == 9.80665
    assert summary['accel_schema']['gravity_removed'] is True
    assert summary['folds'][0]['sigma_acc'] in (1.0, 2.0, 4.0)
    rows = (out / 'per_sensor_metrics.csv').read_text()
    assert 'T3-M2' in rows and 'T3-M3' in rows and 'T3-M0' in rows
