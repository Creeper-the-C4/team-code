import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from spine import so3  # noqa: E402
from spine.spine_information import (QOIS, analyze_static, static_information,  # noqa: E402
                                     trajectory_information)
from spine.spine_kinematics import FE, MODELS, deadlift_pose, segment_rotations  # noqa: E402
from spine.spine_observation import (CONFIGS, NotRepresentable, config_sensors,  # noqa: E402
                                     jacobian_relative_errors, orientation_jacobians,
                                     position_jacobian, sensor_position, state_dim)

RNG = np.random.default_rng(7)


def random_state(model, scale=0.3):
    return RNG.normal(0, scale, 3), RNG.normal(0, scale, (model.n_joints, 3))


def numeric_orientation_jacobian(model, phi_p, theta, seg, h=1e-6):
    d = state_dim(model)
    base = segment_rotations(model, phi_p, theta)[seg]
    jac = np.zeros((3, d))
    for k in range(d - 3):
        cols = []
        for sgn in (1.0, -1.0):
            dphi, dth = np.zeros(3), np.zeros_like(theta)
            if k < 3:
                dphi[k] = sgn * h
                r0 = so3.exp(phi_p) @ so3.exp(dphi)          # right perturbation of pelvis
                rots = [r0]
                for j in range(model.n_joints):
                    rots.append(rots[-1] @ so3.exp(theta[j]))
                r = rots[seg]
            else:
                dth.flat[k - 3] = sgn * h
                r = segment_rotations(model, phi_p, theta + dth)[seg]
            cols.append(so3.log(base.T @ r))
        jac[:, k] = (cols[0] - cols[1]) / (2 * h)
    return jac


@pytest.mark.parametrize('name', sorted(MODELS))
def test_orientation_jacobian_matches_finite_difference(name):
    model = MODELS[name]()
    for _ in range(3):
        phi_p, theta = random_state(model)
        gs = orientation_jacobians(model, theta)
        for seg in range(len(model.segments)):
            num = numeric_orientation_jacobian(model, phi_p, theta, seg)
            assert np.max(np.abs(gs[seg][:, :-3] - num[:, :-3])) < 1e-6


@pytest.mark.parametrize('name', sorted(MODELS))
def test_jacobian_relative_error_across_step_sizes(name):
    model = MODELS[name]()
    phi_p, theta = random_state(model)
    assert phi_p.dtype == np.float64 and theta.dtype == np.float64
    errs = jacobian_relative_errors(model, phi_p, theta, steps=(1e-5, 1e-6, 1e-7))
    for h, e in errs.items():
        assert e < 1e-6, (h, e)


def test_right_jacobian_matches_sympy_derivation():
    t = sp.symbols('t0:3', real=True)
    k = sp.Matrix([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    n = sp.sqrt(t[0]**2 + t[1]**2 + t[2]**2)
    r = sp.eye(3) + sp.sin(n) / n * k + (1 - sp.cos(n)) / n**2 * k * k
    for _ in range(3):
        val = RNG.normal(0, 0.5, 3)
        subs = dict(zip(t, val))
        rn = np.array(r.subs(subs).evalf(), dtype=float)
        jr = np.zeros((3, 3))
        for i in range(3):
            dr = np.array(sp.diff(r, t[i]).subs(subs).evalf(), dtype=float)
            jr[:, i] = so3.vee(rn.T @ dr)
        assert np.max(np.abs(jr - so3.right_jacobian(val))) < 1e-6


@pytest.mark.parametrize('name', sorted(MODELS))
def test_position_jacobian_matches_finite_difference(name):
    model = MODELS[name]()
    phi_p, theta = random_state(model)
    p_p = RNG.normal(0, 1, 3)
    rots = segment_rotations(model, phi_p, theta)
    gs = orientation_jacobians(model, theta)
    d, h = state_dim(model), 1e-6
    for seg in range(len(model.segments)):
        ana = position_jacobian(model, rots, gs, seg)
        num = np.zeros_like(ana)
        for k in range(d):
            vals = []
            for sgn in (1.0, -1.0):
                dphi, dth, dp = np.zeros(3), np.zeros_like(theta), np.zeros(3)
                if k < 3:
                    dphi[k] = sgn * h
                elif k < d - 3:
                    dth.flat[k - 3] = sgn * h
                else:
                    dp[k - (d - 3)] = sgn * h
                # pelvis right perturbation: Exp(phi) Exp(dphi) == Exp(phi') with phi' = Log(...)
                phi2 = so3.log(so3.exp(phi_p) @ so3.exp(dphi))
                vals.append(sensor_position(model, phi2, theta + dth, p_p + dp, seg))
            num[:, k] = (vals[0] - vals[1]) / (2 * h)
        assert np.max(np.abs(ana - num)) < 1e-6


EXPECTED_RANK = {
    ('M3', 'C0'): 3, ('M3', 'C1'): 6, ('M3', 'C1p'): 6, ('M3', 'C1pp'): 6,
    ('M3', 'C2a'): 4, ('M3', 'C2b'): 7,
    ('M2', 'C0'): 3, ('M2', 'C1'): 6, ('M2', 'C1p'): 3, ('M2', 'C1pp'): 6,
    ('M2', 'C2a'): 4, ('M2', 'C2b'): 4,
    ('M1', 'C0'): 3, ('M1', 'C1'): 3, ('M1', 'C1p'): 3, ('M1', 'C1pp'): 6,
}


@pytest.mark.parametrize('key', sorted(EXPECTED_RANK))
def test_orientation_only_rank(key):
    name, config = key
    model = MODELS[name]()
    res = analyze_static(model, config)
    n_imu = sum(s.kind == 'imu' for s in config_sensors(model, config))
    n_bend = sum(s.kind == 'bend' for s in config_sensors(model, config))
    assert res['meas']['rank'] <= 3 * (n_imu - 1) + n_bend
    assert res['meas']['rank'] == EXPECTED_RANK[key]
    meta = res['metadata']
    assert meta['numerical_rank'] == res['meas']['rank']
    assert meta['numerical_rank'] <= meta['theoretical_rank_upper_bound']
    assert meta['numerical_rank'] == meta['theoretical_rank_upper_bound']   # bound attained
    assert meta['gauge_dimension'] == 3
    assert meta['state_dimension'] == 3 + 3 * model.n_joints
    assert res['post']['rank'] == res['post']['dim']        # prior makes it full rank


def test_bend_sensor_not_representable_in_m1():
    for config in ('C2a', 'C2b'):
        with pytest.raises(NotRepresentable):
            config_sensors(MODELS['M1'](), config)


def test_information_vanishes_for_infinite_noise():
    model = MODELS['M3']()
    i0, _, _ = static_information(model, 'C1')
    big, _, _ = static_information(model, 'C1', sigma_ori_deg=2e6, sigma_bend_deg=2e6)
    assert np.max(np.abs(big)) < 1e-10 * np.max(np.abs(i0))


def test_qoi_identifiability_pattern():
    model = MODELS['M3']()
    q = {c: analyze_static(model, c)['qoi'] for c in CONFIGS}
    inf = float('inf')
    assert q['C0']['lumbar_flexion_L1_S1']['meas_only_std_deg'] == inf
    assert q['C1']['lumbar_flexion_L1_L5']['meas_only_std_deg'] < inf
    assert q['C1']['lumbosacral_flexion_L5_S1']['meas_only_std_deg'] == inf
    assert q['C1pp']['lumbosacral_flexion_L5_S1']['meas_only_std_deg'] < inf
    assert q['C1pp']['lumbar_flexion_L1_S1']['meas_only_std_deg'] == inf
    assert q['C2a']['lumbar_flexion_L1_L5']['meas_only_std_deg'] < inf
    for c in CONFIGS:
        for name in QOIS:
            v = q[c][name]
            if v is not None:
                assert v['posterior_std_deg'] <= v['prior_std_deg'] + 1e-9


def test_acceleration_adds_weak_information():
    model = MODELS['M3']()
    ori = trajectory_information(model, 'C0', n_frames=11)
    acc = trajectory_information(model, 'C0', n_frames=11, kappa=0.5)
    r_ori = np.linalg.matrix_rank(ori, tol=1e-9 * np.abs(ori).max())
    r_acc = np.linalg.matrix_rank(acc, tol=1e-9 * np.abs(acc).max())
    assert r_ori == 3 * 11
    assert r_acc > r_ori


def test_deadlift_pose_flexes_forward():
    model = MODELS['M3']()
    phi_p, theta = deadlift_pose(model, 1.0)
    assert phi_p[0] > 0 and np.all(theta[:, FE] > 0)
    up = segment_rotations(model, phi_p, theta)[-1] @ np.array([0.0, 1.0, 0.0])
    assert up[2] > 0          # trunk axis tilts toward +Z (anterior)
