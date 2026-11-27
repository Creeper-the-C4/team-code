import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'diagnostics'))

import diag_accel_unit  # noqa: E402
import diag_orientation_independence  # noqa: E402
import diag_subject_protocol  # noqa: E402
from common import G, SENSORS, SubjectData  # noqa: E402
from make_synthetic_fixture import build  # noqa: E402


@pytest.fixture(scope='module')
def g_root(tmp_path_factory):
    return build(tmp_path_factory.mktemp('g'), subjects=(1, 2, 3), acc_unit='g', seed=1)


@pytest.fixture(scope='module')
def mps2_root(tmp_path_factory):
    return build(tmp_path_factory.mktemp('mps2'), subjects=(1, 2), acc_unit='mps2', seed=2)


def test_fixture_matches_loader_layout(g_root):
    d = SubjectData(g_root, 1)
    assert d.signals.shape[1:] == (len(SENSORS), 10)
    values, labels = d.read_trial(int(d.trials.trial.iloc[0]), SENSORS)
    assert values.shape[0] == labels.shape[0]
    q = values[:, :, 6:10]
    assert np.all(np.sum(q[1:] * q[:-1], axis=2) >= 0)          # loader made signs continuous


def test_accel_unit_g(g_root):
    rep = diag_accel_unit.run(str(g_root))
    assert rep['decision']['verdict'] == 'g'
    assert abs(rep['T1_flight_plateau']['median'] + 1.0) < 0.05
    assert 0.30 < rep['T2_implied_m']['g']['stairs_rise'] < 0.38          # 2 x 0.17 m
    assert 0.25 < rep['T3_knee_lever_arms_raw']['r1']['median'] * G < 0.45


def test_accel_unit_mps2(mps2_root):
    rep = diag_accel_unit.run(str(mps2_root))
    assert rep['decision']['verdict'] == 'mps2'
    assert abs(rep['T1_flight_plateau']['median'] + G) < 0.5


def test_accel_unit_never_flips_under_heavy_smoothing(tmp_path):
    root = build(tmp_path, subjects=(1,), acc_unit='g', smooth_frames=15, seed=3)
    rep = diag_accel_unit.run(str(root))
    assert rep['decision']['verdict'] in ('g', 'unresolved')


def test_orientation_independence_flags_hinge_perfect_quaternions(tmp_path):
    root = build(tmp_path, subjects=(1,), wobble=0.6, seed=4)
    rep = diag_orientation_independence.run(str(root))
    knee = rep['hinge']['Right_knee']
    assert knee['quat_ratio']['median'] < 0.05
    assert knee['skeleton_constrained_suspected']
    assert rep['gyro_vs_quat_residual']['RightThigh']['r2']['median'] > 0.9


def test_subject_protocol_cadence_and_split_audit(g_root, tmp_path):
    rep = diag_subject_protocol.run(str(g_root), str(g_root / 'splits.json'))
    assert rep['split_audit']['ok']
    assert rep['non_monotonic_subjects'] == []
    meta = json.loads((g_root / 'fixture.json').read_text())
    for c in rep['walking_cadence'].values():
        f = c['moderate_walking']['step_freq_hz']['median']
        assert 0.85 * meta['cadence_steps_per_s']['5'] < f < 1.15 * meta['cadence_steps_per_s']['5']
    bad = json.loads((g_root / 'splits.json').read_text())
    bad['folds'][0]['train'].append(bad['folds'][0]['test'][0])
    (tmp_path / 'bad.json').write_text(json.dumps(bad))
    assert not diag_subject_protocol.run(str(g_root), str(tmp_path / 'bad.json'))['split_audit']['ok']
