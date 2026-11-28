import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts' / 'diagnostics'))
sys.path.insert(0, str(REPO / 'scripts' / 'spine'))

import calibrate_noise_from_p0 as cal  # noqa: E402
from make_synthetic_fixture import build  # noqa: E402


@pytest.fixture(scope='module')
def root(tmp_path_factory):
    r = build(tmp_path_factory.mktemp('cal'), subjects=(1, 2, 3), acc_unit='g', seed=31)
    (r / 'dims.csv').write_text('subject,torso_length_m\n1,0.50\n2,0.52\n3,0.48\n')
    return r


def test_robust_noise_recovers_planted_sd(root):
    # the fixture adds N(0, 0.05 m/s^2) per axis, stored in g; X and Y carry no sway
    summary, _ = cal.run(str(root), str(root / 'dims.csv'), with_supplement=False)
    for sensor in ('LowerBack', 'UpperBack'):
        for ax in ('X', 'Y'):
            med = summary['sensors'][sensor]['robust_sd_si'][ax]['median']
            assert 0.85 * 0.05 < med < 1.15 * 0.05
    assert summary['n_subjects'] == 3


def test_kappa_definition_and_si_scaling(root):
    summary, table = cal.run(str(root), str(root / 'dims.csv'), taus=(1.0, 2.0),
                             with_supplement=False)
    k1 = summary['kappa_nominal']['1']['median']
    k2 = summary['kappa_nominal']['2']['median']
    assert abs(k2 / k1 - 4.0) < 1e-6                         # kappa scales with tau^2
    cov_g = np.array(summary['sensors']['LowerBack']['cov_g2_mean'])
    cov_si = np.array(summary['sensors']['LowerBack']['cov_si2_mean'])
    assert np.allclose(cov_si, cov_g * 9.80665 ** 2)
    assert any(r['quantity'] == 'kappa_nominal' for r in table)


def test_outputs_are_aggregate_only_and_supplement_present(root, tmp_path):
    summary, table = cal.run(str(root), str(root / 'dims.csv'))
    cal.write(tmp_path / 'out', summary, table)
    text = ''.join((tmp_path / 'out' / f).read_text() for f in
                   ('calibration_summary.json', 'calibration_table.csv', 'README.md'))
    for sid in ('P01', 'P02', 'P03', '"subject"'):
        assert sid not in text
    sup = json.loads((tmp_path / 'out' / 'calibration_summary.json').read_text())['supplementary_wp3']
    assert sup['C0_orientation_only'] == 18 and sup['C1_orientation_only'] == 36
    assert all(v >= 18 for v in sup['C0_plus_acc_by_kappa'].values())
