import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.preprocessing.independent_data_access import SENSORS
from scripts.preprocessing.independent_features import sensor_features
from scripts.task1_v4 import extract_v4_features, train_task1_v4
from scripts.task2_v4 import run_sensor_selection
from src.v4_features import feature_families
from src.v4_features.feature_families import extract_features, feature_dependencies, feature_names
from src.v4_features.sticky_hmm import fit_sticky_hmm, viterbi_decode_trials
from src.v4_features.workflow import feature_columns, load_folds


def sample_signals():
    rng = np.random.default_rng(127)
    values = rng.normal(size=(3, 120, len(SENSORS), 10))
    values[..., 6:] /= np.linalg.norm(values[..., 6:], axis=-1, keepdims=True)
    return values


def test_f1h_does_not_read_orientation():
    values = sample_signals()
    expected = extract_features(values, "F1H", SENSORS)
    assert expected.shape == (3, 17, 24)
    values[..., 6:] = np.nan
    assert np.array_equal(expected, extract_features(values, "F1H", SENSORS))
    values[..., 6:] = np.random.default_rng(94).normal(size=values[..., 6:].shape)
    assert np.array_equal(expected, extract_features(values, "F1H", SENSORS))


def test_world_and_full_match_existing_baseline():
    values = sample_signals()
    expected = sensor_features(values)
    assert np.array_equal(expected[..., :32], extract_features(values, "F1W", SENSORS))
    assert np.array_equal(expected, extract_features(values, "F3", SENSORS))
    assert extract_features(values, "F1B", SENSORS).shape == (3, 17, 32)
    assert np.array_equal(expected[..., 32:], extract_features(values, "F2", SENSORS))


@pytest.mark.parametrize("heading", [0.2, -1.3, 2.1, np.pi])
def test_heading_invariant_families(heading):
    values = sample_signals()
    yaw = np.array([[np.cos(heading), 0, np.sin(heading)], [0, 1, 0],
                    [-np.sin(heading), 0, np.cos(heading)]])
    changed = values.copy()
    for span in (slice(0, 3), slice(3, 6)):
        changed[..., span] = values[..., span] @ yaw.T
    x, y, z, real = np.moveaxis(values[..., 6:], -1, 0)
    sin, cos = np.sin(heading / 2), np.cos(heading / 2)
    changed[..., 6:] = np.stack([cos*x + sin*z, cos*y + sin*real,
                                  cos*z - sin*x, cos*real - sin*y], axis=-1)
    for family in ("F1H", "F1B", "F2p"):
        np.testing.assert_allclose(extract_features(values, family, SENSORS),
                                   extract_features(changed, family, SENSORS), atol=1e-5)


def test_f4_dependency_mask_matches_physical_subset():
    values = sample_signals()
    sensors = ["UpperBack", "LeftThigh", "LeftShank"]
    indices = [SENSORS.index(sensor) for sensor in sensors]
    subset = extract_features(values[:, :, indices], "F4", sensors)
    names = feature_names("F4", sensors)
    assert subset.shape == (3, 127)  # 3 * 40, one pair and one thigh.
    assert list(feature_dependencies("F4", sensors)) == names
    assert all(set(needed) <= set(sensors) for needed in feature_dependencies("F4", sensors).values())
    full = extract_features(values, "F4", SENSORS)
    np.testing.assert_array_equal(subset, full[:, feature_columns("F4", sensors)])


def test_f4_rejects_unavailable_dependency(monkeypatch):
    actual = feature_families.feature_dependencies

    def invalid(family, sensor_names):
        result = actual(family, sensor_names)
        result[next(iter(result))] = ("Head",)
        return result

    monkeypatch.setattr(feature_families, "feature_dependencies", invalid)
    with pytest.raises(ValueError, match="Unavailable dependency"):
        extract_features(sample_signals()[:, :, :1], "F4", ["LowerBack"])


def test_hmm_only_learns_transitions_inside_training_trials():
    train = pd.DataFrame({
        "subject": [1] * 90, "trial": np.repeat([1, 2, 3], 30),
        "source_start": np.arange(90) * 60, "majority_label": np.repeat([0, 1, 2], 30),
    })
    params = fit_sticky_hmm(train, dt_seconds=1, n_classes=3)
    np.testing.assert_allclose(params["tau_seconds"], [30, 30, 30])
    np.testing.assert_allclose(params["transition"][0, 1], params["transition"][0, 2])
    np.testing.assert_allclose(np.diag(params["transition"]), np.exp(-1 / 30))
    test = pd.DataFrame({
        "subject": [2, 2, 2], "trial": [20, 10, 10], "source_start": [120, 0, 60],
        "p_0": [0.40, 0.99, 0.99], "p_1": [0.59, 0.005, 0.005],
        "p_2": [0.01, 0.005, 0.005], "y_true": [2, 2, 2],
    })
    prediction = viterbi_decode_trials(test, ["p_0", "p_1", "p_2"], params)
    np.testing.assert_array_equal(prediction, [1, 0, 0])
    test["y_true"] = 1
    np.testing.assert_array_equal(viterbi_decode_trials(test, ["p_0", "p_1", "p_2"], params), prediction)
    test["trial"] = 10
    np.testing.assert_array_equal(viterbi_decode_trials(test, ["p_0", "p_1", "p_2"], params), [0, 0, 0])


def test_fold_validation_rejects_shared_participants(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"folds": [{"fold": 0, "train": [1, 2],
                                         "validation": [2], "test": [3]}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Roles overlap"):
        load_folds(path, fast_smoke=True)


def test_synthetic_end_to_end_two_folds(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "diagnostics"))
    fixture = importlib.import_module("scripts.diagnostics.make_synthetic_fixture")
    dataset = fixture.build(tmp_path / "independent_v1", subjects=(1, 2, 3))
    cache = tmp_path / "cache"
    splits = dataset / "splits.json"
    for seconds in ("2", "4"):
        extract_v4_features.main([
            "--data-root", str(dataset), "--out-cache", str(cache), "--window-sec", seconds,
            "--families", "F1H", "F1W", "F1B", "F2", "F2p", "F3", "--subjects", "1", "2", "3",
        ])
    f1h = np.load(cache / "4s" / "F1H" / "P01" / "features.npy")
    assert f1h.shape[1:] == (17, 24)
    task1_out = tmp_path / "task1"
    common = ["--data-root", str(dataset), "--feature-cache", str(cache), "--splits", str(splits),
              "--fast-smoke"]
    task1_args = common + ["--family", "F4", "--window-sec", "2", "--enable-sticky-hmm",
                           "--out", str(task1_out)]
    train_task1_v4.main(task1_args)
    summary = json.loads((task1_out / "summary.json").read_text(encoding="utf-8"))
    for mode in ("raw", "hmm"):
        assert 0 <= summary[mode]["subject_equal_macro_f1_13"] <= 1
        assert 0 <= summary[mode]["pooled_macro_f1_13"] <= 1
    for fold in load_folds(splits, fast_smoke=True):
        hmm = json.loads((task1_out / f"fold_{fold['fold']}" / "hmm_parameters.json").read_text())
        assert hmm["fitting_subjects"] == fold["train"]
        assert not set(hmm["fitting_subjects"]) & set(fold["validation"] + fold["test"])
    before = (task1_out / "summary.json").stat().st_mtime_ns
    train_task1_v4.main(task1_args)
    assert (task1_out / "summary.json").stat().st_mtime_ns == before
    task2_out = tmp_path / "task2"
    run_sensor_selection.main(common + ["--out", str(task2_out), "--sensors", "LowerBack", "RightThigh",
                                         "RightShank", "--n-estimators", "3"])
    subsets = json.loads((task2_out / "per_fold_selected_subsets.json").read_text())
    assert subsets["family"] == "F1H"
    assert set(subsets["folds"]) == {"0", "1"}
    for layout in subsets["folds"].values():
        assert set(layout) == {"1", "2", "3"}
        assert all(len(layout[str(size)]) == size for size in (1, 2, 3))
        assert set(layout["1"]) < set(layout["2"]) < set(layout["3"])
    assert pd.read_csv(task2_out / "pareto_curve.csv")["k"].tolist() == [1, 2, 3]
