"""Check rotation geometry, withheld-channel isolation, fold roles, CUDA and the CLI."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.preprocessing import independent_data_access as access
from scripts.task3 import run_task3_wnoj as stage_a
from scripts.task3_dl.train_bigru_6d import main
from src.task3_dl import dataset
from src.task3_dl.evaluation import layouts_for_fold, predict_layouts, wnoj_post, world_rates
from src.task3_dl.model import SensorDropoutBiGRU, reconstruction_loss
from src.task3_dl.rotation_6d import geodesic_loss, matrix_to_rot6d, rot6d_to_matrix

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def trial_data(frames, seed=2):
    rng = np.random.default_rng(seed)
    values = rng.normal(0, .2, (frames, 17, 10)).astype("float32")
    phase = np.arange(frames)[:, None] * .012 + np.arange(17) * .03 + seed * .01
    values[..., 6:10] = 0
    values[..., 7] = np.sin(phase / 2)
    values[..., 9] = np.cos(phase / 2)
    values[1::2, :, 6:10] *= -1
    return values


@pytest.mark.parametrize("device", DEVICES)
def test_rot6d_orthonormality_and_geodesic_grad(device):
    torch.manual_seed(1)
    angles = torch.tensor([0., np.pi - 1e-5, np.pi], device=device)
    c, s = angles.cos(), angles.sin()
    z = torch.zeros_like(c)
    exact = torch.stack((c, -s, z, s, c, z, z, z, z + 1), dim=-1).reshape(-1, 3, 3)
    bad = torch.tensor([[0., 0., 0., 0., 0., 0.], [1., 0., 0., 2., 0., 0.]], device=device)
    x = torch.cat([torch.randn(64, 6, device=device), matrix_to_rot6d(exact), bad]).requires_grad_()
    r = rot6d_to_matrix(x)
    eye = torch.eye(3, device=device).expand_as(r)
    assert (r.transpose(-1, -2) @ r - eye).abs().max().detach().item() < 1e-6
    torch.testing.assert_close(torch.linalg.det(r), torch.ones_like(r[:, 0, 0]), rtol=0, atol=8e-7)
    torch.testing.assert_close(r[64:67], exact)
    geodesic_loss(r, eye).backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_masked_sensors_invariant_to_nan_and_noise(device):
    values = trial_data(24)
    rng = np.random.default_rng(4)
    mask = np.isin(access.SENSORS, ["LowerBack", "Head", "RightFoot"])
    scaler = dataset.Scaler(rng.normal(size=(17, 6)).astype("float32"), np.full((17, 6), 2, dtype="float32"))
    model = SensorDropoutBiGRU(hidden=16, layers=1).eval().to(device)
    saved = dataset.build_inputs(values, mask, scaler)
    reference = predict_layouts(model, values, [mask], scaler, device)
    encoded = saved.reshape(len(values), 17, 13)
    assert torch.count_nonzero(encoded[:, ~mask]) == 0
    assert torch.all(encoded[:, mask, -1] == 1)
    for replacement in (np.nan, rng.normal(0, 1e6, values[:, ~mask].shape)):
        corrupted = values.copy()
        corrupted[:, ~mask] = replacement
        assert torch.equal(dataset.build_inputs(corrupted, mask, scaler), saved)
        np.testing.assert_array_equal(predict_layouts(model, corrupted, [mask], scaler, device), reference)


def test_scaler_fit_training_only_and_si_scaling(monkeypatch):
    seen = []
    data = trial_data(160)
    class Source:
        def __init__(self, root, subject):
            seen.append(subject)
            assert subject == 2
            self.signals = data
            self.labels = np.zeros(len(data))
    monkeypatch.setattr(access, "SubjectData", Source)
    fold = {"train": [2], "validation": [1], "test": [3]}
    scaler = dataset.fit_scaler(None, fold)
    units = np.array([9.80665] * 3 + [1.] * 3)
    data_si = data[..., :6].astype(float) * units
    np.testing.assert_allclose(scaler.mean, data_si.mean(0), rtol=1e-6)
    np.testing.assert_allclose(scaler.std, data_si.std(0), rtol=1e-6)
    inp = dataset.build_inputs(data, np.ones(17, bool), scaler).reshape(len(data), 17, 13).numpy()
    np.testing.assert_allclose(inp[..., :6], (data_si - data_si.mean(0)) / data_si.std(0), atol=5e-7)
    assert seen == [2]
    with pytest.raises(ValueError, match="held-out"):
        dataset.fit_scaler(None, dict(fold, validation=[2]))


def test_structured_dropout_and_rates():
    rng = np.random.default_rng(15)
    layouts = [dataset.sample_mask(rng) for _ in range(700)]
    assert set(range(3, 17)) <= {m.sum() for m in layouts}
    assert all(any(set(np.flatnonzero(~m)) == set(limb) for m in layouts) for limb in dataset.LIMBS)
    r = dataset.rotation_targets(trial_data(25))
    w = world_rates(r)
    np.testing.assert_allclose(w[:, :, 1], .72, atol=2e-5)
    np.testing.assert_allclose(w[:, :, [0, 2]], 0, atol=1e-6)
    post, rate = wnoj_post(r[:, :1], [access.SENSORS[0]])
    np.testing.assert_allclose(post, r[:, :1], atol=1e-5)
    np.testing.assert_allclose(rate, w[:, :1], atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_bigru_cuda_forward_backward_smoke():
    torch.manual_seed(16)
    model = SensorDropoutBiGRU(dropout=0.).to("cuda").train()
    params = sum(p.numel() for p in model.parameters())
    assert 1_000_000 < params < 4_000_000
    values = trial_data(24)
    mask = np.arange(17) < 4
    scaler = dataset.Scaler(np.zeros((17, 6)), np.ones((17, 6)))
    x = dataset.build_inputs(values, mask, scaler)[None].cuda()
    target = torch.from_numpy(dataset.rotation_targets(values))[None].cuda()
    observed = torch.from_numpy(mask)[None].cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    losses = []
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        loss = reconstruction_loss(model(x), target, observed)
        losses.append(float(loss.detach()))
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
    assert losses[-1] < losses[0]
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | {params} parameters | "
          f"training loss {losses[0]:.6f} -> {losses[-1]:.6f} rad")


def test_h3_layout_validation():
    names = ["RightFoot", "LeftFoot", "Head"]
    obj = {"folds": {"0": {"3": names}, "1": {"3": names[::-1]}}}
    assert layouts_for_fold(1, ["H3"], [], obj, [3]) == {"H3:k3": names[::-1]}
    for selection in ([names[0]] * 3, ["other", *names[1:]], names[:2]):
        with pytest.raises(ValueError):
            layouts_for_fold(0, ["H3"], [], {"folds": {"0": {"3": selection}}}, [3])
    with pytest.raises(ValueError):
        layouts_for_fold(0, ["H3"], [], obj, [3, 4])


@pytest.mark.parametrize("index_format", ["in_memory", "parquet"])
def test_h3_layout_validation_and_cli_synthetic(tmp_path, monkeypatch, index_format):
    if index_format == "parquet":
        pytest.importorskip("pyarrow")
    root = tmp_path / "independent_v1"
    tables = {}
    for subject in [1, 2, 3]:
        folder = root / "subjects" / f"P{subject:02d}"
        folder.mkdir(parents=True)
        (folder / "complete.json").write_text("{}")
        data = trial_data(596, subject)
        labels = np.repeat([3, 4, 5], [135, 280, 181]).astype("int8")
        labels[20:35] = 0
        np.save(folder / "signals.npy", data)
        np.save(folder / "labels.npy", labels)
        frame = pd.DataFrame({"trial": [0, 1, 2], "source_start": [0, 135, 415],
                              "source_stop_exclusive": [135, 415, 596], "majority_label": [3, 4, 5]})
        path = folder / "trials.parquet"
        if index_format == "parquet":
            frame.to_parquet(path, index=False)
        tables[path] = frame
    if index_format == "in_memory":
        monkeypatch.setattr(pd, "read_parquet", lambda path: tables[Path(path)].copy())
    canonical = list(stage_a.subject_chunks(root, 3, 2, 30., 2.))
    actual = list(dataset.test_chunks(access.SubjectData(root, 3), 2, 30., 2.))
    assert len(actual) == len(canonical) == 4
    for (v, label), (_, expected, expected_label) in zip(actual, canonical):
        np.testing.assert_array_equal(v[..., :6], expected[..., :6])
        np.testing.assert_array_equal(label, expected_label)
        np.testing.assert_allclose(dataset.rotation_targets(v), dataset.rotation_targets(expected), atol=3e-7)
    folds = [{"fold": i, "train": [1 + i % 3], "validation": [1 + (i + 1) % 3],
              "test": [1 + (i + 2) % 3]} for i in range(3)]
    split = tmp_path / "splits.json"
    split.write_text(json.dumps({"folds": folds}))
    subsets = tmp_path / "layouts.json"
    subsets.write_text(json.dumps({"folds": {"0": {"15": access.SENSORS[:-2]}}}))
    out = tmp_path / "public"
    dims = tmp_path / "dims.csv"
    columns = ["head_height_m", "neck_length_m", "torso_length_m", "shoulder_width_m", "upper_arm_length_m",
               "forearm_length_m", "bi_iliac_breadth_m", "thigh_length_m", "shank_length_m", "foot_length_m"]
    pd.DataFrame([{"subject": 3, **dict.fromkeys(columns, .3)}]).to_csv(dims, index=False)
    args = ["--data-root", str(root), "--splits", str(split), "--out", str(out), "--folds", "0",
            "--fast-smoke", "--epochs", "1", "--fold-subsets-json", str(subsets),
            "--schemes", "H1", "H3", "--h1-sensors", "Head", "--subset-k", "15", "--chunk-seconds", "2",
            "--body-dims", str(dims), "--apply-wnoj-post"]
    main(args)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["reference_type"] == "withheld_sensor_reference"
    assert summary["accel_g_to_si"] == 9.80665
    assert summary["input_masks"]["0"]["H3:k15"] == access.SENSORS[:-2]
    assert summary["folds"][0]["best_epoch"] == 1
    assert summary["folds"][0]["calibrated_v3_subject_count"] == 1
    assert set(pd.read_csv(out / "v3_endpoint_metrics.csv").method) == {"T3-M4", "T3-M5"}
    metrics = pd.read_csv(out / "per_sensor_metrics.csv")
    assert set(metrics.method) == {"T3-M4", "T3-M5"}
    assert (metrics.n_frames == 480).all()
    assert set(metrics.withheld_sensor) == {"Head", *access.SENSORS[-2:]}
    assert np.isfinite(metrics[["median_deg", "rmse_deg", "p95_deg", "subject_equal_median_deg", "v2_rmse_rad_s"]]).all().all()
    assert set(pd.read_csv(out / "per_activity_metrics.csv").activity) == {"Others", "standing", "slow_walking", "moderate_walking"}
    assert not list(out.rglob("*.pt")) and list((tmp_path / "checkpoints").rglob("best.pt"))
    assert not any("subject" in f.name for f in out.iterdir())
    before = (out / "summary.json").stat().st_mtime_ns
    main(args + ["--resume"])
    assert (out / "summary.json").stat().st_mtime_ns == before
    with pytest.raises(ValueError, match="Output exists"):
        main(args + ["--resume", "--seed", "42"])
