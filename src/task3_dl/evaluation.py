"""Matched H1/H3 evaluation and exact pooled statistics for Stage B."""

from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from scripts.preprocessing.independent_data_access import SENSORS, SubjectData
from scripts.task3.run_task3_wnoj import ACTIVITY
from src.task3 import withheld_eval
from src.task3.kinematic_tree import END_EFFECTORS, bone_lengths, calibrate_bones, forward_kinematics
from src.task3.smoother import OrientationFactor, SegmentProblem, initial_rates, smooth
from src.task3.so3_lie import chordal_mean, geodesic_deg, log_so3
from .dataset import FS, build_inputs, rotation_targets, test_chunks


def layouts_for_fold(fold, schemes, h1_sensors, subsets, sizes):
    layouts = {}
    if "H1" in schemes:
        if not h1_sensors or len(set(h1_sensors)) != len(h1_sensors) or set(h1_sensors) - set(SENSORS):
            raise ValueError("H1 needs distinct known sensor names")
        for name in h1_sensors:
            layouts[f"H1:{name}"] = [s for s in SENSORS if s != name]
    if "H3" in schemes:
        if not subsets or not isinstance(subsets.get("folds"), dict):
            raise ValueError("H3 requires Task 2 per_fold_selected_subsets.json")
        entry = subsets["folds"].get(str(fold), {})
        for size in sizes:
            selected = entry.get(str(size))
            if not isinstance(selected, list) or len(selected) != size or not 0 < size < len(SENSORS):
                raise ValueError(f"Fold {fold}: missing or wrong-length H3 layout for k={size}")
            if not all(isinstance(s, str) and s in SENSORS for s in selected):
                raise ValueError(f"Fold {fold}: unknown sensor in k={size}")
            if len(set(selected)) != size:
                raise ValueError(f"Fold {fold}: duplicate sensor in k={size}")
            layouts[f"H3:k{size}"] = selected
    if not layouts:
        raise ValueError("Select at least one non-empty H1 or H3 scheme")
    return layouts


def world_rates(rotations, dt=1 / FS):
    """Centered world-frame log differences, with one-sided end frames as in Stage A."""
    steps = log_so3(rotations[1:] @ rotations[:-1].swapaxes(-2, -1)) / dt
    rates = np.empty(rotations.shape[:-2] + (3,), dtype=float)
    rates[0], rates[-1] = steps[0], steps[-1]
    rates[1:-1] = (steps[:-1] + steps[1:]) / 2
    return rates


def wnoj_post(priors, names):
    """Treat each predicted orientation as a soft WNOJ factor. No withheld measurements."""
    shared = getattr(withheld_eval, "smooth_with_external_prior", None)
    if shared is not None:
        predictions = dict(zip(names, priors.swapaxes(0, 1), strict=True))
        smoothed = shared(predictions, gyro_obs_or_none=None, dt=1 / FS, qc=1e3, sigma_prior_deg=4.)
        return tuple(np.stack([smoothed[name][i] for name in names], axis=1) for i in (0, 1))
    # Earlier Stage A releases expose the same factors and solver individually.
    output, rates = [], []
    for prior in priors.swapaxes(0, 1):
        prior = np.asarray(prior, dtype=float)
        omega, alpha = initial_rates(prior, 1 / FS)
        problem = SegmentProblem(dt=1 / FS, qc=1e3,
                                 orientation=[OrientationFactor(prior, np.deg2rad(4.0))])
        r, w, _, _ = smooth(problem, prior, omega, alpha, max_iter=30)
        output.append(r)
        rates.append(w)
    return np.stack(output, axis=1), np.stack(rates, axis=1)


@torch.inference_mode()
def predict_layouts(model, values, masks, scaler, device):
    model.eval()
    batch = torch.stack([build_inputs(values, mask, scaler) for mask in masks])
    return model(batch.to(device)).cpu().numpy()


def standing_bones(data, dims):
    """Conditional V3 uses all sensors during a recorded standing calibration."""
    rows = data.trials.loc[data.trials["majority_label"].eq(3)]
    if dims is None or rows.empty:
        return None
    row = rows.iloc[0]
    start = int(row.source_start)
    stop = min(int(row.source_stop_exclusive), start + 20 * FS)
    if stop - start < 120:
        return None
    standing = rotation_targets(data.signals[start:stop])
    rotations = {name: chordal_mean(standing[:, i]) for i, name in enumerate(SENSORS)}
    return calibrate_bones(rotations, bone_lengths(dims))


def evaluate_subject(model, scaler, device, root, subject, layouts, cfg, destination, dims=None):
    data = SubjectData(root, subject)
    bones = standing_bones(data, dims.get(subject)) if dims is not None else None
    masks = [np.isin(SENSORS, observed) for observed in layouts.values()]
    records = defaultdict(list)
    for values, labels in test_chunks(data, cfg.max_trials_per_activity,
                                      cfg.max_seconds_per_trial, cfg.chunk_seconds):
        predictions = predict_layouts(model, values, masks, scaler, device)
        # References are accessed only after every layout has made its prediction.
        reference = rotation_targets(values)
        if bones is not None:
            fk_ref = forward_kinematics(dict(zip(SENSORS, reference.swapaxes(0, 1))), bones)
        for scheme, mask, prediction in zip(layouts, masks, predictions):
            indices = np.flatnonzero(~mask)
            rotations = prediction[:, indices].astype(float)
            methods = {"T3-M4": (rotations, world_rates(rotations))}
            if cfg.apply_wnoj_post:
                methods["T3-M5"] = wnoj_post(rotations, [SENSORS[i] for i in indices])
            for method, (r, w) in methods.items():
                orientation = geodesic_deg(reference[:, indices], r)
                angular_rate = np.linalg.norm(w - values[:, indices, 3:6], axis=-1)
                for local, idx in enumerate(indices):
                    records[(scheme, method, SENSORS[idx])].append(
                        (orientation[:, local], angular_rate[:, local], labels))
                if bones is not None:
                    full = reference.copy()       # observed sensors retain their own orientation
                    full[:, indices] = r
                    fk = forward_kinematics(dict(zip(SENSORS, full.swapaxes(0, 1))), bones)
                    for name in END_EFFECTORS:
                        error = np.linalg.norm(fk[name] - fk_ref[name], axis=-1)
                        records[(scheme, method, "EE-" + name)].append((error,))
    if not records:
        raise ValueError(f"No test chunks of at least 120 frames for subject {subject}")
    for (scheme, method, sensor), chunks in records.items():
        folder = destination / scheme.replace(":", "_") / method / sensor
        folder.mkdir(parents=True, exist_ok=True)
        arrays = [np.concatenate(values) for values in zip(*chunks)]
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("Non-finite evaluation reference or prediction")
        names = ("v3",) if sensor.startswith("EE-") else ("v1", "v2", "labels")
        np.savez(folder / f"s{subject}.npz", **dict(zip(names, arrays)))
    return bones is not None


def _errors(v1, v2):
    return {"n_frames": len(v1), "rmse_deg": np.sqrt(np.mean(v1 ** 2)),
            "median_deg": np.median(v1), "p95_deg": np.percentile(v1, 95),
            "v2_rmse_rad_s": np.sqrt(np.mean(v2 ** 2))}


def aggregate(fold_dirs, input_masks):
    """Pool frames exactly, retaining one sensor/method/layout in memory at a time."""
    grouped = defaultdict(list)
    for directory in fold_dirs:
        for path in (directory / "errors").glob("*/*/*/*.npz"):
            scheme, method, sensor = path.relative_to(directory / "errors").parts[:3]
            grouped[(scheme.replace("_", ":", 1), method, sensor)].append(path)
    sensors, activities, endpoints = [], [], []
    for (scheme, method, sensor), paths in sorted(grouped.items()):
        arrays = defaultdict(list)
        for path in paths:
            with np.load(path, allow_pickle=False) as record:
                for key in record.files:
                    arrays[key].append(record[key])
        base = {"scheme": scheme, "method": method}
        if sensor.startswith("EE-"):
            values = np.concatenate(arrays["v3"])
            endpoints.append({**base, "end_effector": sensor[3:], "n_frames": len(values),
                              "median_m": np.median(values), "p95_m": np.percentile(values, 95),
                              "rmse_m": np.sqrt(np.mean(values ** 2))})
            continue
        v1, v2, labels = (np.concatenate(arrays[k]) for k in ("v1", "v2", "labels"))
        masks = {"+".join(layout[scheme]) for layout in input_masks.values() if scheme in layout}
        sensors.append({**base, "withheld_sensor": sensor,
                        "input_mask": masks.pop() if len(masks) == 1 else "per-fold (see summary.json)",
                        **_errors(v1, v2), "n_subjects": len(paths),
                        "subject_equal_median_deg": np.mean([np.median(a) for a in arrays["v1"]])})
        for label in np.unique(labels):
            keep = labels == label
            activities.append({**base, "sensor": sensor, "activity": ACTIVITY[int(label)],
                               **_errors(v1[keep], v2[keep])})
    if not sensors:
        raise ValueError("No completed test results to aggregate")
    sensor_table = pd.DataFrame(sensors)
    ablation = sensor_table.groupby(["scheme", "method"], as_index=False).agg(
        withheld_sensors=("withheld_sensor", "count"),
        sensor_mean_subject_equal_median_deg=("subject_equal_median_deg", "mean"),
        sensor_mean_v2_rmse_rad_s=("v2_rmse_rad_s", "mean"))
    endpoint_table = pd.DataFrame(endpoints, columns=["scheme", "method", "end_effector", "n_frames",
                                                     "median_m", "p95_m", "rmse_m"])
    return {"per_sensor_metrics.csv": sensor_table, "per_activity_metrics.csv": pd.DataFrame(activities),
            "ablation_summary.csv": ablation, "v3_endpoint_metrics.csv": endpoint_table}
