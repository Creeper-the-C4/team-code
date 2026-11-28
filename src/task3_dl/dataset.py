"""Sensor-isolated inputs and trial-bound sampling for the independent IMU cache."""

from dataclasses import dataclass

import numpy as np
import torch

from scripts.preprocessing.independent_data_access import (
    SENSORS, SubjectData, fit_acc_gyr_scaler, prepare_quaternions,
)
from src.task3.so3_lie import quat_xyzw_to_rot

ACCEL_G_TO_SI = 9.80665
FS = 60
LIMBS = tuple(tuple(i for i, name in enumerate(SENSORS)
                   if name.startswith(side) and any(name.endswith(s) for s in parts))
             for side in ("Left", "Right")
             for parts in (("Thigh", "Shank", "Foot"), ("Shoulder", "UpperArm", "ForeArm", "Wrist")))


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray


def fit_scaler(root, fold):
    """Fit on all continuous training frames, then express the inertial moments in SI."""
    held_out = [*fold["validation"], *fold["test"]]
    moments = fit_acc_gyr_scaler(root, fold["train"], SENSORS, held_out)
    units = np.array([ACCEL_G_TO_SI] * 3 + [1.] * 3)
    return Scaler((moments["mean"] * units).astype("float32"),
                  (moments["std"] * units).astype("float32"))


def rotation_targets(values):
    """Training labels or evaluation references, supplied separately from model input."""
    q = prepare_quaternions(values)[..., 6:10]
    return quat_xyzw_to_rot(q).astype(np.float32)


def build_inputs(values, observed, scaler):
    """Read only observed columns. Omitted signals never reach QC or normalization."""
    observed = np.asarray(observed, dtype=bool)
    if observed.shape != (len(SENSORS),) or not observed.any():
        raise ValueError("Expected a non-empty observed layout in canonical sensor order")
    selected = prepare_quaternions(values[:, observed])
    inertial = selected[..., :6]
    if not np.isfinite(inertial).all():
        raise ValueError("Non-finite inertial reading in an observed sensor")
    inertial[..., :3] *= ACCEL_G_TO_SI
    rotations = quat_xyzw_to_rot(selected[..., 6:10])
    encoded = np.zeros((len(values), len(SENSORS), 13), dtype=np.float32)
    encoded[:, observed, :6] = (inertial - scaler.mean[observed]) / scaler.std[observed]
    encoded[:, observed, 6:9] = rotations[..., :, 0]
    encoded[:, observed, 9:12] = rotations[..., :, 1]
    encoded[:, observed, 12] = 1
    return torch.from_numpy(encoded.reshape(len(values), -1))


def sample_mask(rng):
    """Equal-probability mixture of random subsets, single holes and whole limbs."""
    mode = rng.integers(3)
    mask = np.ones(len(SENSORS), dtype=bool)
    if mode == 0:
        mask[:] = False
        mask[rng.choice(len(SENSORS), size=rng.integers(3, 16), replace=False)] = True
    elif mode == 1:
        mask[rng.integers(len(SENSORS))] = False
    else:
        mask[list(LIMBS[rng.integers(len(LIMBS))])] = False
    return mask


class TrainingWindows:
    """Sample a subject, then an official trial, then one unpadded contiguous crop."""

    def __init__(self, root, subjects, frames):
        self.sources = []
        self.frames = frames
        for subject in subjects:
            data = SubjectData(root, subject)
            bounds = data.trials[["source_start", "source_stop_exclusive"]].to_numpy(dtype=int)
            valid = bounds[bounds[:, 1] - bounds[:, 0] >= frames]
            if not len(valid):
                raise ValueError(f"No eligible training or validation trial for subject {subject}")
            self.sources.append((data, valid))

    def sample(self, rng, scaler, subject_index=None):
        index = rng.integers(len(self.sources)) if subject_index is None else subject_index
        data, bounds = self.sources[index]
        start, stop = bounds[rng.integers(len(bounds))]
        offset = rng.integers(start, stop - self.frames + 1)
        values = np.asarray(data.signals[offset:offset + self.frames])
        mask = sample_mask(rng)
        return (build_inputs(values, mask, scaler),
                torch.from_numpy(rotation_targets(values)), torch.from_numpy(mask))


def test_chunks(data, max_trials, max_seconds, chunk_seconds):
    """Same official trials, limits, frame starts and 120-frame minimum as Stage A.

    Use raw time slices so a hidden channel is never validated on the inference path.
    """
    per_activity = {}
    step = int(chunk_seconds * FS)
    for row in data.trials.itertuples():
        activity = int(row.majority_label)
        if max_trials and per_activity.get(activity, 0) >= max_trials:
            continue
        per_activity[activity] = per_activity.get(activity, 0) + 1
        start, stop = int(row.source_start), int(row.source_stop_exclusive)
        if max_seconds:
            stop = min(stop, start + int(max_seconds * FS))
        for left in range(start, stop, step):
            right = min(left + step, stop)
            if right - left < 120:
                break
            yield np.asarray(data.signals[left:right]), np.asarray(data.labels[left:right])
