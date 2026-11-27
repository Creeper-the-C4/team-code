"""Fixed feature families for exported world-frame IMU channels."""

import numpy as np


FAMILIES = ("F1W", "F1B", "F1H", "F2", "F2p", "F3", "F4")
BY_SENSOR = {"F1W": 32, "F1B": 32, "F1H": 24, "F2": 8, "F2p": 5, "F3": 40}
VECTOR_NAMES = ("Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z")
HORIZONTAL_NAMES = ("Acc_H", "Acc_Y", "Gyr_H", "Gyr_Y")
STATS = ("mean", "std", "rms", "range")
MAGNITUDES = tuple(f"{signal}_magnitude_{stat}" for signal in ("acc", "gyr") for stat in STATS[:3])
INCREMENTS = ("orientation_increment_mean_rad", "orientation_increment_rms_rad")
ORIENTATION = tuple(f"orientation_center_R_{i}{j}" for j in (0, 1) for i in range(3)) + INCREMENTS
ORIENTATION_H = tuple(f"orientation_center_R_1{j}" for j in range(3)) + INCREMENTS
PAIRS = (
    ("LowerBack", "UpperBack"), ("LowerBack", "LeftThigh"), ("LowerBack", "RightThigh"),
    ("LeftThigh", "LeftShank"), ("RightThigh", "RightShank"),
    ("LeftShank", "LeftFoot"), ("RightShank", "RightFoot"), ("UpperBack", "Head"),
    ("RightUpperArm", "RightForeArm"), ("LeftUpperArm", "LeftForeArm"),
)
THIGHS = ("LeftThigh", "RightThigh")
PAIR_NAMES = ("angle_mean_rad", "angle_std_rad", "angle_range_rad", "gyro_mean", "gyro_rms")
THIGH_NAMES = ("swing_rad", "step_frequency_hz")


def _frequency_names(signal):
    return (f"{signal}_dominant_frequency_hz", f"{signal}_peak_fraction_0_5_to_4_hz")


def _names(channels, spectrum):
    return tuple(f"{channel}_{stat}" for channel in channels for stat in STATS) + (
        MAGNITUDES + _frequency_names(spectrum)
    )


WORLD_NAMES = _names(VECTOR_NAMES, "acc_y")
LOCAL_NAMES = {
    "F1W": WORLD_NAMES,
    "F1B": _names(tuple(f"body_{channel}" for channel in VECTOR_NAMES), "acc_mag"),
    "F1H": _names(HORIZONTAL_NAMES, "acc_y"),
    "F2": ORIENTATION,
    "F2p": ORIENTATION_H,
    "F3": WORLD_NAMES + ORIENTATION,
}


def _validate_names(sensor_names):
    names = tuple(sensor_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Provide a nonempty sequence of distinct sensor names")
    return names


def feature_dependencies(family, sensor_names):
    """Return the ordered dependency mapping for flattened feature columns."""
    names = _validate_names(sensor_names)
    local = "F3" if family == "F4" else family
    dependencies = {
        f"{name}.{column}": (name,)
        for name in names
        for column in LOCAL_NAMES[local]
    }
    if family == "F4":
        for left, right in PAIRS:
            if left in names and right in names:
                for column in PAIR_NAMES:
                    dependencies[f"{left}--{right}.{column}"] = (left, right)
        for name in THIGHS:
            if name in names:
                for column in THIGH_NAMES:
                    dependencies[f"{name}.kinematics.{column}"] = (name,)
    return dependencies


def feature_names(family, sensor_names):
    return list(feature_dependencies(family, sensor_names))


def validate_dependencies(dependencies, sensor_names):
    available = set(sensor_names)
    for name, required in dependencies.items():
        if not required or not set(required).issubset(available):
            raise ValueError(f"Unavailable dependency in {name}: {required}")


def _rms(values):
    return np.sqrt(np.mean(np.square(values), axis=1))


def _statistics(values, include_range=False):
    pieces = [values.mean(axis=1), values.std(axis=1), _rms(values)]
    if include_range:
        pieces.append(np.ptp(values, axis=1))
    return np.stack(pieces, axis=-1)


def _spectrum(signal, fs):
    frames = signal.shape[1]
    frequencies = np.fft.rfftfreq(frames, 1 / fs)
    band = (frequencies >= 0.5) & (frequencies <= 4)
    if not band.any():
        return np.stack([np.full_like(signal[:, 0], np.nan), np.zeros_like(signal[:, 0])], axis=-1)
    centered = signal - signal.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered * np.hanning(frames)[None, :, None], axis=1)) ** 2
    selected = power[:, band]
    peak = selected.argmax(axis=1)
    highest = np.take_along_axis(selected, peak[:, None], axis=1)[:, 0]
    total = selected.sum(axis=1)
    ratio = np.divide(highest, total, out=np.zeros_like(highest), where=total > 1e-15)
    active = np.var(signal, axis=1) > 1e-10
    return np.stack([np.where(active, frequencies[band][peak], np.nan),
                     np.where(active, ratio, 0)], axis=-1)


def _inertial_features(channels, vectors, spectrum_signal, fs):
    return np.concatenate([
        _statistics(channels, include_range=True).reshape(len(channels), channels.shape[2], -1),
        _statistics(np.linalg.norm(vectors[..., :3], axis=-1)),
        _statistics(np.linalg.norm(vectors[..., 3:], axis=-1)),
        _spectrum(spectrum_signal, fs),
    ], axis=-1)


def _normalized(quaternions):
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if not np.isfinite(quaternions).all() or np.any(norms < 1e-6):
        raise ValueError("Invalid source orientation")
    return quaternions / norms


def _rotation(q):
    x, y, z, w = np.moveaxis(q, -1, 0)
    entries = (1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w),
               2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w),
               2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y))
    return np.stack(entries, axis=-1).reshape(q.shape[:-1] + (3, 3))


def _distance(left, right):
    """SO(3) geodesic angle with the nearer quaternion hemisphere."""
    sign = np.where(np.sum(left * right, axis=-1, keepdims=True) < 0, -1, 1)
    chord = np.linalg.norm(right * sign - left, axis=-1)
    return 4 * np.arcsin(np.clip(chord / 2, 0, 1))


def _orientation_features(q, horizontal=False):
    center = _rotation(q[:, q.shape[1] // 2])
    encoded = center[..., 1, :] if horizontal else center[..., :2].swapaxes(-1, -2).reshape(
        len(q), q.shape[2], 6
    )
    increments = _distance(q[:, :-1], q[:, 1:])
    return np.concatenate([encoded, np.stack([increments.mean(axis=1), _rms(increments)], axis=-1)], axis=-1)


def _cross_features(base, vectors, q, names, fs):
    indices = {name: index for index, name in enumerate(names)}
    pieces = [base.reshape(len(base), -1)]
    for left, right in PAIRS:
        if left in indices and right in indices:
            u, v = indices[left], indices[right]
            # Geodesic distance between orientations is the angle of R_u.T @ R_v.
            angle = _distance(q[:, :, u], q[:, :, v])
            gyro = np.linalg.norm(vectors[:, :, v, 3:] - vectors[:, :, u, 3:], axis=-1)
            pieces.append(np.stack([angle.mean(axis=1), angle.std(axis=1), np.ptp(angle, axis=1),
                                    gyro.mean(axis=1), _rms(gyro)], axis=-1))
    for name in THIGHS:
        if name in indices:
            i = indices[name]
            # Approximate sagittal pitch about the sensor's local X axis with
            # the projected vertical direction. One thigh cycle is two steps.
            rotation = _rotation(q[:, :, i])
            pitch = np.unwrap(np.arctan2(-rotation[..., 1, 2], rotation[..., 1, 1]), axis=1)
            gyro_body = np.einsum("btij,bti->btj", rotation, vectors[:, :, i, 3:])
            stride_hz = _spectrum(gyro_body[:, :, :1], fs)[:, 0, 0]
            pieces.append(np.stack([np.ptp(pitch, axis=1), 2 * stride_hz], axis=-1))
    return np.concatenate(pieces, axis=1)


def extract_features(values, family, sensor_names, fs=60.0):
    """Transform [batch, frames, sensors, 10] with no fitted statistics.

    F1B uses centered acceleration magnitude for its 0.5–4 Hz spectrum.
    F1H and F1W access only the six inertial channels. NaN frequency
    denotes a near-constant or unresolved signal, as in the original baseline.
    """
    names = _validate_names(sensor_names)
    if family not in FAMILIES:
        raise ValueError(f"Unknown feature family: {family}")
    if values.ndim != 4 or values.shape[2:] != (len(names), 10) or values.shape[1] < 2:
        raise ValueError("Expected [batch, frames >= 2, sensors, 10] in sensor_names order")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling frequency must be positive")
    if family in ("F2", "F2p"):
        q = _normalized(values[..., 6:].astype(np.float64))
        return _orientation_features(q, horizontal=family == "F2p").astype(np.float32)
    vectors = values[..., :6].astype(np.float64)
    if family == "F1H":
        ax, ay, az, gx, gy, gz = np.moveaxis(vectors, -1, 0)
        horizontal = np.stack([np.hypot(ax, az), ay, np.hypot(gx, gz), gy], axis=-1)
        return _inertial_features(horizontal, vectors, ay, fs).astype(np.float32)
    if family == "F1W":
        return _inertial_features(vectors, vectors, vectors[..., 1], fs).astype(np.float32)
    q = _normalized(values[..., 6:].astype(np.float64))
    if family == "F1B":
        both_vectors = vectors.reshape(vectors.shape[:-1] + (2, 3))
        body = np.einsum("...ij,...ki->...kj", _rotation(q), both_vectors).reshape(vectors.shape)
        magnitude = np.linalg.norm(vectors[..., :3], axis=-1)
        return _inertial_features(body, vectors, magnitude, fs).astype(np.float32)
    world = _inertial_features(vectors, vectors, vectors[..., 1], fs)
    base = np.concatenate([world, _orientation_features(q)], axis=-1)
    if family == "F3":
        return base.astype(np.float32)
    result = _cross_features(base, vectors, q, names, fs)
    dependencies = feature_dependencies(family, names)
    validate_dependencies(dependencies, names)
    if result.shape[1] != len(dependencies):
        raise ValueError("F4 output columns do not match their dependency table")
    return result.astype(np.float32)
