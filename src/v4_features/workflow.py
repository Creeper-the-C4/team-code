"""Window-cache, fold, and reporting helpers shared by Task 1 and Task 2."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import numpy as np
import pandas as pd

from scripts.diagnostics.common import ACTIVITY
from scripts.preprocessing.independent_data_access import SENSORS, SubjectData
from src.v4_features.feature_families import (
    BY_SENSOR, extract_features, feature_dependencies, feature_names, validate_dependencies,
)


N_CLASSES = 13
LABELS = np.arange(N_CLASSES)
PROBABILITY_COLUMNS = [f"p_{label}" for label in LABELS]
WINDOW_COLUMNS = [
    "subject", "trial", "source_start", "source_stop_exclusive", "stride_frames", "majority_label",
]
CACHE_VERSION = 1


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def completed(directory):
    """Accept only a finished summary, including unversioned older summaries."""
    for filename in ("summary.json", "complete.json"):
        path = directory / filename
        if path.is_file() and read_json(path).get("complete", True):
            return True
    return False


def load_folds(path, fast_smoke=False):
    folds = read_json(path)["folds"]
    if not folds or len({fold["fold"] for fold in folds}) != len(folds):
        raise ValueError("Missing folds or duplicate fold identifiers")
    all_test = []
    universe = None
    for fold in folds:
        roles = [fold[name] for name in ("train", "validation", "test")]
        if any(not ids or len(set(ids)) != len(ids) for ids in roles):
            raise ValueError("Every fold role must contain distinct participants")
        if any(not isinstance(subject, int) or subject < 1 for role in roles for subject in role):
            raise ValueError("Participant IDs must be positive integers")
        members = set().union(*roles)
        if sum(map(len, roles)) != len(members):
            raise ValueError(f"Roles overlap in fold {fold['fold']}")
        if universe is not None and members != universe:
            raise ValueError("Folds have different participant populations")
        universe = members
        all_test.extend(roles[2])
    if len(all_test) != len(universe) or set(all_test) != universe:
        raise ValueError("Outer folds must test every participant exactly once")
    if not fast_smoke and (
        len(folds) != 6 or universe != set(range(1, 31))
        or any(tuple(len(fold[key]) for key in ("train", "validation", "test")) != (20, 5, 5)
               for fold in folds)
    ):
        raise ValueError("A formal run requires the 30-participant, six-fold protocol")
    return folds[:2] if fast_smoke else folds


def window_table(data_root, subject, window_sec):
    path = data_root / "subjects" / f"P{subject:02d}" / f"windows_{window_sec}s.parquet"
    windows = pd.read_parquet(path)
    if "stride_frames" not in windows:
        windows["stride_frames"] = window_sec * 30
    if not set(WINDOW_COLUMNS).issubset(windows.columns):
        raise ValueError(f"Missing window columns in {path}")
    if not len(windows) or not (windows["subject"] == subject).all():
        raise ValueError(f"Empty or mismatched participant windows in {path}")
    if windows.duplicated(["trial", "source_start"]).any():
        raise ValueError(f"Duplicate window coordinates in {path}")
    if not np.isin(windows["majority_label"], LABELS).all():
        raise ValueError(f"Unexpected labels in {path}")
    return path, windows


def _cache_location(out_cache, family, subject, window_sec):
    return out_cache / f"{window_sec}s" / family / f"P{subject:02d}"


def _cache_spec(family, subject, window_sec, source, length):
    return {
        "complete": True, "version": CACHE_VERSION, "family": family, "subject": subject,
        "window_seconds": window_sec, "windows": length, "sensors": list(SENSORS),
        "window_index_sha256": file_sha256(source),
        "features": feature_names(family, SENSORS),
    }


def _valid_cache(directory, spec):
    done = directory / "complete.json"
    features = directory / "features.npy"
    if not done.is_file() or not features.is_file() or read_json(done) != spec:
        return False
    expected = (spec["windows"], len(SENSORS), BY_SENSOR[spec["family"]]) if (
        spec["family"] in BY_SENSOR
    ) else (spec["windows"], len(spec["features"]))
    data = np.load(features, mmap_mode="r", allow_pickle=False)
    return data.shape == expected and data.dtype == np.float32


def build_subject_cache(data_root, out_cache, subject, window_sec, families, overwrite=False):
    """Read only official trial-local windows and write fixed, per-subject caches."""
    source, windows = window_table(data_root, subject, window_sec)
    specifications = {family: _cache_spec(family, subject, window_sec, source, len(windows))
                      for family in families}
    pending = {family: _cache_location(out_cache, family, subject, window_sec)
               for family in families}
    if not overwrite:
        pending = {family: folder for family, folder in pending.items()
                   if not _valid_cache(folder, specifications[family])}
    if not pending:
        return len(windows)
    data = SubjectData(data_root, subject)
    if data.signals.ndim != 3 or data.signals.shape[1:] != (len(SENSORS), 10):
        raise ValueError("Unexpected independent_v1 signal shape")
    bounds = data.trials.set_index("trial").reindex(windows["trial"])
    starts = windows["source_start"].to_numpy(dtype=np.int64)
    stops = windows["source_stop_exclusive"].to_numpy(dtype=np.int64)
    width = window_sec * 60
    if bounds[["source_start", "source_stop_exclusive"]].isna().any().any() or (
        (starts < bounds["source_start"].to_numpy()).any()
        or (stops > bounds["source_stop_exclusive"].to_numpy()).any()
        or (starts < 0).any() or (stops > len(data.signals)).any()
        or (stops - starts != width).any()
    ):
        raise ValueError(f"P{subject:02d}: a window exceeds its official trial boundary")
    for family, directory in pending.items():
        directory.mkdir(parents=True, exist_ok=True)
        spec = specifications[family]
        write_json(directory / "complete.json", {"complete": False})
        shape = (len(windows), len(spec["features"]))
        if family in BY_SENSOR:
            shape = (len(windows), len(SENSORS), BY_SENSOR[family])
        with NamedTemporaryFile(suffix=".npy", dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
        output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
        try:
            offsets = np.arange(width)
            for offset in range(0, len(windows), 128):
                frames = starts[offset:offset + 128, None] + offsets
                output[offset:offset + len(frames)] = extract_features(data.signals[frames], family, SENSORS)
            output.flush()
        except BaseException:
            del output
            temporary.unlink(missing_ok=True)
            raise
        del output
        temporary.replace(directory / "features.npy")
        write_json(directory / "complete.json", spec)
    return len(windows)


def feature_columns(family, sensors):
    """Only refer to features with dependencies fully inside the retained layout."""
    full = feature_dependencies(family, SENSORS)
    available = feature_dependencies(family, sensors)
    validate_dependencies(available, sensors)
    lookup = {name: index for index, name in enumerate(full)}
    return np.array([lookup[name] for name in available], dtype=np.intp)


def load_split(data_root, out_cache, participants, family, window_sec, cap=None):
    arrays, frames = [], []
    for subject in participants:
        directory = _cache_location(out_cache, family, subject, window_sec)
        table_path, windows = window_table(data_root, subject, window_sec)
        spec = _cache_spec(family, subject, window_sec, table_path, len(windows))
        if not _valid_cache(directory, spec):
            if family != "F4":
                raise ValueError(f"Missing or stale {family} feature cache for P{subject:02d}: {directory}")
            # The cloud extractor's standard command omits F4. Build it on demand,
            # per participant, so test features are opened only in the test stage.
            build_subject_cache(data_root, out_cache, subject, window_sec, [family])
        feat = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
        windows = windows.copy()
        windows["window_id"] = np.arange(len(windows))
        if cap is not None:
            y = windows["majority_label"].to_numpy()
            rng = np.random.default_rng(42 + subject)
            indexes = np.concatenate([
                rng.choice(pool, min(len(pool), cap), replace=False)
                for label in LABELS if len(pool := np.flatnonzero(y == label))
            ])
            indexes.sort()
            windows = windows.iloc[indexes]
            feat = feat[indexes]
        arrays.append(feat.reshape(len(windows), -1))
        frames.append(windows)
    matrix = np.concatenate(arrays)
    if np.isinf(matrix).any():
        raise ValueError("Feature cache contains infinity")
    return matrix, pd.concat(frames, ignore_index=True)


def _confusion(y, predictions):
    return np.bincount(N_CLASSES * np.asarray(y, dtype=int) + predictions, minlength=N_CLASSES**2
                       ).reshape(N_CLASSES, N_CLASSES)


def _f1(cm):
    tp = cm.diagonal()
    support = cm.sum(axis=1) + cm.sum(axis=0)
    return np.divide(2 * tp, support, out=np.zeros(N_CLASSES), where=support > 0)


def score_predictions(meta, predictions):
    predictions = np.asarray(predictions, dtype=np.int64)
    y = meta["majority_label"].to_numpy(dtype=np.int64)
    if len(y) != len(predictions) or not len(y) or not np.isin(predictions, LABELS).all():
        raise ValueError("Require one valid prediction per window")
    cm = _confusion(y, predictions)
    class_f1 = _f1(cm)
    individuals = []
    for subject, indices in meta.groupby("subject", sort=True).indices.items():
        subject_cm = _confusion(y[indices], predictions[indices])
        f1 = _f1(subject_cm)
        individuals.append({
            "subject": int(subject), "n_windows": len(indices), "macro_f1_13": float(f1.mean()),
            "macro_f1_named12": float(f1[1:].mean()), "walk3_macro_f1": float(f1[4:7].mean()),
            "accuracy_13": float(subject_cm.trace() / subject_cm.sum()),
        })
    scores = {
        "subject_equal_macro_f1_13": float(np.mean([row["macro_f1_13"] for row in individuals])),
        "pooled_macro_f1_13": float(class_f1.mean()), "macro_f1_named12": float(class_f1[1:].mean()),
        "walk3_macro_f1": float(class_f1[4:7].mean()), "accuracy_13": float(cm.trace() / cm.sum()),
        "n_windows": len(y), "n_subjects": len(individuals),
        "others_f1": float(class_f1[0]), "rowing_f1": float(class_f1[12]),
        "worst_class_f1": float(class_f1.min()), "worst_class_id": int(class_f1.argmin()),
        "worst_class_name": ACTIVITY[int(class_f1.argmin())],
    }
    return scores, individuals, cm


def class_metrics(cm):
    recall = np.divide(cm.diagonal(), cm.sum(axis=1), out=np.zeros(N_CLASSES), where=cm.sum(axis=1) > 0)
    precision = np.divide(cm.diagonal(), cm.sum(axis=0), out=np.zeros(N_CLASSES), where=cm.sum(axis=0) > 0)
    return pd.DataFrame({"class_id": LABELS, "class_name": [ACTIVITY[i] for i in LABELS],
                         "support": cm.sum(axis=1), "precision": precision, "recall": recall, "f1": _f1(cm)})


def fit_classifier(train, train_meta, validation, validation_meta, n_estimators, fast_smoke=False):
    import lightgbm as lgb

    y = train_meta["majority_label"].to_numpy(dtype=np.int64)
    val_y = validation_meta["majority_label"].to_numpy(dtype=np.int64)
    classes = np.unique(y)
    if len(classes) < 2 or np.setdiff1d(val_y, classes).size:
        raise ValueError("Training must cover all validation classes")
    if not fast_smoke and len(classes) != N_CLASSES:
        raise ValueError("Formal training must contain all 13 classes")
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators, learning_rate=0.08, num_leaves=15 if fast_smoke else 31,
        max_bin=63, min_child_samples=2 if fast_smoke else 50,
        n_jobs=1 if fast_smoke else min(12, os.cpu_count() or 1),
        random_state=42, verbosity=-1, metric="None", force_col_wise=True, deterministic=True,
    )

    def validation_metric(encoded_labels, probability):
        if probability.ndim == 1:
            probability = np.stack([1 - probability, probability], axis=1)
        predicted = classes[probability.argmax(axis=1)]
        target = classes[encoded_labels.astype(np.intp)]
        return "macro_f1_13", float(_f1(_confusion(target, predicted)).mean()), True

    model.fit(train, y, eval_set=[(validation, val_y)], eval_metric=validation_metric,
              callbacks=[lgb.early_stopping(min(40, max(2, n_estimators // 5)),
                                          first_metric_only=True, verbose=False)])
    return model


def predict_probabilities(model, matrix):
    # Use the Booster to preserve the NumPy column order without scikit-learn's
    # generated-feature-name warnings.
    raw = model.booster_.predict(matrix, num_iteration=model.best_iteration_)
    if raw.ndim == 1:
        raw = np.stack([1 - raw, raw], axis=1)
    probs = np.zeros((len(matrix), N_CLASSES))
    probs[:, model.classes_.astype(np.intp)] = raw
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Classifier returned invalid probabilities")
    return probs


def begin_run(output, config, overwrite):
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if not overwrite and config_path.exists():
        previous = read_json(config_path)
        if previous.get("parameters") != config:
            raise ValueError("Run parameters changed; use a new output directory or --overwrite")
        if completed(output):
            print(f"Already complete: {output}", flush=True)
            return None
        run_id = previous["run_id"]
    else:
        run_id = uuid4().hex
    write_json(output / "summary.json", {"complete": False})
    if (output / "complete.json").exists():
        write_json(output / "complete.json", {"complete": False})
    if overwrite or not config_path.exists():
        write_json(config_path, {"parameters": config, "run_id": run_id})
    return run_id


def cached_fold(directory, run_id):
    path = directory / "complete.json"
    if not path.is_file():
        return False
    marker = read_json(path)
    return marker.get("complete") is True and marker.get("run_id") == run_id
