"""Select nested sensor layouts on validation participants, then evaluate each on held-out participants."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.v4_features.workflow import (
    SENSORS, WINDOW_COLUMNS, begin_run, cached_fold, completed, feature_columns, file_sha256,
    fit_classifier, load_folds, load_split, predict_probabilities, read_json, score_predictions, write_json,
)


BACKWARD = "backward_greedy"
FORWARD = "forward_greedy"


def validation_path(train, train_meta, validation, validation_meta, family, strategy, sensors,
                    n_estimators, fast_smoke):
    """Learn all best layouts without access to the outer test fold."""
    sensors = tuple(sensors)
    fixed = sensors if strategy == BACKWARD else ()
    paths = {}
    for step in range(len(sensors)):
        if strategy == BACKWARD:
            candidates = [fixed] if step == 0 else [tuple(s for s in fixed if s != removed) for removed in fixed]
        else:
            candidates = [tuple(s for s in sensors if s in fixed or s == added)
                          for added in sensors if added not in fixed]
        winner = None
        for subset in candidates:
            columns = feature_columns(family, subset)
            val_matrix = validation[:, columns]
            model = fit_classifier(train[:, columns], train_meta, val_matrix, validation_meta,
                                   n_estimators, fast_smoke)
            predicted = predict_probabilities(model, val_matrix).argmax(axis=1)
            scores, _, _ = score_predictions(validation_meta, predicted)
            compare = (scores["subject_equal_macro_f1_13"], scores["pooled_macro_f1_13"])
            if winner is None or compare > winner[0]:
                winner = (compare, subset, columns, model)
        score, fixed, columns, model = winner
        paths[len(fixed)] = (fixed, columns, model, score[0])
        print(f"Validation selected {len(fixed)}/{len(sensors)} sensors: {', '.join(fixed)}", flush=True)
    return paths


def run_fold(args, fold, directory, run_id, sensors):
    directory.mkdir(parents=True, exist_ok=True)
    path_file = directory / "selected.json"
    prediction_file = directory / "test_predictions.parquet"
    if cached_fold(directory, run_id) and path_file.is_file() and prediction_file.is_file():
        return read_json(path_file), pd.read_parquet(prediction_file)

    def get_data(role):
        return load_split(args.data_root, args.feature_cache, fold[role], args.family, args.window_sec,
                          cap=16 if args.fast_smoke else None)

    train, train_meta = get_data("train")
    validation, validation_meta = get_data("validation")
    paths = validation_path(train, train_meta, validation, validation_meta, args.family, args.strategy,
                            sensors, args.n_estimators, args.fast_smoke)
    del train, validation
    selections = {str(size): list(item[0]) for size, item in paths.items()}
    write_json(path_file, selections)
    # Test values first become accessible after every subset for this fold is fixed.
    test, test_meta = get_data("test")
    predictions = []
    for size, (_, columns, model, _) in paths.items():
        result = test_meta[WINDOW_COLUMNS].copy()
        result["fold"] = fold["fold"]
        result["k"] = size
        result["y_pred"] = predict_probabilities(model, test[:, columns]).argmax(axis=1)
        predictions.append(result)
    results = pd.concat(predictions, ignore_index=True)
    results.to_parquet(prediction_file, index=False)
    write_json(directory / "complete.json", {
        "complete": True, "run_id": run_id, "participants": fold,
        "validation_subject_equal_macro_f1_13": {str(size): item[3] for size, item in paths.items()},
    })
    return selections, results


def report(output, config, selections, predictions, sensors):
    if predictions.duplicated(["k", "subject", "trial", "source_start"]).any():
        raise ValueError("A held-out window was predicted more than once at the same k")
    rows = []
    best_score = -np.inf
    for size, rows_at_size in predictions.groupby("k", sort=True):
        scores, _, _ = score_predictions(rows_at_size, rows_at_size["y_pred"])
        fold_scores = [
            score_predictions(frame, frame["y_pred"])[0] for _, frame in rows_at_size.groupby("fold")
        ]
        record = {"k": int(size), **scores}
        for key in ("subject_equal_macro_f1_13", "pooled_macro_f1_13"):
            record[f"{key}_fold_sd"] = float(np.std([row[key] for row in fold_scores], ddof=1)) if (
                len(fold_scores) > 1
            ) else None
        counts = {sensor: sum(sensor in fold[str(size)] for fold in selections.values()) for sensor in sensors}
        record["sensor_selection_frequency"] = json.dumps(counts, separators=(",", ":"))
        score = record["subject_equal_macro_f1_13"]
        record["pareto_nondominated"] = bool(score > best_score)
        best_score = max(best_score, score)
        rows.append(record)
    pd.DataFrame(rows).to_csv(output / "pareto_curve.csv", index=False)
    write_json(output / "per_fold_selected_subsets.json", {"family": config["family"], "folds": selections})
    (output / "README.md").write_text(
        "# Held-out sensor selection\n\n"
        "Validation participant-equal Macro-F1 determines each greedy addition or removal. "
        "Pooled Macro-F1 breaks validation ties. Canonical sensor order breaks remaining ties.\n\n"
        "The outer test participants are predicted once per selected layout size. "
        "The reported fold standard deviations use ddof=1. Sensor frequencies count folds. "
        "The Pareto flag compares held-out scores and is descriptive. "
        "Use the corresponding fold of per_fold_selected_subsets.json in downstream held-out work.\n",
        encoding="utf-8",
    )
    write_json(output / "summary.json", {"complete": True, "config": config, "pareto": rows})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--family", choices=("F1H", "F3", "F4"), default="F1H")
    parser.add_argument("--strategy", choices=(BACKWARD, FORWARD), default=BACKWARD)
    parser.add_argument("--window-sec", type=int, choices=(2, 4), default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument("--fast-smoke", action="store_true")
    parser.add_argument("--sensors", nargs="+", choices=SENSORS,
                        help="Restrict the starting layout for a smoke test")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not args.overwrite and completed(args.out):
        print(f"Already complete: {args.out}", flush=True)
        return
    if args.n_estimators is None:
        args.n_estimators = 6 if args.fast_smoke else 80
    if args.n_estimators < 2:
        parser.error("--n-estimators must be at least 2")
    if args.sensors is not None and (not args.fast_smoke or len(set(args.sensors)) != len(args.sensors)):
        parser.error("--sensors requires --fast-smoke and distinct sensor names")
    sensors = SENSORS if args.sensors is None else [sensor for sensor in SENSORS if sensor in args.sensors]
    folds = load_folds(args.splits, args.fast_smoke)
    config = {
        "data_root": str(args.data_root.resolve()), "feature_cache": str(args.feature_cache.resolve()),
        "split_sha256": file_sha256(args.splits), "family": args.family, "strategy": args.strategy,
        "window_seconds": args.window_sec, "fast_smoke": args.fast_smoke, "sensors": sensors,
        "n_estimators": args.n_estimators,
    }
    run_id = begin_run(args.out, config, args.overwrite)
    if run_id is None:
        return
    selections = {}
    predictions = []
    for fold in folds:
        layout, result = run_fold(args, fold, args.out / f"fold_{fold['fold']}", run_id, sensors)
        selections[str(fold["fold"])] = layout
        predictions.append(result)
        print(f"Fold {fold['fold']} complete", flush=True)
    report(args.out, config, selections, pd.concat(predictions, ignore_index=True), sensors)


if __name__ == "__main__":
    main()
