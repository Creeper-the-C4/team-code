"""Select nested sensor layouts on validation participants, then evaluate each on held-out participants."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.v4_features.feature_families import feature_dependencies
from src.v4_features.workflow import (
    SENSORS, WINDOW_COLUMNS, begin_run, cached_fold, feature_columns, file_sha256,
    fit_classifier, load_folds, load_split, predict_probabilities, read_json, score_predictions, write_json,
)


BACKWARD = "backward_greedy"
FORWARD = "forward_greedy"
RANKING = "ranking_prefix"


def fit_layout(train, train_meta, validation, validation_meta, family, subset, n_estimators, fast_smoke):
    columns = feature_columns(family, subset)
    val_matrix = validation[:, columns]
    model = fit_classifier(train[:, columns], train_meta, val_matrix, validation_meta, n_estimators, fast_smoke)
    predicted = predict_probabilities(model, val_matrix).argmax(axis=1)
    scores, _, _ = score_predictions(validation_meta, predicted)
    return columns, model, (scores["subject_equal_macro_f1_13"], scores["pooled_macro_f1_13"])


def validation_path(train, train_meta, validation, validation_meta, family, strategy, sensors,
                    n_estimators, fast_smoke):
    """Learn all best layouts without access to the outer test fold."""
    sensors = tuple(sensors)
    fixed = sensors if strategy == BACKWARD else ()
    paths = {}
    timings = {}
    if strategy == RANKING:
        started = perf_counter()
        cols, model, score = fit_layout(train, train_meta, validation, validation_meta, family,
                                        sensors, n_estimators, fast_smoke)
        deps = feature_dependencies(family, sensors)
        gains = model.booster_.feature_importance(importance_type="gain", iteration=model.best_iteration_)
        sensor_gain = dict.fromkeys(sensors, 0.0)
        for required, gain in zip(deps.values(), gains, strict=True):
            for sensor in required:
                sensor_gain[sensor] += float(gain) / len(required)
        ranking = sorted(sensors, key=sensor_gain.get, reverse=True)
        paths[len(sensors)] = (sensors, cols, model, score[0])
        timings[len(sensors)] = perf_counter() - started
        for size in range(1, len(sensors)):
            started = perf_counter()
            subset = tuple(s for s in sensors if s in ranking[:size])
            cols, model, score = fit_layout(train, train_meta, validation, validation_meta, family,
                                            subset, n_estimators, fast_smoke)
            paths[size] = (subset, cols, model, score[0])
            timings[size] = perf_counter() - started
        return paths, timings

    for step in range(len(sensors)):
        started = perf_counter()
        if strategy == BACKWARD:
            candidates = [fixed] if step == 0 else [tuple(s for s in fixed if s != removed) for removed in fixed]
        else:
            candidates = [tuple(s for s in sensors if s in fixed or s == added)
                          for added in sensors if added not in fixed]
        winner = None
        for subset in candidates:
            columns, model, compare = fit_layout(train, train_meta, validation, validation_meta, family,
                                                 subset, n_estimators, fast_smoke)
            if winner is None or compare > winner[0]:
                winner = (compare, subset, columns, model)
        score, fixed, columns, model = winner
        paths[len(fixed)] = (fixed, columns, model, score[0])
        timings[len(fixed)] = perf_counter() - started
        print(f"Validation selected {len(fixed)}/{len(sensors)} sensors: {', '.join(fixed)}", flush=True)
    return paths, timings


def run_fold(args, fold, directory, run_id, sensors):
    directory.mkdir(parents=True, exist_ok=True)
    path_file = directory / "selected.json"
    prediction_file = directory / "test_predictions.parquet"
    if cached_fold(directory, run_id) and path_file.is_file() and prediction_file.is_file():
        return read_json(path_file), pd.read_parquet(prediction_file), read_json(directory / "complete.json")

    roles = [set(fold[role]) for role in ("train", "validation", "test")]
    if len(set.union(*roles)) != sum(map(len, roles)):
        raise ValueError("Participant roles overlap before layout selection")

    def get_data(role):
        matrix, meta = load_split(args.data_root, args.feature_cache, fold[role], args.family, args.window_sec,
                                  cap=16 if args.fast_smoke else None)
        if set(meta["subject"]) != set(fold[role]):
            raise ValueError(f"Loaded participants differ from assigned {role} participants")
        return matrix, meta

    train, train_meta = get_data("train")
    validation, validation_meta = get_data("validation")
    paths, timings = validation_path(train, train_meta, validation, validation_meta, args.family, args.strategy,
                                     sensors, args.n_estimators, args.fast_smoke)
    del train, validation
    selections = {str(size): list(item[0]) for size, item in paths.items()}
    write_json(path_file, selections)
    # Test values first become accessible after every subset for this fold is fixed.
    test, test_meta = get_data("test")
    predictions = []
    test_calls = {}
    for size, (subset, columns, model, _) in paths.items():
        started = perf_counter()
        if not np.array_equal(columns, feature_columns(args.family, subset)):
            raise ValueError("Model columns differ from locked layout dependencies")
        result = test_meta[WINDOW_COLUMNS].copy()
        result["fold"] = fold["fold"]
        result["k"] = size
        result["y_pred"] = predict_probabilities(model, test[:, columns]).argmax(axis=1)
        test_calls[str(size)] = test_calls.get(str(size), 0) + 1
        timings[size] += perf_counter() - started
        predictions.append(result)
    results = pd.concat(predictions, ignore_index=True)
    results.to_parquet(prediction_file, index=False)
    details = {
        "complete": True, "run_id": run_id, "participants": fold,
        "validation_subject_equal_macro_f1_13": {str(size): item[3] for size, item in paths.items()},
        "step_timing_seconds": {str(k): t for k, t in timings.items()},
        "anti_leakage_audit": {
            "roles_disjoint": True, "layouts_locked_before_test": True, "feature_dependencies_checked": True,
            "outer_test_predictions_per_k": test_calls,
            "train": fold["train"], "validation": fold["validation"], "test": fold["test"],
        },
    }
    write_json(directory / "complete.json", details)
    return selections, results, details


def report(output, config, selections, predictions, sensors, details):
    if predictions.duplicated(["k", "subject", "trial", "source_start"]).any():
        raise ValueError("A held-out window was predicted more than once at the same k")
    rows = []
    fold_rows = []
    stability = []
    fold_best = dict.fromkeys(selections, -np.inf)
    best_score = -np.inf
    for size, rows_at_size in predictions.groupby("k", sort=True):
        size = int(size)
        scores, _, _ = score_predictions(rows_at_size, rows_at_size["y_pred"])
        fold_scores = []
        flags = {}
        elapsed = {}
        for fold_id, frame in rows_at_size.groupby("fold", sort=True):
            fold = str(fold_id)
            values = score_predictions(frame, frame["y_pred"])[0]
            score = values["subject_equal_macro_f1_13"]
            flags[fold] = bool(score > fold_best[fold])
            fold_best[fold] = max(score, fold_best[fold])
            elapsed[fold] = details[fold]["step_timing_seconds"][str(size)]
            fold_rows.append({"strategy": config["strategy"], "fold": int(fold_id), "k": size, **values,
                              "pareto_nondominated": flags[fold], "step_timing_seconds": elapsed[fold]})
            fold_scores.append(values)
        record = {"strategy": config["strategy"], "k": size, **scores,
                  "per_fold_pareto": flags, "step_timing_seconds": elapsed,
                  "anti_leakage_audit": {
                      f: v["anti_leakage_audit"]["roles_disjoint"] and
                         v["anti_leakage_audit"]["feature_dependencies_checked"] and
                         v["anti_leakage_audit"]["layouts_locked_before_test"] and
                         v["anti_leakage_audit"]["outer_test_predictions_per_k"][str(size)] == 1
                      for f, v in details.items()
                  }}
        for key in ("subject_equal_macro_f1_13", "pooled_macro_f1_13"):
            record[f"{key}_fold_sd"] = float(np.std([row[key] for row in fold_scores], ddof=1)) if (
                len(fold_scores) > 1
            ) else None
        counts = {sensor: sum(sensor in fold[str(size)] for fold in selections.values()) for sensor in sensors}
        record["sensor_selection_frequency"] = counts
        for sensor, count in counts.items():
            entry = [min(int(k) for k, layout in fold.items() if sensor in layout) for fold in selections.values()]
            ranks = [len(sensors) + 1 - k for k in entry] if config["strategy"] == BACKWARD else entry
            stability.append({"strategy": config["strategy"], "sensor": sensor, "k": size,
                              "folds_selected": count, "fold_count": len(ranks),
                              "selection_frequency": count / len(ranks),
                              "rank_mean": float(np.mean(ranks)),
                              "rank_sd": float(np.std(ranks, ddof=1)) if len(ranks) > 1 else None})
        score = record["subject_equal_macro_f1_13"]
        record["pareto_nondominated"] = bool(score > best_score)
        best_score = max(best_score, score)
        rows.append(record)
    table = pd.DataFrame(rows)
    for key in ("sensor_selection_frequency", "per_fold_pareto", "step_timing_seconds", "anti_leakage_audit"):
        table[key] = table[key].map(json.dumps)
    table.to_csv(output / "pareto_curve.csv", index=False)
    comparison = output / "ranking_prefix_pareto_curve.csv"
    if config["strategy"] == RANKING:
        table.to_csv(comparison, index=False)
    else:
        comparison.unlink(missing_ok=True)
    pd.DataFrame(fold_rows).to_csv(output / "per_fold_pareto.csv", index=False)
    pd.DataFrame(stability).to_csv(output / "sensor_rank_stability.csv", index=False)
    write_json(output / "per_fold_selected_subsets.json", {
        "family": config["family"], "strategy": config["strategy"], "folds": selections,
    })
    method = ("Gain ranking on one full-layout training fit determines the ranking_prefix comparison. "
              "Shared feature gains are divided equally among their dependent sensors. "
              "Each smaller prefix gets a separate training fit with validation early stopping."
              if config["strategy"] == RANKING else
              "The exact wrapper evaluates every candidate addition or removal on validation participants. "
              "Participant-equal Macro-F1 determines the winner and pooled Macro-F1 breaks ties.")
    (output / "README.md").write_text(
        f"# Held-out sensor selection ({config['strategy']})\n\n{method} "
        "Canonical sensor order breaks remaining ties.\n\n"
        "The outer test participants are predicted once per selected layout size. "
        "The reported fold standard deviations use ddof=1. Sensor frequencies count folds. "
        "If classes share the minimum F1, the lowest activity ID is reported. "
        "Ranks count removal steps for backward selection (the final survivor ranks last), "
        "or entry steps for forward selection and gain prefixes. Step times include fitting and prediction. "
        "The Pareto flag compares participant-equal held-out scores against smaller layouts and is descriptive. "
        "Use the corresponding fold of per_fold_selected_subsets.json in downstream held-out work.\n",
        encoding="utf-8",
    )
    write_json(output / "summary.json", {
        "complete": True, "config": config, "pareto": rows, "per_fold_pareto": fold_rows,
        "sensor_rank_stability": stability, "step_timing_seconds": {f: v["step_timing_seconds"] for f, v in details.items()},
        "anti_leakage_audit": {f: v["anti_leakage_audit"] for f, v in details.items()},
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--family", choices=("F1H", "F3", "F4"), default="F1H")
    parser.add_argument("--strategy", choices=(BACKWARD, FORWARD, RANKING), default=BACKWARD)
    parser.add_argument("--window-sec", type=int, choices=(2, 4), default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument("--fast-smoke", action="store_true")
    parser.add_argument("--sensors", nargs="+", choices=SENSORS,
                        help="Restrict the starting layout for a smoke test")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
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
        "n_estimators": args.n_estimators, "report_version": 2,
    }
    run_id = begin_run(args.out, config, args.overwrite)
    if run_id is None:
        return
    selections = {}
    predictions = []
    details = {}
    for fold in folds:
        layout, result, info = run_fold(args, fold, args.out / f"fold_{fold['fold']}", run_id, sensors)
        selections[str(fold["fold"])] = layout
        details[str(fold["fold"])] = info
        predictions.append(result)
        print(f"Fold {fold['fold']} complete", flush=True)
    report(args.out, config, selections, pd.concat(predictions, ignore_index=True), sensors, details)


if __name__ == "__main__":
    main()
