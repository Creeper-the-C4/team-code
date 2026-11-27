"""Evaluate an IMU feature family under the fixed participant-wise folds."""

import argparse
from pathlib import Path
import sys
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.v4_features.feature_families import FAMILIES
from src.v4_features.sticky_hmm import fit_sticky_hmm, viterbi_decode_trials
from src.v4_features.workflow import (
    ACTIVITY, PROBABILITY_COLUMNS, WINDOW_COLUMNS, begin_run, cached_fold, class_metrics, file_sha256,
    fit_classifier, load_folds, load_split, predict_probabilities, read_json, score_predictions, write_json,
)


def run_fold(args, fold, directory, run_id):
    directory.mkdir(parents=True, exist_ok=True)
    prediction_file = directory / "test_predictions.parquet"
    if cached_fold(directory, run_id) and prediction_file.exists():
        return pd.read_parquet(prediction_file), read_json(directory / "complete.json")["fold_metrics"]

    def get_data(role):
        return load_split(args.data_root, args.feature_cache, fold[role], args.family, args.window_sec,
                          cap=24 if args.fast_smoke else None)

    train, train_meta = get_data("train")
    validation, validation_meta = get_data("validation")
    rounds = 8 if args.fast_smoke else args.n_estimators
    started = perf_counter()
    model = fit_classifier(train, train_meta, validation, validation_meta, rounds, args.fast_smoke)
    training_seconds = perf_counter() - started
    val_scores = score_predictions(validation_meta, predict_probabilities(model, validation).argmax(axis=1))[0]
    hmm_params = None
    if args.enable_sticky_hmm:
        hmm_params = fit_sticky_hmm(train_meta, dt_seconds=args.window_sec / 2)
        write_json(directory / "hmm_parameters.json", {
            key: value.tolist() if hasattr(value, "tolist") else value for key, value in hmm_params.items()
        } | {"fitting_subjects": fold["train"]})
    elif args.overwrite:
        (directory / "hmm_parameters.json").unlink(missing_ok=True)
    del train, validation
    test, test_meta = get_data("test")
    probabilities = predict_probabilities(model, test)
    keep = WINDOW_COLUMNS + [key for key in ("window_id", "purity", "mixed_labels") if key in test_meta]
    predictions = test_meta[keep].copy()
    predictions["fold"] = fold["fold"]
    predictions["y_true"] = predictions["majority_label"]
    predictions["y_pred"] = probabilities.argmax(axis=1)
    predictions[PROBABILITY_COLUMNS] = probabilities
    if hmm_params is not None:
        predictions["y_pred_hmm"] = viterbi_decode_trials(predictions, PROBABILITY_COLUMNS, hmm_params)
    predictions.to_parquet(prediction_file, index=False)
    test_scores = score_predictions(test_meta, predictions["y_pred"])[0]
    fold_metrics = {
        "fold": fold["fold"], "training_seconds": training_seconds, "best_iteration": int(model.best_iteration_),
        "validation_macro_f1": val_scores["pooled_macro_f1_13"],
        "test_pooled_macro_f1_13": test_scores["pooled_macro_f1_13"],
        "test_subject_equal_macro_f1_13": test_scores["subject_equal_macro_f1_13"],
    }
    if hmm_params is not None:
        fold_metrics["dt_seconds"] = hmm_params["dt_seconds"]
    write_json(directory / "complete.json", {
        "complete": True, "run_id": run_id, "participants": fold, "windows": len(predictions),
        "best_iteration": int(model.best_iteration_), "fold_metrics": fold_metrics,
    })
    return predictions, fold_metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--window-sec", type=int, choices=(2, 4), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--enable-sticky-hmm", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--fast-smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.n_estimators < 2:
        parser.error("--n-estimators must be at least 2")
    folds = load_folds(args.splits, args.fast_smoke)
    config = {
        "data_root": str(args.data_root.resolve()), "feature_cache": str(args.feature_cache.resolve()),
        "split_sha256": file_sha256(args.splits), "family": args.family, "window_seconds": args.window_sec,
        "sticky_hmm": args.enable_sticky_hmm, "fast_smoke": args.fast_smoke,
        "n_estimators": 8 if args.fast_smoke else args.n_estimators, "report_version": 2,
    }
    run_id = begin_run(args.out, config, args.overwrite)
    if run_id is None:
        return
    frames = []
    fold_rows = []
    for fold in folds:
        predictions, fold_row = run_fold(args, fold, args.out / f"fold_{fold['fold']}", run_id)
        frames.append(predictions)
        fold_rows.append(fold_row)
        print(f"Fold {fold['fold']} complete", flush=True)
    oof = pd.concat(frames, ignore_index=True).sort_values(["subject", "trial", "source_start"])
    if oof.duplicated(["subject", "trial", "source_start"]).any():
        raise ValueError("A held-out window was predicted more than once")
    oof.to_parquet(args.out / "oof_predictions.parquet", index=False)
    modes = {"raw": "y_pred"}
    if args.enable_sticky_hmm:
        modes["hmm"] = "y_pred_hmm"
    metrics = []
    people = []
    classes = []
    others = []
    summary = {"complete": True, "config": config}
    for mode, column in modes.items():
        scores, individual, cm = score_predictions(oof, oof[column])
        summary[mode] = scores
        classes.append(class_metrics(cm).assign(mode=mode))
        walk_names = [ACTIVITY[i] for i in (4, 5, 6)]
        walk = pd.DataFrame(cm[4:7, 4:7], index=walk_names, columns=walk_names)
        suffix = "_hmm" if mode == "hmm" else ""
        walk.to_csv(args.out / f"walk3_confusion_3x3{suffix}.csv", index_label="true_activity")
        others.extend({"mode": mode, "class_id": i, "class_name": ACTIVITY[i],
                       "others_to_class": int(cm[0, i]), "class_to_others": int(cm[i, 0])}
                      for i in range(1, len(cm)))
        metrics.append({"mode": mode, **scores})
        people.extend({"mode": mode, **row} for row in individual)
    pd.DataFrame(metrics).to_csv(args.out / "metrics.csv", index=False)
    pd.DataFrame(people).to_csv(args.out / "per_subject_metrics.csv", index=False)
    pd.concat(classes, ignore_index=True).to_csv(args.out / "per_class_metrics.csv", index=False)
    pd.DataFrame(others).to_csv(args.out / "others_error_breakdown.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.out / "fold_summary.csv", index=False)
    if args.enable_sticky_hmm:
        steps = {row["dt_seconds"] for row in fold_rows}
        if len(steps) != 1:
            raise ValueError("Expected the same window stride across training folds")
        summary["dt_seconds"] = steps.pop()
        delta = {}
        for metric in ("pooled_macro_f1_13", "subject_equal_macro_f1_13", "walk3_macro_f1", "others_f1", "rowing_f1"):
            raw, hmm = summary["raw"][metric], summary["hmm"][metric]
            delta[metric] = {"raw": raw, "hmm": hmm, "delta": hmm - raw}
        write_json(args.out / "hmm_delta_summary.json", delta)
    elif args.overwrite:
        for name in ("hmm_delta_summary.json", "walk3_confusion_3x3_hmm.csv"):
            (args.out / name).unlink(missing_ok=True)
    write_json(args.out / "summary.json", summary)


if __name__ == "__main__":
    main()
