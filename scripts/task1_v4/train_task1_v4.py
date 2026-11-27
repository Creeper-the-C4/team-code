"""Evaluate an IMU feature family under the fixed participant-wise folds."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.v4_features.feature_families import FAMILIES
from src.v4_features.sticky_hmm import fit_sticky_hmm, viterbi_decode_trials
from src.v4_features.workflow import (
    PROBABILITY_COLUMNS, WINDOW_COLUMNS, begin_run, cached_fold, completed, file_sha256,
    fit_classifier, load_folds, load_split, predict_probabilities, score_predictions, write_json,
)


def run_fold(args, fold, directory, run_id):
    directory.mkdir(parents=True, exist_ok=True)
    prediction_file = directory / "test_predictions.parquet"
    if cached_fold(directory, run_id) and prediction_file.exists():
        return pd.read_parquet(prediction_file)

    def get_data(role):
        return load_split(args.data_root, args.feature_cache, fold[role], args.family, args.window_sec,
                          cap=24 if args.fast_smoke else None)

    train, train_meta = get_data("train")
    validation, validation_meta = get_data("validation")
    rounds = 8 if args.fast_smoke else args.n_estimators
    model = fit_classifier(train, train_meta, validation, validation_meta, rounds, args.fast_smoke)
    hmm_params = None
    if args.enable_sticky_hmm:
        hmm_params = fit_sticky_hmm(train_meta, dt_seconds=args.window_sec / 2)
        write_json(directory / "hmm_parameters.json", {
            key: value.tolist() if hasattr(value, "tolist") else value for key, value in hmm_params.items()
        } | {"fitting_subjects": fold["train"]})
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
    write_json(directory / "complete.json", {
        "complete": True, "run_id": run_id, "participants": fold, "windows": len(predictions),
        "best_iteration": int(model.best_iteration_),
    })
    return predictions


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
    if not args.overwrite and completed(args.out):
        print(f"Already complete: {args.out}", flush=True)
        return
    if args.n_estimators < 2:
        parser.error("--n-estimators must be at least 2")
    folds = load_folds(args.splits, args.fast_smoke)
    config = {
        "data_root": str(args.data_root.resolve()), "feature_cache": str(args.feature_cache.resolve()),
        "split_sha256": file_sha256(args.splits), "family": args.family, "window_seconds": args.window_sec,
        "sticky_hmm": args.enable_sticky_hmm, "fast_smoke": args.fast_smoke,
        "n_estimators": 8 if args.fast_smoke else args.n_estimators,
    }
    run_id = begin_run(args.out, config, args.overwrite)
    if run_id is None:
        return
    frames = []
    for fold in folds:
        frames.append(run_fold(args, fold, args.out / f"fold_{fold['fold']}", run_id))
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
    summary = {"complete": True, "config": config}
    for mode, column in modes.items():
        scores, individual, _ = score_predictions(oof, oof[column])
        summary[mode] = scores
        metrics.append({"mode": mode, **scores})
        people.extend({"mode": mode, **row} for row in individual)
    pd.DataFrame(metrics).to_csv(args.out / "metrics.csv", index=False)
    pd.DataFrame(people).to_csv(args.out / "per_subject_metrics.csv", index=False)
    write_json(args.out / "summary.json", summary)


if __name__ == "__main__":
    main()
