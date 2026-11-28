"""Train and evaluate a sensor-dropout BiGRU on participant-wise folds of independent_v1."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch

from scripts.preprocessing.independent_data_access import SENSORS
from scripts.task3.run_task3_wnoj import DEFAULT_H1
from src.task3_dl.dataset import ACCEL_G_TO_SI, TrainingWindows, fit_scaler
from src.task3_dl.evaluation import aggregate, evaluate_subject, layouts_for_fold
from src.task3_dl.model import SensorDropoutBiGRU, reconstruction_loss
from src.v4_features.workflow import file_sha256, load_folds, write_json


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--body-dims", type=Path, help="Optional standing-calibrated V3 endpoint comparison")
    p.add_argument("--out", type=Path, default=Path("results/task3_bigru_v4"))
    p.add_argument("--checkpoint-dir", type=Path, help="Private directory outside the public output")
    p.add_argument("--fold-subsets-json", type=Path)
    p.add_argument("--subset-k", "--h3-k", type=int, nargs="+", default=[3, 4, 5, 6])
    p.add_argument("--schemes", nargs="+", choices=["H1", "H3"])
    p.add_argument("--h1-sensors", nargs="+", default=DEFAULT_H1)
    p.add_argument("--folds", type=int, nargs="+")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--samples-per-epoch", type=int, default=256)
    p.add_argument("--train-frames", type=int, default=120)
    p.add_argument("--chunk-seconds", type=float, default=10.)
    p.add_argument("--max-trials-per-activity", type=int, default=2)
    p.add_argument("--max-seconds-per-trial", type=float, default=30.)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--apply-wnoj-post", action="store_true", help="Also evaluate orientation-only WNOJ post-smoothing")
    p.add_argument("--fast-smoke", action="store_true", help="Small training model/budget; retain test chunk boundaries")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Reuse completed folds with matching inputs and options")
    mode.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    if min(a.epochs, a.batch_size, a.samples_per_epoch, a.patience) < 1 or a.train_frames < 2:
        p.error("Training budgets must be positive and training crops must contain at least two frames")
    if a.chunk_seconds < 2 or a.max_trials_per_activity < 0 or a.max_seconds_per_trial < 0:
        p.error("Use test chunks >= 2 seconds and nonnegative trial limits")
    if a.h1_sensors == ["all"]:
        a.h1_sensors = list(SENSORS)
    if a.schemes is None:
        a.schemes = ["H1", "H3"] if a.fold_subsets_json else ["H1"]
    if a.fast_smoke:
        a.epochs = min(a.epochs, 2)
        a.batch_size = min(a.batch_size, 4)
        a.samples_per_epoch = min(a.samples_per_epoch, 8)
    return a


def fit_model(args, fold, scaler, device, checkpoint):
    seed = args.seed + int(fold["fold"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train = TrainingWindows(args.data_root, fold["train"], args.train_frames)
    validation = TrainingWindows(args.data_root, fold["validation"], args.train_frames)
    val_rng = np.random.default_rng(seed + 10000)
    fixed_val = [validation.sample(val_rng, scaler, i) for i in range(len(validation.sources))
                 for _ in range(1 if args.fast_smoke else 4)]
    model = SensorDropoutBiGRU(hidden=32 if args.fast_smoke else 192,
                              layers=1 if args.fast_smoke else 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4)
    use_amp = device.type == "cuda"
    amp = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, best_epoch = float("inf"), 0
    history = []

    def batch_loss(samples):
        x, y, observed = (torch.stack(parts).to(device) for parts in zip(*samples))
        with torch.autocast(device.type, enabled=use_amp):
            pred = model(x)
        return reconstruction_loss(pred, y, observed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for start in range(0, args.samples_per_epoch, args.batch_size):
            count = min(args.batch_size, args.samples_per_epoch - start)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss([train.sample(rng, scaler) for _ in range(count)])
            amp.scale(loss).backward()
            amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp.step(optimizer)
            amp.update()
        model.eval()
        total = 0.
        with torch.inference_mode():
            for start in range(0, len(fixed_val), args.batch_size):
                samples = fixed_val[start:start + args.batch_size]
                total += float(batch_loss(samples)) * len(samples)
        score = total / len(fixed_val)
        if not np.isfinite(score):
            raise ValueError("Non-finite validation loss")
        history.append(score)
        if score < best:
            best, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "scaler_mean": torch.from_numpy(scaler.mean),
                        "scaler_std": torch.from_numpy(scaler.std)}, checkpoint)
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    return model.eval(), {"best_epoch": best_epoch, "validation_loss_rad": history,
                          "parameter_count": sum(p.numel() for p in model.parameters())}


@contextmanager
def output_lock(path):
    """An OS-held lock is released automatically if a run is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


TABLES = ("per_sensor_metrics.csv", "per_activity_metrics.csv", "ablation_summary.csv",
          "v3_endpoint_metrics.csv")


def dataset_stamp(source, subjects):
    """Invalidate resumable folds when a local signal, label, or trial index is replaced."""
    stamp = hashlib.sha256()
    for subject in sorted(subjects):
        folder = source / "subjects" / f"P{subject:02d}"
        for name in ("signals.npy", "labels.npy", "trials.parquet", "complete.json"):
            path = folder / name
            if path.exists():
                st = path.stat()
                stamp.update(f"{subject}:{name}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    return stamp.hexdigest()


def run(a):
    source, out = a.data_root.resolve(), a.out.resolve()
    private = (a.checkpoint_dir or out.parent / "checkpoints" / out.name).resolve()
    if private == out or out in private.parents:
        raise ValueError("Keep private weights and frame-level diagnostics outside --out")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    folds = load_folds(a.splits, a.fast_smoke)
    if a.folds is not None:
        if a.fast_smoke:
            # The shared validator checks the entire split, then defaults to its first two folds.
            folds = json.loads(a.splits.read_text(encoding="utf-8"))["folds"]
        requested = set(a.folds)
        folds = [fold for fold in folds if fold["fold"] in requested]
        if requested != {fold["fold"] for fold in folds}:
            raise ValueError("Requested fold is absent from the selected protocol")
    subsets = json.loads(a.fold_subsets_json.read_text(encoding="utf-8")) if a.fold_subsets_json else None
    masks = {str(f["fold"]): layouts_for_fold(f["fold"], a.schemes, a.h1_sensors, subsets, a.subset_k)
             for f in folds}
    settings = {k: v for k, v in vars(a).items() if not isinstance(v, Path)
                and k not in ("resume", "overwrite", "checkpoint_dir", "body_dims", "fold_subsets_json")}
    inputs = {name: file_sha256(path) for name, path in (("splits", a.splits), ("subsets", a.fold_subsets_json),
                                                         ("dimensions", a.body_dims)) if path}
    participants = {s for f in folds for role in ("train", "validation", "test") for s in f[role]}
    inputs["continuous_cache"] = dataset_stamp(source, participants)
    signature = json.dumps([3, str(source), str(private), device.type, settings, inputs, masks], sort_keys=True)
    key = hashlib.sha256(signature.encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "summary.json"
    lock_path = out.parent / "checkpoints" / (out.name + ".lock")
    with output_lock(lock_path):
        if marker.exists() and not a.overwrite:
            prior = json.loads(marker.read_text(encoding="utf-8"))
            if prior.get("run_id") != key or not a.resume:
                raise ValueError("Output exists. Use a matching --resume, --overwrite, or a new --out")
            if prior.get("complete") and all((out / n).is_file() for n in TABLES):
                print("Matching run already complete.", flush=True)
                return
        dims = None
        if a.body_dims:
            dims = {int(r.subject): r._asdict() for r in pd.read_csv(a.body_dims).itertuples(index=False)}
        fold_dirs, reports = [], []
        started = time.perf_counter()
        for fold in folds:
            number = fold["fold"]
            folder = private / key[:20] / f"fold-{number}"
            completed = folder / "complete.json"
            state = json.loads(completed.read_text(encoding="utf-8")) if a.resume and completed.is_file() else None
            if state:
                if state["run_id"] != key:
                    raise ValueError("Saved fold belongs to a different run")
                part = folder / state["attempt"]
                if not (part / "best.pt").is_file() or len(list((part / "errors").glob("*/*/*/*.npz"))) != state["files"]:
                    state = None
            if state:
                report = state["report"]
            else:
                part = folder / uuid4().hex[:12]
                part.mkdir(parents=True)
                scaler = fit_scaler(source, fold)
                model, report = fit_model(a, fold, scaler, device, part / "best.pt")
                calibrated = 0
                for subject in fold["test"]:
                    calibrated += evaluate_subject(model, scaler, device, source, subject, masks[str(number)],
                                                   a, part / "errors", dims)
                report.update(fold=number, training_subject_count=len(fold["train"]),
                              validation_subject_count=len(fold["validation"]),
                              test_subject_count=len(fold["test"]), calibrated_v3_subject_count=calibrated)
                write_json(completed, {"run_id": key, "attempt": part.name, "report": report,
                                       "files": len(list((part / "errors").glob("*/*/*/*.npz")))})
            reports.append(report)
            fold_dirs.append(part)
            print(f"Fold {number} complete; best epoch {report['best_epoch']}.", flush=True)
        tables = aggregate(fold_dirs, masks)
        write_json(marker, {"complete": False, "run_id": key})
        for name, table in tables.items():
            temp = out / (name + ".tmp")
            table.to_csv(temp, index=False)
            temp.replace(out / name)
        write_json(marker, {
            "complete": True, "run_id": key, "reference_type": "withheld_sensor_reference",
            "accel_g_to_si": ACCEL_G_TO_SI, "accel_semantics": "gravity_compensated_linear_acceleration",
            "gyro_unit": "rad/s", "orientation_representation": "first two matrix columns",
            "input_masks": masks, "folds": reports, "settings": settings, "source_digests": inputs,
            "v3_calibration_note": "V3 uses all-sensor standing calibration for derived endpoint displacement. "
                                   "Only V1 and V2 are strictly withheld. Coverage is counted per fold.",
            "v1_subject_equal": "arithmetic mean of the subject-specific median angles",
            "evaluation": "offline bidirectional, within official test trials; metrics use frame activity labels",
            "methods": ["T3-M4", "T3-M5"] if a.apply_wnoj_post else ["T3-M4"],
            "wnoj_post": {"qc": 1000., "sigma_prior_deg": 4., "withheld_gyro_used": False} if a.apply_wnoj_post else None,
            "device_type": device.type, "runtime_seconds": round(time.perf_counter() - started, 3),
        })


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
