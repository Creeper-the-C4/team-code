"""Summarize available cloud experiments and archive small reporting artifacts."""

import argparse
import csv
from hashlib import sha256
import io
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile


STAGES = {
    "p0_diagnostics": "p0_diagnostics",
    "task1_v4": "task1",
    "task2_v4": "task2",
    "task3_v4": "task3",
    "task3_v4_accel_g": "task3_accel_g",
    "task3_bigru_v4": "task3_bigru_v4",
    "spine_p0_calibration": "spine_p0_calibration",
    "task12_paper_figures": "task12_paper_figures",
    "task5_bigru": "task5_bigru",
}
ROOT_FILES = ("run_manifest.json", "stage_status.json", "timing_by_stage.csv", "hardware_environment.json")
EXTRAS = ("figures", "logs", "failures", "final_report")
ALLOWED = {".json", ".csv", ".md", ".txt", ".log", ".png", ".svg"}
GENERATED = {"experiment_registry.json", "final_report/summary_report.md", "final_report/rowing_2s_vs_4s.csv"}
PREFIX = "cloud_v4_final_bundle/"
MAX_BYTES = 32 * 1024 * 1024


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


SKIP_DIRS = {'_parts', '_checkpoints'}   # resume intermediates never enter the bundle


def collect(root):
    files = {}
    oversized = []

    def add(path, name):
        if name in GENERATED or path.is_symlink() or not path.is_file() or path.suffix.lower() not in ALLOWED:
            return
        if path.stat().st_size > MAX_BYTES:
            oversized.append(name)
        else:
            files[name] = path

    for name in ROOT_FILES:
        add(root / name, name)
    for source, target in (STAGES | {name: name for name in EXTRAS}).items():
        folder = root / source
        if folder.is_symlink():
            continue
        for parent, dirs, names in os.walk(folder):
            dirs[:] = sorted(n for n in dirs if not (Path(parent) / n).is_symlink()
                             and n not in SKIP_DIRS)
            for name in sorted(names):
                path = Path(parent) / name
                add(path, f"{target}/{path.relative_to(folder).as_posix()}")
    return files, oversized


def summarize(files):
    experiments = []
    rowing = []
    for name, path in sorted(files.items()):
        stage = name.split("/")[0]
        if stage not in STAGES.values() or path.name != "summary.json":
            continue
        summary = json.loads(path.read_text(encoding="utf-8-sig"))
        config_name = name.removesuffix("summary.json") + "config.json"
        config = summary.get("config", {})
        digest = None
        source = None
        if config_name in files:
            raw = files[config_name].read_bytes()
            digest, source = sha256(raw).hexdigest(), config_name
            external = json.loads(raw.decode("utf-8-sig"))
            config = external.get("parameters", external) if not config else config
        elif "config" in summary:
            raw = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            digest, source = sha256(raw).hexdigest(), "canonical summary.config"
        metrics = {k: v for k, v in summary.items() if k not in ("config", "complete")}
        experiments.append({"stage": stage, "summary": name, "complete": summary.get("complete"),
                            "config_sha256": digest, "config_source": source, "metrics": metrics})
        window = config.get("window_seconds") if isinstance(config, dict) else None
        if stage == "task1" and summary.get("complete", True) and window in (2, 4):
            for mode in ("raw", "hmm"):
                if isinstance(summary.get(mode), dict) and isinstance(summary[mode].get("rowing_f1"), (int, float)):
                    rowing.append({"family": config.get("family"), "mode": mode, "window_seconds": window,
                                   "rowing_f1": summary[mode]["rowing_f1"], "config_sha256": digest,
                                   "summary": name})
    return experiments, rowing


def package(root, destination):
    if not root.is_dir() or destination.suffix.lower() != ".zip":
        raise ValueError("Provide an existing results directory and a .zip destination")
    if (root / "final_report").is_symlink():
        raise ValueError("final_report must be a directory inside the results root")
    files, oversized = collect(root)
    experiments, rowing = summarize(files)
    missing = [name for name in ROOT_FILES if name not in files]
    registry = {"experiments": experiments, "missing_root_files": missing, "oversized_files": oversized}
    generated = {"experiment_registry.json": json_bytes(registry)}
    report = ["# Cloud experiment report", "", "| Stage | Summary files |", "| --- | ---: |"]
    for stage in STAGES.values():
        report.append(f"| {stage} | {sum(e['stage'] == stage for e in experiments)} |")
    report.extend(["", "| Experiment | Result |", "| --- | --- |"])
    for entry in experiments:
        metrics = entry["metrics"]
        values = []
        for mode in ("raw", "hmm"):
            if isinstance(metrics.get(mode), dict):
                for metric in ("subject_equal_macro_f1_13", "rowing_f1"):
                    if isinstance(metrics[mode].get(metric), (int, float)):
                        values.append(f"{mode} {metric}={metrics[mode][metric]:.5g}")
        if isinstance(metrics.get("pareto"), list):
            values.append(f"{len(metrics['pareto'])} layout sizes")
        values.extend(f"{key}={value}" for key, value in metrics.items()
                      if isinstance(value, (int, float)) and not isinstance(value, bool))
        detail = ", ".join(values[:8]) or "See registry"
        report.append(f"| {entry['summary']} | {detail} |")
    report.extend(["", f"Missing root files: {', '.join(missing) or 'none'}.",
                   f"Files above the 32 MiB limit: {len(oversized)}.", "",
                   "Per-experiment metrics and configuration digests are in experiment_registry.json."])
    if {r["window_seconds"] for r in rowing} == {2, 4}:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=list(rowing[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rowing, key=lambda r: (r["family"] or "", r["mode"], r["window_seconds"], r["summary"])))
        generated["final_report/rowing_2s_vs_4s.csv"] = output.getvalue().encode("utf-8")
        report.extend(["", "Rowing F1 for the available 2 s and 4 s runs is in rowing_2s_vs_4s.csv."])
    generated["final_report/summary_report.md"] = ("\n".join(report) + "\n").encode("utf-8")
    if any(len(data) > MAX_BYTES for data in generated.values()):
        raise ValueError("Generated report exceeds the 32 MiB per-file archive limit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    (root / "final_report").mkdir(exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as f:
        temp_zip = Path(f.name)
    try:
        with ZipFile(temp_zip, "w", compression=ZIP_DEFLATED) as archive:
            for directory in (*STAGES.values(), *EXTRAS):
                archive.writestr(PREFIX + directory + "/", b"")
            for name, path in sorted(files.items()):
                archive.write(path, PREFIX + name)
            for name, data in generated.items():
                archive.writestr(PREFIX + name, data)
        for name, data in generated.items():
            target = root / name
            with NamedTemporaryFile(dir=target.parent, delete=False) as f:
                f.write(data)
            Path(f.name).replace(target)
        if "final_report/rowing_2s_vs_4s.csv" not in generated:
            (root / "final_report/rowing_2s_vs_4s.csv").unlink(missing_ok=True)
        temp_zip.replace(destination)
    finally:
        temp_zip.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out-zip", type=Path, required=True)
    args = parser.parse_args(argv)
    package(args.results_root, args.out_zip)


if __name__ == "__main__":
    main()
