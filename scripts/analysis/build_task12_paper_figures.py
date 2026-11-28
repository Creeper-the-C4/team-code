"""Generate English paper figures and aggregate tables from completed Task 1/2 outputs."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.preprocessing.independent_data_access import SENSORS
from src.v4_features.workflow import score_predictions

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ["F1H_2s", "F1W_2s", "F1B_2s", "F2p_2s", "F3_2s", "F3_4s", "F4_2s", "F4_4s"]
FOCUS = ["F1B_2s", "F1H_2s", "F3_2s"]
METRICS = {"subject_equal_macro_f1_13": "Subject-equal macro-F1",
           "pooled_macro_f1_13": "Pooled macro-F1", "worst_class_f1": "Worst-class F1"}
COLORS = ["#12676C", "#CF7936", "#456B9B", "#7D5791"]


def save(fig, out, name):
    fig.savefig(out / name, dpi=195, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def task1_figures(root, out):
    base = root / "task1_v4" if (root / "task1_v4").is_dir() else root
    frames = []
    for config in CONFIGS:
        table = pd.read_csv(base / config / "metrics.csv").set_index("mode")
        for mode in ("raw", "hmm"):
            frames.append({"config": config, "mode": mode, **table.loc[mode, list(METRICS)[:2]].to_dict()})
    result = pd.DataFrame(frames)
    result.to_csv(out / "task1_ablation.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained", sharey=True)
    x = np.arange(len(CONFIGS))
    for ax, (metric, title) in zip(axes, list(METRICS.items())[:2]):
        pivot = result.pivot(index="config", columns="mode", values=metric).loc[CONFIGS] * 100
        for j, (mode, label) in enumerate((("raw", "Raw"), ("hmm", "Sticky HMM"))):
            ax.bar(x + (j - 0.5) * 0.38, pivot[mode], width=0.36, color=COLORS[j], label=label)
        ax.set_xticks(x, CONFIGS, rotation=32, ha="right")
        ax.set_ylim(0, 110)
        top = pivot["hmm"].idxmax()
        ax.annotate(f"Best {pivot.loc[top, 'hmm']:.2f}%", (CONFIGS.index(top) + .19, pivot.loc[top, "hmm"]),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("Fixed 13-class macro-F1 (%)")
    axes[1].legend(loc="lower right")
    hmm = result[result["mode"].eq("hmm")].set_index("config")
    qfree = hmm.loc["F1H_2s", "pooled_macro_f1_13"]
    improved = all(qfree > hmm.loc[f, "pooled_macro_f1_13"] for f in ["F3_2s", "F4_2s"])
    note = "F1H uses no quaternions."
    if improved:
        note += " Its 2 s pooled HMM score exceeds both F3 and F4 at 2 s."
    fig.suptitle("Activity recognition on held-out participants\n" + note, fontsize=12)
    save(fig, out, "fig_task1_ablation_raw_vs_hmm.png")

    fig, panels = plt.subplots(2, 3, figsize=(13.2, 8.2), layout="constrained",
                               gridspec_kw={"height_ratios": [1., 1.45]})
    for column, config in enumerate(FOCUS):
        directory = base / config
        cm = pd.read_csv(directory / "walk3_confusion_3x3_hmm.csv", index_col=0).to_numpy(dtype=float)
        share = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm),
                          where=cm.sum(axis=1, keepdims=True) > 0)
        ax = panels[0, column]
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for row, col in np.ndindex(share.shape):
            ax.text(col, row, f"{100 * share[row, col]:.1f}%", ha="center", va="center",
                    color="white" if share[row, col] > .55 else "black")
        ax.set(title=config, xlabel="Predicted walk speed", ylabel="Recorded walk speed")
        ax.set_xticks(range(3), ["Slow", "Moderate", "Brisk"])
        ax.set_yticks(range(3), ["Slow", "Moderate", "Brisk"])
        other = pd.read_csv(directory / "others_error_breakdown.csv").query("mode == 'hmm'").sort_values("class_id")
        ax = panels[1, column]
        y = np.arange(len(other))
        ax.barh(y - .18, other["others_to_class"], height=.34, color=COLORS[0], label="Others predicted as class")
        ax.barh(y + .18, other["class_to_others"], height=.34, color=COLORS[1], label="Class predicted as Others")
        ax.set_yticks(y, other["class_name"].str.replace("_", " "))
        ax.invert_yaxis()
        ax.set_xlabel("Misclassified windows")
        ax.grid(axis="x", alpha=.2)
        if column == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Sticky HMM errors on held-out windows\nWalking cells are row percentages among the three walking predictions")
    save(fig, out, "fig_task1_others_and_walk3.png")


def protocol_bounds(path):
    table = pd.read_csv(path).rename(columns={"Participant ID": "subject", "Start Index": "left",
                                             "End Index": "right"})
    table["trial"] = table.groupby("subject", sort=False).cumcount()
    table["right"] += 1                     # official table stores inclusive end indices
    return table[["subject", "trial", "left", "right"]]


def oof_strata(frame, bounds, config, mode, prediction):
    work = frame.rename(columns={"y_true": "truth"}).copy()
    work["subject"] = work["subject"].astype(str).str.removeprefix("P").astype(int)
    if not work["subject"].between(1, 30).all():
        raise ValueError("Subject outside the registered three cohorts")
    if work.duplicated(["subject", "trial", "source_start"]).any():
        raise ValueError("Duplicate held-out window")
    work["majority_label"] = work["truth"]
    _, people, _ = score_predictions(work, work[prediction])
    people = pd.DataFrame(people)
    people["cohort"] = np.select([people.subject <= 24, people.subject == 25], ["A", "B"], default="C")
    cohorts = []
    base = {"config": config, "mode": mode}
    # A singleton is recoverable by subtraction from published subject-equal totals
    # if both neighbouring means are released. Suppress a complementary mean too.
    sizes = people.groupby("cohort").size()
    singleton = set(sizes[sizes < 2].index)
    others = sizes[~sizes.index.isin(singleton)]
    complement = others.idxmin() if singleton and not others.empty else None
    for group, members in people.groupby("cohort", sort=True):
        for metric in ["macro_f1_13", "accuracy_13"]:
            values = members[metric]
            cohorts.append({**base, "cohort": group, "n_subjects": len(values), "metric": metric,
                            "mean": values.mean() if group not in singleton and group != complement else np.nan,
                            "q25": values.quantile(.25) if len(values) >= 3 else np.nan,
                            "q75": values.quantile(.75) if len(values) >= 3 else np.nan,
                            "mean_status": "singleton suppressed" if group in singleton else
                                           "complementary suppression" if group == complement else "released"})
    work = work.merge(bounds, on=["subject", "trial"], how="left", validate="many_to_one")
    if work[["left", "right"]].isna().any().any() or (work.source_start < work.left).any() or (
        work.source_stop_exclusive > work.right).any() or (work.source_start >= work.source_stop_exclusive).any():
        raise ValueError("OOF windows do not match the official trial boundaries")
    midpoint = (work.source_start + work.source_stop_exclusive) / 2
    work["decile"] = np.minimum(9, np.floor(10 * (midpoint - work.left) / (work.right - work.left))).astype(int) + 1
    if not work["purity"].between(0, 1).all():
        raise ValueError("Invalid window purity")
    work["purity_band"] = np.where(work.purity < .9, "transition", "stable")
    work["error"] = work["truth"] != work[prediction]
    stratified = []
    for field in ["purity_band", "decile"]:
        for value, rows in work.groupby(field):
            count = rows.subject.nunique()
            # No per-person strata outside the designated group either.
            if count >= 2:
                stratified.append({**base, "stratum": field, "level": value, "n_subjects": count,
                                   "n_windows": len(rows), "error_rate": rows.error.mean()})
    return cohorts, stratified


def cloud_tables(root, trial_info, out):
    files = {}
    for path in root.rglob("oof_predictions.parquet"):
        name = path.parent.name
        if name in CONFIGS:
            if name in files:
                raise ValueError(f"Multiple OOF runs for {name}; select a single cloud run")
            files[name] = path
    if not files:
        raise ValueError("No Task 1 OOF files found under the supplied cloud root")
    bounds = protocol_bounds(trial_info)
    cohorts, strata = [], []
    columns = ["subject", "trial", "source_start", "source_stop_exclusive", "purity", "y_true",
               "y_pred", "y_pred_hmm"]
    for name, path in sorted(files.items()):
        frame = pd.read_parquet(path, columns=columns)
        for mode, col in (("raw", "y_pred"), ("hmm", "y_pred_hmm")):
            c, s = oof_strata(frame, bounds, name, mode, col)
            cohorts.extend(c)
            strata.extend(s)
    pd.DataFrame(cohorts).to_csv(out / "task1_cohort_summary.csv", index=False)
    pd.DataFrame(strata).to_csv(out / "task1_purity_and_trial_position.csv", index=False)


def load_task2(argument):
    label, sep, value = argument.partition("=")
    directory = Path(value if sep else argument)
    label = label if sep else directory.name
    # Whitelist public fields to exclude per-subject data and local source paths.
    curve = pd.read_csv(directory / "pareto_curve.csv").sort_values("k")
    folds = pd.read_csv(directory / "per_fold_pareto.csv")
    ranks = pd.read_csv(directory / "sensor_rank_stability.csv")
    if curve["k"].duplicated().any() or folds.duplicated(["fold", "k"]).any() or ranks.duplicated(["sensor", "k"]).any():
        raise ValueError(f"Duplicate Task 2 result in {label}")
    if sorted(ranks.sensor.unique()) != sorted(SENSORS):
        raise ValueError("Rank table must cover all 17 sensors")
    if set(curve.k) != set(folds.k) or set(curve.k) != set(ranks.k):
        raise ValueError(f"Curve, folds and stability tables cover different sizes in {label}")
    path = directory / "per_fold_selected_subsets.json"
    layouts = json.loads(path.read_text(encoding="utf-8"))["folds"] if path.is_file() else None
    return label, curve, folds, ranks, layouts


def task2_summary(runs, out):
    thresholds, lower_back, pairs, activities, rank_rows = [], [], [], [], []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), layout="constrained")
    freq_fig, freq_axes = plt.subplots(1, len(runs), figsize=(max(5.8, 3.9 * len(runs)), 6.5),
                                       squeeze=False, layout="constrained", sharey=True)
    for index, (label, curve, folds, ranks, layouts) in enumerate(runs):
        color = COLORS[index % len(COLORS)]
        for ax, (metric, title) in zip(axes, METRICS.items()):
            ax.plot(curve.k, 100 * curve[metric], marker=["o", "s", "^", "D"][index % 4], ms=3,
                    color=color, label=label)
            q = folds.groupby("k")[metric].quantile([.25, .75]).unstack()
            ax.fill_between(q.index, 100 * q[.25], 100 * q[.75], alpha=.10, color=color)
            ax.set(xlabel="Observed sensors (k)", title=title, ylabel="Score (%)")
            ax.grid(alpha=.2)
        for metric in [*list(METRICS)[:2], "accuracy_13"]:
            if metric not in curve:
                continue
            ref = curve.loc[curve.k.eq(len(SENSORS)), metric]
            for target in (.90, .92):
                comparisons = {"absolute": target}
                if len(ref):
                    comparisons["fraction of 17-sensor score"] = target * ref.iloc[0]
                for basis, threshold in comparisons.items():
                    ks = curve.loc[curve[metric] >= threshold, "k"]
                    thresholds.append({"strategy": label, "metric": metric, "target": target, "basis": basis,
                                       "required_score": threshold, "minimum_tested_k": ks.min() if len(ks) else np.nan})
        for row in ranks[ranks.sensor.eq("LowerBack")].itertuples():
            lower_back.append({"strategy": label, "k": row.k, "selection_frequency": row.selection_frequency,
                               "folds_selected": row.folds_selected, "fold_count": row.fold_count})
        for row in ranks[ranks.k.eq(ranks.k.min())].itertuples():
            rank_rows.append({"strategy": label, "sensor": row.sensor, "rank_mean": row.rank_mean, "rank_sd": row.rank_sd})
        for metric in ("walk3_macro_f1", "rowing_f1", "others_f1", "worst_class_f1"):
            if metric in curve:
                full = curve.loc[curve.k.eq(len(SENSORS)), metric]
                for k, value in zip(curve.k, curve[metric]):
                    activities.append({"strategy": label, "k": k, "metric": metric, "score": value,
                                       "change_from_17": value - full.iloc[0] if len(full) else np.nan})
        if layouts is not None:
            for k in curve.k:
                expected = folds.loc[folds.k.eq(k), "fold"].astype(str)
                choices = [layouts[f][str(k)] for f in expected]
                if any(len(v) != k or len(set(v)) != k or set(v) - set(SENSORS) for v in choices):
                    raise ValueError(f"Invalid per-fold layout for {label}, k={k}")
                for left in (s for s in SENSORS if s.startswith("Left")):
                    right = left.replace("Left", "Right", 1)
                    counts = np.zeros((2, 2), dtype=int)
                    for selection in choices:
                        counts[int(left in selection), int(right in selection)] += 1
                    pairs.append({"strategy": label, "k": k, "left": left, "right": right, "n_folds": len(choices),
                                  "left_only": counts[1, 0], "right_only": counts[0, 1],
                                  "both": counts[1, 1], "neither": counts[0, 0]})
        pivot = ranks.pivot(index="sensor", columns="k", values="selection_frequency").reindex(SENSORS)
        if pivot.isna().any().any():
            raise ValueError(f"Missing sensor frequencies in {label}")
        ax = freq_axes[0, index]
        image = ax.imshow(pivot, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set(title=label.replace("_", " "), xlabel="Observed sensors (k)")
        ticks = np.arange(0, len(pivot.columns), 2 if len(pivot.columns) > 10 else 1)
        ax.set_xticks(ticks, pivot.columns[ticks])
        ax.set_yticks(np.arange(len(SENSORS)), SENSORS)
    axes[0].legend(fontsize=7)
    fig.suptitle("Held-out recognition by layout size | shading: fold interquartile range")
    save(fig, out, "fig_task2_pareto.png")
    freq_fig.colorbar(image, ax=list(freq_axes.flat), label="Fraction of folds selected", shrink=.7)
    save(freq_fig, out, "fig_task2_selection_frequency.png")
    ranks = pd.DataFrame(rank_rows)
    fig, ax = plt.subplots(figsize=(max(5.8, 1.65 * len(runs)), 7), layout="constrained")
    means = ranks.pivot(index="sensor", columns="strategy", values="rank_mean").reindex(SENSORS)
    sds = ranks.pivot(index="sensor", columns="strategy", values="rank_sd").reindex_like(means)
    image = ax.imshow(means, cmap="cividis", aspect="auto", vmin=1, vmax=len(SENSORS))
    for i, j in np.ndindex(means.shape):
        value, sd = means.iloc[i, j], sds.iloc[i, j]
        label = f"{value:.1f}" if pd.isna(sd) else f"{value:.1f} ± {sd:.1f}"
        ax.text(j, i, label, ha="center", va="center", fontsize=8, color="white" if value < 9 else "black")
    ax.set_xticks(np.arange(len(means.columns)), [s.replace("_", "\n") for s in means.columns])
    ax.set_yticks(np.arange(len(SENSORS)), SENSORS)
    ax.set_title("Sensor rank, mean ± fold SD\nBackward ranks count removal steps")
    fig.colorbar(image, ax=ax, label="Mean step")
    save(fig, out, "fig_task2_rank_stability.png")
    pd.DataFrame(thresholds).to_csv(out / "task2_thresholds.csv", index=False)
    pd.DataFrame(lower_back).to_csv(out / "task2_lowerback.csv", index=False)
    pd.DataFrame(pairs, columns=["strategy","k","left","right","n_folds","left_only","right_only","both","neither"]
                 ).to_csv(out / "task2_bilateral_selection.csv", index=False)
    pd.DataFrame(activities).to_csv(out / "task2_activity_scores.csv", index=False)
    ranks.to_csv(out / "task2_rank_summary.csv", index=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task1-root", type=Path, default=ROOT / "results/task1_v4/run_20261128_full")
    p.add_argument("--cloud-results-root", type=Path, help="Optional private OOF directory")
    p.add_argument("--trial-info", type=Path, default=ROOT / "scripts/preprocessing/trial_information.csv")
    p.add_argument("--task2", action="extend", nargs="+", default=[], metavar="LABEL=DIR",
                   help="One or more completed Task 2 strategy directories")
    p.add_argument("--out", type=Path, default=Path("results/task12_paper_figures"))
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    task1_figures(a.task1_root, a.out)
    if a.cloud_results_root:
        cloud_tables(a.cloud_results_root, a.trial_info, a.out)
    if a.task2:
        runs = [load_task2(v) for v in a.task2]
        if len(set(v[0] for v in runs)) != len(runs):
            p.error("Use a distinct label for each Task 2 run")
        task2_summary(runs, a.out)


if __name__ == "__main__":
    main()
