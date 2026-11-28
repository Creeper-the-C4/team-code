"""Paper aggregates, official trial bins and privacy checks."""

import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analysis import build_task12_paper_figures as paper
from scripts.preprocessing.independent_data_access import SENSORS


def synthetic_oof():
    records, bounds = [], []
    for subject in (1, 2, 3, 4, 25, 26, 27, 28):
        bounds.append({"subject": subject, "trial": 0, "left": 600, "right": 1800})
        for i in range(10):
            label = 3 if i < 5 else 4
            records.append({"subject": subject, "trial": 0, "source_start": 600 + 120*i,
                            "source_stop_exclusive": 720 + 120*i, "purity": .8 if i < 2 else .95,
                            "y_true": label, "y_pred": 0 if i < 2 else label,
                            "y_pred_hmm": 0 if i == 0 else label})
    return pd.DataFrame(records), pd.DataFrame(bounds)


def test_preserves_cohort_privacy_and_uses_official_trial_position():
    frame, bounds = synthetic_oof()
    cohorts, strata = paper.oof_strata(frame, bounds, "F1H_2s", "raw", "y_pred")
    cohorts = pd.DataFrame(cohorts)
    single = cohorts[cohorts.cohort.eq("B")]
    assert single[["mean", "q25", "q75"]].isna().all().all()
    assert set(single.n_subjects) == {1}
    assert cohorts.loc[cohorts.cohort.eq("C"), "mean"].isna().all()
    assert cohorts.loc[cohorts.cohort.eq("A"), "mean"].notna().all()
    assert "subject" not in cohorts
    purity = {r["level"]: r["error_rate"] for r in strata if r["stratum"] == "purity_band"}
    assert purity == {"stable": 0., "transition": 1.}
    assert {r["level"] for r in strata if r["stratum"] == "decile"} == set(range(1, 11))
    with pytest.raises(ValueError, match="boundaries"):
        paper.oof_strata(frame, bounds.assign(left=800), "F1H_2s", "raw", "y_pred")


def strategy(directory, delta=0.):
    directory.mkdir()
    selections = {"0": {}, "1": {}}
    rows, ranks, scores = [], [], []
    for k, observed_score in ((2, .87), (3, .895), (4, .915), (17, .95)):
        score = observed_score + delta
        row = {"k": k, "subject_equal_macro_f1_13": score, "pooled_macro_f1_13": score + .015,
               "worst_class_f1": score - .3, "accuracy_13": score + .03,
               "walk3_macro_f1": score, "others_f1": score - .3, "rowing_f1": .98}
        rows.append(row)
        for f, side in enumerate(("LeftFoot", "RightFoot")):
            order = ["LowerBack", side] + [s for s in SENSORS if s not in ("LowerBack", side, "LeftFoot", "RightFoot")]
            order += [s for s in SENSORS if s not in order]
            selections[str(f)][str(k)] = order[:k]
            scores.append(dict(row, fold=f))
        for i, s in enumerate(SENSORS):
            count = sum(s in layouts[str(k)] for layouts in selections.values())
            ranks.append({"sensor": s, "k": k, "folds_selected": count, "fold_count": 2,
                          "selection_frequency": count / 2, "rank_mean": i+1., "rank_sd": .5})
    pd.DataFrame(rows).to_csv(directory / "pareto_curve.csv", index=False)
    pd.DataFrame(scores).to_csv(directory / "per_fold_pareto.csv", index=False)
    pd.DataFrame(ranks).to_csv(directory / "sensor_rank_stability.csv", index=False)
    (directory / "per_fold_selected_subsets.json").write_text(json.dumps({"folds": selections}))


def test_task12_paper_figures_and_threshold_definitions(tmp_path):
    first, second, out = (tmp_path / n for n in ["F1H_backward", "F3_backward", "figures"])
    strategy(first)
    strategy(second, -.015)
    paper.main(["--task2", str(first), str(second), "--out", str(out)])
    assert set(pd.read_csv(out / "task1_ablation.csv").config) == set(paper.CONFIGS)
    for file in ["fig_task1_ablation_raw_vs_hmm.png", "fig_task1_others_and_walk3.png",
                 "fig_task2_pareto.png", "fig_task2_selection_frequency.png", "fig_task2_rank_stability.png"]:
        with Image.open(out / file) as image:
            assert image.width > 850 and np.asarray(image.convert("L")).std() > 12
    thresholds = pd.read_csv(out / "task2_thresholds.csv")
    selected = thresholds[(thresholds.strategy == "F1H_backward") & (thresholds.metric == "subject_equal_macro_f1_13")]
    fixed = selected[selected.basis.eq("absolute")].set_index("target")
    relative = selected[selected.basis.str.startswith("fraction")].set_index("target")
    assert fixed.loc[.9, "minimum_tested_k"] == 4 and fixed.loc[.92, "minimum_tested_k"] == 17
    assert relative.loc[.9, "minimum_tested_k"] == 2 and relative.loc[.92, "minimum_tested_k"] == 3
    rows = pd.read_csv(out / "task2_bilateral_selection.csv").query("strategy == 'F1H_backward' and k == 2 and left == 'LeftFoot'")
    assert tuple(rows.iloc[0][["left_only", "right_only", "both", "neither"]]) == (1, 1, 0, 0)
    assert pd.read_csv(out / "task2_lowerback.csv").selection_frequency.eq(1).all()


def test_cloud_tables_publish_only_aggregates(tmp_path, monkeypatch):
    frame, bounds = synthetic_oof()
    root = tmp_path / "cloud" / "task1_v4" / "F1H_2s"
    root.mkdir(parents=True)
    (root / "oof_predictions.parquet").touch()
    info = tmp_path / "trials.csv"
    pd.DataFrame({"Participant ID": bounds.subject, "Start Index": bounds.left,
                  "End Index": bounds.right - 1}).to_csv(info, index=False)
    out = tmp_path / "public"
    out.mkdir()
    monkeypatch.setattr(pd, "read_parquet", lambda path, columns: frame[columns].copy())
    paper.cloud_tables(root.parent, info, out)
    assert sorted(p.name for p in out.iterdir()) == ["task1_cohort_summary.csv", "task1_purity_and_trial_position.csv"]
    for p in out.iterdir():
        table = pd.read_csv(p)
        assert set(table["mode"]) == {"raw", "hmm"}
        assert not {"subject", "trial", "source_start", "y_true", "y_pred"} & set(table)
        assert not re.search(r"\bP\d{2}\b", p.read_text())
