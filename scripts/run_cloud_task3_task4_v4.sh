#!/usr/bin/env bash
# Post-Task-2 unattended cloud runner for P0-calibrated Task 4, Task 3 (WNOJ none/g + BiGRU 6D/WNOJ),
# Task 1/2 paper figures, and lightweight bundle packaging.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE=0
OVERWRITE=0
SKIP_BIGRU=0
PYTHON_BIN="${PY:-/root/autodl-tmp/C_Problem/.venv/bin/python}"
DATA_ROOT="${DATA:-/root/autodl-tmp/C_Problem/data/processed/independent_v1}"
TASK1_ROOT="results/task1_v4/run_20261128_full"
TASK1_CLOUD_ROOT="/root/autodl-tmp/C_Problem/results/task1_v4_20261128"
TASK2_ROOT=""
RESULTS_ROOT="results"
CHECKPOINT_DIR=""
SPLITS_PATH="results/preprocessing/independent_v1/subject_splits_6fold.json"
BODY_DIMS_PATH="results/preprocessing/independent_v1/body_dimensions_m.csv"
OUT_ZIP=""
WORKERS=8
BIGRU_EPOCHS=40
BIGRU_PATIENCE=6
BIGRU_SAMPLES=1024
BIGRU_FRAMES=240
BIGRU_DEVICE="auto"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_cloud_task3_task4_v4.sh [OPTIONS]

Run Task 4 P0 static-noise calibration, Task 3 WNOJ (T3-M0..M3), Task 3 BiGRU 6D + WNOJ post (T3-M4/M5),
and Task 1/2 paper figures using completed Task 2 F1H_backward per-fold subsets (H3).

Options:
  --smoke                      Run end-to-end synthetic smoke test (CPU-safe, ~1-2 min)
  --python PATH                Python executable (default: cloud .venv or local .venv/python3)
  --data-root PATH             independent_v1 data root
  --task1-root PATH            Completed Task 1 public aggregate dir (default: results/task1_v4/run_20261128_full)
  --task1-cloud-root PATH      Optional private Task 1 OOF dir for cohort/purity/decile stratified tables
  --task2-root PATH            Completed Task 2 directory containing F1H_backward/ (auto-detected if omitted)
  --results-root PATH          Output root (default: results)
  --checkpoint-dir PATH        Private dir for BiGRU checkpoints/errors (outside task3_bigru_v4)
  --splits PATH                Subject splits JSON
  --body-dims PATH             Body dimensions CSV
  --workers N                  Worker processes for Task 3 WNOJ (default: 8)
  --bigru-epochs N             BiGRU max epochs on cloud GPU (default: 40)
  --bigru-patience N           BiGRU early-stopping patience (default: 6)
  --bigru-samples-per-epoch N  Training crops per epoch (default: 1024)
  --bigru-train-frames N       Training crop frames at 60 Hz (default: 240)
  --bigru-device DEV           auto|cuda|cpu (default: auto; smoke forces cpu unless overridden)
  --skip-bigru                 Skip Stage T3-DL (BiGRU 6D + T3-M5)
  --out-zip PATH               Output bundle ZIP (default: <results-root>/cloud_task34_bundle.zip)
  --overwrite                  Ignore cached stage completion markers and recompute
  -h, --help                   Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --task1-root) TASK1_ROOT="$2"; shift 2 ;;
    --task1-cloud-root) TASK1_CLOUD_ROOT="$2"; shift 2 ;;
    --task2-root) TASK2_ROOT="$2"; shift 2 ;;
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --splits) SPLITS_PATH="$2"; shift 2 ;;
    --body-dims) BODY_DIMS_PATH="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --bigru-epochs) BIGRU_EPOCHS="$2"; shift 2 ;;
    --bigru-patience) BIGRU_PATIENCE="$2"; shift 2 ;;
    --bigru-samples-per-epoch) BIGRU_SAMPLES="$2"; shift 2 ;;
    --bigru-train-frames) BIGRU_FRAMES="$2"; shift 2 ;;
    --bigru-device) BIGRU_DEVICE="$2"; shift 2 ;;
    --skip-bigru) SKIP_BIGRU=1; shift ;;
    --out-zip) OUT_ZIP="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Resolve Python binary
if [[ ! -x "$PYTHON_BIN" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "ERROR: Python executable not found ($PYTHON_BIN)" >&2
    exit 1
  fi
fi

# Configure smoke paths or resolve Task 2 output root
if [[ "$SMOKE" -eq 1 ]]; then
  if [[ "$RESULTS_ROOT" == "results" ]]; then
    RESULTS_ROOT="/tmp/task34_smoke_results"
  fi
  if [[ "$DATA_ROOT" == "/root/autodl-tmp/C_Problem/data/processed/independent_v1" ]]; then
    DATA_ROOT="$RESULTS_ROOT/_smoke_data"
  fi
  SPLITS_PATH="$DATA_ROOT/splits_smoke.json"
  BODY_DIMS_PATH="$DATA_ROOT/body_dims_smoke.csv"
  TASK2_ROOT="$RESULTS_ROOT/_smoke_task2"
  TASK1_CLOUD_ROOT=""
  WORKERS=1
  if [[ "$BIGRU_DEVICE" == "auto" ]]; then
    BIGRU_DEVICE="cpu"
  fi
else
  if [[ -z "$TASK2_ROOT" ]]; then
    for cand in \
      "$RESULTS_ROOT/task2_v4" \
      "/root/autodl-tmp/C_Problem/results/task2_v4_20261128" \
      "/root/autodl-tmp/C_Problem/results/task2_v4" \
      "results/task2_v4"; do
      if [[ -f "$cand/F1H_backward/per_fold_selected_subsets.json" || -f "$cand/per_fold_selected_subsets.json" ]]; then
        TASK2_ROOT="$cand"
        break
      fi
    done
  fi
fi

if [[ -z "$CHECKPOINT_DIR" ]]; then
  if [[ "$SMOKE" -eq 0 && -d "/root/autodl-tmp/C_Problem" && -w "/root/autodl-tmp/C_Problem" ]]; then
    CHECKPOINT_DIR="/root/autodl-tmp/C_Problem/checkpoints/task3_bigru_v4"
  else
    CHECKPOINT_DIR="$RESULTS_ROOT/_checkpoints/task3_bigru_v4"
  fi
fi

if [[ -z "$OUT_ZIP" ]]; then
  OUT_ZIP="$RESULTS_ROOT/cloud_task34_bundle.zip"
fi

mkdir -p "$RESULTS_ROOT" "$RESULTS_ROOT/logs" "$RESULTS_ROOT/failures" "$RESULTS_ROOT/final_report"

LOCK_FILE="$RESULTS_ROOT/.task34_run.lock"
exec 9>"$LOCK_FILE"
if command -v flock >/dev/null 2>&1; then
  if ! flock -n 9; then
    echo "ERROR: Another run_cloud_task3_task4_v4.sh process holds $LOCK_FILE" >&2
    exit 1
  fi
fi

# Build synthetic data + Task 2 fixture once in --smoke mode
if [[ "$SMOKE" -eq 1 && ( "$OVERWRITE" -eq 1 || ! -f "$DATA_ROOT/summary.json" || ! -f "$TASK2_ROOT/F1H_backward/per_fold_selected_subsets.json" ) ]]; then
  echo "[SMOKE] Building synthetic Task 3/4 + Task 2 fixture under $RESULTS_ROOT ..."
  "$PYTHON_BIN" - "$DATA_ROOT" "$SPLITS_PATH" "$BODY_DIMS_PATH" "$TASK2_ROOT" <<'PYEOF'
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "scripts" / "diagnostics"))
import make_synthetic_fixture as fix
from scripts.preprocessing.independent_data_access import SENSORS

root = Path(sys.argv[1])
splits_path = Path(sys.argv[2])
dims_path = Path(sys.argv[3])
task2_root = Path(sys.argv[4])

fix.build(root, acc_unit="g", subjects=(1, 2, 3, 4), seed=31)
subs = [1, 2, 3, 4]
splits = {"folds": [
    {"fold": i, "test": [subs[i]], "validation": [subs[(i + 1) % 4]],
     "train": [x for x in subs if x not in (subs[i], subs[(i + 1) % 4])]}
    for i in range(4)
]}
splits_path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")

cols = {
    "height_m": 1.75, "head_height_m": 0.19, "neck_length_m": 0.10, "torso_length_m": 0.50,
    "shoulder_width_m": 0.40, "upper_arm_length_m": 0.30, "forearm_length_m": 0.26,
    "bi_iliac_breadth_m": 0.27, "thigh_length_m": 0.43, "shank_length_m": 0.42,
    "heel_height_m": 0.12, "foot_length_m": 0.26,
}
lines = ["subject," + ",".join(cols)] + [
    f"{s}," + ",".join(str(v) for v in cols.values()) for s in subs
]
dims_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
(root / "summary.json").write_text(json.dumps({"smoke": True, "acc_unit": "g", "subjects": 4}) + "\n", encoding="utf-8")

base_order = [
    "LowerBack", "RightShank", "UpperBack", "Head", "LeftThigh", "RightFoot",
    "LeftFoot", "LeftShank", "RightThigh", "LeftForeArm", "RightForeArm",
    "LeftUpperArm", "RightUpperArm", "LeftShoulder", "RightShoulder", "LeftWrist", "RightWrist",
]
for strat, delta in (("F1H_backward", 0.0), ("F3_backward", -0.01)):
    sdir = task2_root / strat
    sdir.mkdir(parents=True, exist_ok=True)
    selections = {str(f): {} for f in range(4)}
    rows, fold_rows, rank_rows = [], [], []
    for k, score in ((3, 0.88), (4, 0.90), (5, 0.915), (6, 0.925), (17, 0.94)):
        sc = score + delta
        row = {
            "k": k, "subject_equal_macro_f1_13": sc, "pooled_macro_f1_13": sc + 0.015,
            "worst_class_f1": sc - 0.25, "accuracy_13": sc + 0.02,
            "walk3_macro_f1": sc, "others_f1": sc - 0.25, "rowing_f1": 0.97,
        }
        rows.append(row)
        for f in range(4):
            selections[str(f)][str(k)] = base_order[:k]
            fold_rows.append(dict(row, fold=f))
        for i, sensor in enumerate(SENSORS):
            rank_rows.append({
                "sensor": sensor, "k": k, "folds_selected": 4 if sensor in base_order[:k] else 0,
                "fold_count": 4, "selection_frequency": 1.0 if sensor in base_order[:k] else 0.0,
                "rank_mean": float(i + 1), "rank_sd": 0.5,
            })
    pd.DataFrame(rows).to_csv(sdir / "pareto_curve.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(sdir / "per_fold_pareto.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(sdir / "sensor_rank_stability.csv", index=False)
    (sdir / "per_fold_selected_subsets.json").write_text(
        json.dumps({"folds": selections}, indent=2) + "\n", encoding="utf-8"
    )
PYEOF
fi

# Locate Task 2 primary H3 subsets JSON
SUBSETS_JSON=""
if [[ -f "$TASK2_ROOT/F1H_backward/per_fold_selected_subsets.json" ]]; then
  SUBSETS_JSON="$TASK2_ROOT/F1H_backward/per_fold_selected_subsets.json"
elif [[ -f "$TASK2_ROOT/per_fold_selected_subsets.json" ]]; then
  SUBSETS_JSON="$TASK2_ROOT/per_fold_selected_subsets.json"
  TASK2_ROOT="$(dirname "$TASK2_ROOT")"
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: DATA_ROOT not found: $DATA_ROOT" >&2
  exit 1
fi
if [[ ! -f "$SPLITS_PATH" || ! -f "$BODY_DIMS_PATH" ]]; then
  echo "ERROR: splits or body_dims missing ($SPLITS_PATH / $BODY_DIMS_PATH)" >&2
  exit 1
fi
if [[ -z "$SUBSETS_JSON" || ! -f "$SUBSETS_JSON" ]]; then
  echo "ERROR: Task 2 H3 subsets missing (F1H_backward/per_fold_selected_subsets.json under '$TASK2_ROOT'). Finish Task 2 or pass --task2-root PATH." >&2
  exit 1
fi

STATUS_FILE="$RESULTS_ROOT/stage_status_task34.json"

stage_hash() {
  local stage_name="$1"
  local spec="$2"
  shift 2
  "$PYTHON_BIN" - "$stage_name" "$spec" "$@" <<'PYEOF'
import hashlib, sys
from pathlib import Path
h = hashlib.sha256()
h.update(sys.argv[1].encode("utf-8") + b"\0" + sys.argv[2].encode("utf-8"))
for p in sys.argv[3:]:
    path = Path(p)
    if path.is_file():
        h.update(b"\0" + p.encode("utf-8") + b"\0" + path.read_bytes())
print(h.hexdigest())
PYEOF
}

stage_is_done() {
  local stage_name="$1"
  local cur_hash="$2"
  [[ "$OVERWRITE" -eq 1 ]] && return 1
  "$PYTHON_BIN" - "$STATUS_FILE" "$stage_name" "$cur_hash" <<'PYEOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    sys.exit(1)
rec = json.loads(p.read_text(encoding="utf-8")).get(sys.argv[2], {})
sys.exit(0 if rec.get("status") == "DONE" and rec.get("config_sha256") == sys.argv[3] else 1)
PYEOF
}

record_stage() {
  local stage_name="$1" status="$2" cfg_hash="$3" elapsed="$4" log_rel="$5" note="${6:-}"
  "$PYTHON_BIN" - "$STATUS_FILE" "$stage_name" "$status" "$cfg_hash" "$elapsed" "$log_rel" "$note" <<'PYEOF'
import datetime, json, sys
from pathlib import Path
from tempfile import NamedTemporaryFile
p = Path(sys.argv[1])
stage, status, cfg_hash, elapsed, log_rel, note = sys.argv[2:8]
data = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
prev = data.get(stage, {})
data[stage] = {
    "stage": stage, "status": status, "config_sha256": cfg_hash,
    "started_at_utc": now if status == "START" else prev.get("started_at_utc", now),
    "updated_at_utc": now, "elapsed_seconds": float(elapsed),
    "log_file": log_rel, "note": note,
}
with NamedTemporaryFile("w", encoding="utf-8", dir=p.parent, delete=False) as f:
    tmp = Path(f.name)
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write("\n")
tmp.replace(p)
PYEOF
}

run_stage() {
  local stage_name="$1" required="$2" cfg_hash="$3" fn_name="$4"
  local log_rel="logs/${stage_name}.log"
  local log_file="$RESULTS_ROOT/$log_rel"
  if stage_is_done "$stage_name" "$cfg_hash"; then
    echo "[RESUME_SKIP] $stage_name (${cfg_hash:0:12})"
    return 0
  fi
  local t0 t1 elapsed rc
  t0="$(date +%s)"
  record_stage "$stage_name" "START" "$cfg_hash" "0" "$log_rel" ""
  echo "[START] $stage_name (log: $log_file)"
  set +e
  "$fn_name" >"$log_file" 2>&1
  rc=$?
  set -e
  t1="$(date +%s)"
  elapsed=$((t1 - t0))
  if [[ "$rc" -eq 0 ]]; then
    record_stage "$stage_name" "DONE" "$cfg_hash" "$elapsed" "$log_rel" ""
    echo "[DONE]  $stage_name (${elapsed}s)"
    return 0
  fi
  tail -n 60 "$log_file" >"$RESULTS_ROOT/failures/${stage_name}.txt" || true
  if [[ "$required" == "optional" ]]; then
    record_stage "$stage_name" "FAIL_ISOLATED" "$cfg_hash" "$elapsed" "$log_rel" "exit=$rc"
    echo "[WARN]  $stage_name failed (exit=$rc), continuing." >&2
    return 0
  fi
  record_stage "$stage_name" "FAIL" "$cfg_hash" "$elapsed" "$log_rel" "exit=$rc"
  echo "[FAIL]  $stage_name failed (exit=$rc). See $RESULTS_ROOT/failures/${stage_name}.txt" >&2
  return "$rc"
}

do_t4_calibration() {
  "$PYTHON_BIN" scripts/spine/calibrate_noise_from_p0.py \
    --data-root "$DATA_ROOT" \
    --body-dims "$BODY_DIMS_PATH" \
    --out "$RESULTS_ROOT/spine_p0_calibration"
}

do_t3_wnoj_none() {
  # exactly one run mode: --resume (default) or --overwrite (mutually exclusive in the trainer)
  local ov_args=(--resume)
  [[ "$OVERWRITE" -eq 1 ]] && ov_args=(--overwrite)
  local base=(
    --data-root "$DATA_ROOT" --splits "$SPLITS_PATH" --body-dims "$BODY_DIMS_PATH"
    --out "$RESULTS_ROOT/task3_v4" --fold-subsets-json "$SUBSETS_JSON"
    --accel-unit none --workers "$WORKERS"
  )
  if [[ "$SMOKE" -eq 1 ]]; then
    "$PYTHON_BIN" scripts/task3/run_task3_wnoj.py "${base[@]}" \
      --schemes H1 H3 --h1-sensors RightShank LeftThigh --h3-k 3 4 \
      --folds 0 1 --max-trials-per-activity 1 --max-seconds-per-trial 10 \
      --no-calibrate --no-ablation "${ov_args[@]}"
  else
    "$PYTHON_BIN" scripts/task3/run_task3_wnoj.py "${base[@]}" \
      --schemes H1 H3 --h3-k 3 4 5 6 "${ov_args[@]}"
  fi
}

do_t3_wnoj_g() {
  # exactly one run mode: --resume (default) or --overwrite (mutually exclusive in the trainer)
  local ov_args=(--resume)
  [[ "$OVERWRITE" -eq 1 ]] && ov_args=(--overwrite)
  local base=(
    --data-root "$DATA_ROOT" --splits "$SPLITS_PATH" --body-dims "$BODY_DIMS_PATH"
    --out "$RESULTS_ROOT/task3_v4_accel_g" --fold-subsets-json "$SUBSETS_JSON"
    --accel-unit g --workers "$WORKERS"
  )
  if [[ "$SMOKE" -eq 1 ]]; then
    "$PYTHON_BIN" scripts/task3/run_task3_wnoj.py "${base[@]}" \
      --schemes H1 H3 --h1-sensors LeftThigh RightShank --h3-k 3 4 \
      --folds 0 1 --max-trials-per-activity 1 --max-seconds-per-trial 10 \
      --no-calibrate --no-ablation "${ov_args[@]}"
  else
    "$PYTHON_BIN" scripts/task3/run_task3_wnoj.py "${base[@]}" \
      --schemes H1 H3 --h3-k 3 4 5 6 "${ov_args[@]}"
  fi
}

do_t3_bigru() {
  if [[ "$SKIP_BIGRU" -eq 1 ]]; then
    echo "Skipping Task 3 BiGRU 6D (--skip-bigru set)."
    return 0
  fi
  "$PYTHON_BIN" -c "import torch; print('torch', torch.__version__, 'cuda_available=', torch.cuda.is_available())"
  # exactly one run mode: --resume (default) or --overwrite (mutually exclusive in the trainer)
  local ov_args=(--resume)
  [[ "$OVERWRITE" -eq 1 ]] && ov_args=(--overwrite)
  local base=(
    --data-root "$DATA_ROOT" --splits "$SPLITS_PATH" --body-dims "$BODY_DIMS_PATH"
    --out "$RESULTS_ROOT/task3_bigru_v4" --checkpoint-dir "$CHECKPOINT_DIR"
    --fold-subsets-json "$SUBSETS_JSON" --device "$BIGRU_DEVICE" --apply-wnoj-post
  )
  if [[ "$SMOKE" -eq 1 ]]; then
    "$PYTHON_BIN" scripts/task3_dl/train_bigru_6d.py "${base[@]}" \
      --fast-smoke --epochs 1 --folds 0 --schemes H1 H3 --h1-sensors RightShank \
      --subset-k 3 --max-trials-per-activity 1 --max-seconds-per-trial 10 --chunk-seconds 2 "${ov_args[@]}"
  else
    "$PYTHON_BIN" scripts/task3_dl/train_bigru_6d.py "${base[@]}" \
      --schemes H1 H3 --subset-k 3 4 5 6 \
      --epochs "$BIGRU_EPOCHS" --patience "$BIGRU_PATIENCE" \
      --samples-per-epoch "$BIGRU_SAMPLES" --train-frames "$BIGRU_FRAMES" "${ov_args[@]}"
  fi
}

do_fig_task12() {
  local args=(--task1-root "$TASK1_ROOT" --out "$RESULTS_ROOT/task12_paper_figures")
  if [[ -n "$TASK1_CLOUD_ROOT" && -d "$TASK1_CLOUD_ROOT" ]]; then
    if find "$TASK1_CLOUD_ROOT" -name "oof_predictions.parquet" -print -quit | grep -q .; then
      args+=(--cloud-results-root "$TASK1_CLOUD_ROOT")
    fi
  fi
  local t2_specs=()
  for name in F1H_ranking_prefix F1H_backward F3_backward F1H_forward; do
    if [[ -f "$TASK2_ROOT/$name/pareto_curve.csv" ]]; then
      t2_specs+=("$name=$TASK2_ROOT/$name")
    fi
  done
  if [[ "${#t2_specs[@]}" -gt 0 ]]; then
    args+=(--task2 "${t2_specs[@]}")
  fi
  "$PYTHON_BIN" scripts/analysis/build_task12_paper_figures.py "${args[@]}"
}

do_package() {
  "$PYTHON_BIN" scripts/package_cloud_v4_bundle.py \
    --results-root "$RESULTS_ROOT" --out-zip "$OUT_ZIP"
  ls -lh "$OUT_ZIP"
}

HCAL="$(stage_hash t4_cal "smoke=$SMOKE data=$DATA_ROOT" scripts/spine/calibrate_noise_from_p0.py)"
HT3N="$(stage_hash t3_none "smoke=$SMOKE data=$DATA_ROOT sub=$SUBSETS_JSON" "$SUBSETS_JSON" scripts/task3/run_task3_wnoj.py src/task3/smoother.py src/task3/withheld_eval.py)"
HT3G="$(stage_hash t3_g "smoke=$SMOKE data=$DATA_ROOT sub=$SUBSETS_JSON" "$SUBSETS_JSON" scripts/task3/run_task3_wnoj.py src/task3/accel_factor.py src/task3/smoother.py src/task3/withheld_eval.py)"
HTDL="$(stage_hash t3_dl "smoke=$SMOKE skip=$SKIP_BIGRU ep=$BIGRU_EPOCHS pat=$BIGRU_PATIENCE smp=$BIGRU_SAMPLES frm=$BIGRU_FRAMES dev=$BIGRU_DEVICE sub=$SUBSETS_JSON" "$SUBSETS_JSON" scripts/task3_dl/train_bigru_6d.py src/task3_dl/dataset.py src/task3_dl/model.py src/task3_dl/evaluation.py)"
HFIG="$(stage_hash fig_12 "smoke=$SMOKE t1=$TASK1_ROOT t2=$TASK2_ROOT c=$TASK1_CLOUD_ROOT" scripts/analysis/build_task12_paper_figures.py)"

run_stage "t4_p0_calibration"     "required" "$HCAL" do_t4_calibration
run_stage "t3_wnoj_accel_none"    "required" "$HT3N" do_t3_wnoj_none
run_stage "t3_wnoj_accel_g"       "required" "$HT3G" do_t3_wnoj_g
run_stage "t3_bigru_6d_wnoj_post" "required" "$HTDL" do_t3_bigru
run_stage "fig_task12_paper"      "required" "$HFIG" do_fig_task12

HPKG="$(stage_hash pkg_34 "c=$HCAL n=$HT3N g=$HT3G dl=$HTDL f=$HFIG zip=$OUT_ZIP" scripts/package_cloud_v4_bundle.py)"
run_stage "package_task34_bundle" "required" "$HPKG" do_package

echo "================================================================="
echo "TASK34_PIPELINE_COMPLETE"
echo "Outputs under: $RESULTS_ROOT/{spine_p0_calibration,task3_v4,task3_v4_accel_g,task3_bigru_v4,task12_paper_figures}"
echo "Bundle archive: $OUT_ZIP"
echo "================================================================="
