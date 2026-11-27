#!/usr/bin/env bash
# Unified unattended V4.0 cloud runner (Stages 0-6) with flock, SHA256 hash
# invalidation, atomic state tracking, breakpoint resume, failure isolation,
# and single-archive output (cloud_v4_final_bundle.zip).
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Defaults (cloud RTX 4090 layout; overridden in --smoke mode or via CLI)
SMOKE=0
OVERWRITE=0
PYTHON_BIN="${PY:-/root/autodl-tmp/C_Problem/.venv/bin/python}"
DATA_ROOT="${DATA:-/root/autodl-tmp/C_Problem/data/processed/independent_v1}"
RESULTS_ROOT="results"
FEATURE_CACHE=""
SPLITS_PATH="results/preprocessing/independent_v1/subject_splits_6fold.json"
BODY_DIMS_PATH="results/preprocessing/independent_v1/body_dimensions_m.csv"
OOF_PATH="${OOF:-/root/autodl-tmp/C_Problem/results/task1_lightgbm_baseline_v1/oof_predictions.parquet}"
OUT_ZIP=""
RUN_ID=""
TASK3_WORKERS=4

usage() {
  cat <<'EOF'
Usage: bash scripts/run_cloud_all_v4.sh [OPTIONS]

Options:
  --smoke                 Run end-to-end synthetic smoke test (Stage 0 -> Stage 6)
  --python PATH           Python executable (default: cloud .venv or local .venv/python3)
  --data-root PATH        independent_v1 data root
  --results-root PATH     Output results root (default: results)
  --feature-cache PATH    V4 feature cache directory (kept outside git)
  --splits PATH           Subject splits JSON
  --body-dims PATH        Body dimensions CSV
  --oof PATH              Baseline OOF parquet for P0-3 (optional if absent)
  --out-zip PATH          Final bundle ZIP path (default: <results-root>/cloud_v4_final_bundle.zip)
  --task3-workers N       Worker processes for Task 3 (default: 4)
  --run-id ID             Explicit run identifier
  --overwrite             Ignore cached stage completion markers and recompute
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --feature-cache) FEATURE_CACHE="$2"; shift 2 ;;
    --splits) SPLITS_PATH="$2"; shift 2 ;;
    --body-dims) BODY_DIMS_PATH="$2"; shift 2 ;;
    --oof) OOF_PATH="$2"; shift 2 ;;
    --out-zip) OUT_ZIP="$2"; shift 2 ;;
    --task3-workers) TASK3_WORKERS="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Resolve Python binary cleanly
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

# Configure smoke defaults if --smoke requested
if [[ "$SMOKE" -eq 1 ]]; then
  if [[ "$RESULTS_ROOT" == "results" ]]; then
    RESULTS_ROOT="/tmp/v4_smoke_results"
  fi
  if [[ "$DATA_ROOT" == "/root/autodl-tmp/C_Problem/data/processed/independent_v1" ]]; then
    DATA_ROOT="$RESULTS_ROOT/_smoke_data"
  fi
  if [[ -z "$FEATURE_CACHE" ]]; then
    FEATURE_CACHE="$RESULTS_ROOT/_smoke_feature_cache"
  fi
  SPLITS_PATH="$DATA_ROOT/splits_smoke.json"
  BODY_DIMS_PATH="$DATA_ROOT/body_dims_smoke.csv"
  OOF_PATH=""
  TASK3_WORKERS=1
else
  if [[ -z "$FEATURE_CACHE" ]]; then
    FEATURE_CACHE="/root/autodl-tmp/C_Problem/data/processed/v4_features_cache"
  fi
fi

if [[ -z "$OUT_ZIP" ]]; then
  OUT_ZIP="$RESULTS_ROOT/cloud_v4_final_bundle.zip"
fi

mkdir -p "$RESULTS_ROOT" "$RESULTS_ROOT/logs" "$RESULTS_ROOT/failures" "$RESULTS_ROOT/figures" "$RESULTS_ROOT/final_report"

# Prevent duplicate concurrent runs via flock
LOCK_FILE="$RESULTS_ROOT/.v4_run.lock"
exec 9>"$LOCK_FILE"
if command -v flock >/dev/null 2>&1; then
  if ! flock -n 9; then
    echo "ERROR: Another run_cloud_all_v4.sh process holds $LOCK_FILE" >&2
    exit 1
  fi
fi

# Build synthetic fixture once in --smoke mode (reused on resume check)
if [[ "$SMOKE" -eq 1 && ! -f "$DATA_ROOT/summary.json" ]]; then
  echo "[SMOKE] Building synthetic fixture under $DATA_ROOT ..."
  "$PYTHON_BIN" - "$DATA_ROOT" "$SPLITS_PATH" "$BODY_DIMS_PATH" <<'PYEOF'
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from scripts.diagnostics import make_synthetic_fixture as fix

root = Path(sys.argv[1])
splits_path = Path(sys.argv[2])
dims_path = Path(sys.argv[3])
fix.build(root, unit='mps2', subjects=(1, 2, 3, 4))

# Ensure both 2s and 4s window parquet tables exist for WP1 extraction
for sid in (1, 2, 3, 4):
    sdir = root / 'subjects' / f'P{sid:02d}'
    trials = pd.read_parquet(sdir / 'trials.parquet')
    labels = np.load(sdir / 'labels.npy')
    for win_sec, stride_frames in ((2, 60), (4, 120)):
        win_frames = win_sec * 60
        rows = []
        wid = 0
        for row in trials.itertuples():
            start0, stop0 = int(row.source_start), int(row.source_stop_exclusive)
            for s in range(start0, stop0 - win_frames + 1, stride_frames):
                seg = labels[s:s + win_frames]
                vals, cnts = np.unique(seg, return_counts=True)
                maj = int(vals[np.argmax(cnts)])
                purity = float(cnts.max() / len(seg))
                rows.append({
                    'subject': sid, 'trial': int(row.trial), 'window_id': wid,
                    'source_start': s, 'source_stop_exclusive': s + win_frames,
                    'stride_frames': stride_frames, 'majority_label': maj,
                    'purity': purity, 'mixed_labels': bool(len(vals) > 1),
                })
                wid += 1
        pd.DataFrame(rows).to_parquet(sdir / f'windows_{win_sec}s.parquet', index=False)

splits = {
    'folds': [
        {'fold': 0, 'train': [1, 2], 'validation': [3], 'test': [4]},
        {'fold': 1, 'train': [3, 4], 'validation': [1], 'test': [2]},
    ]
}
splits_path.write_text(json.dumps(splits, indent=2) + '\n', encoding='utf-8')
cols = {
    'body_height': 1.75, 'hip_width': 0.30, 'shoulder_width': 0.40,
    'thigh_length': 0.43, 'shank_length': 0.42, 'foot_length': 0.26,
    'upper_arm_length': 0.30, 'forearm_length': 0.26,
    'trunk_length': 0.50, 'neck_length': 0.12, 'head_height': 0.22,
}
lines = ['subject,' + ','.join(cols)] + [
    f'{s},' + ','.join(str(v) for v in cols.values()) for s in (1, 2, 3, 4)
]
dims_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
(root / 'summary.json').write_text(json.dumps({'smoke': True, 'subjects': 4}) + '\n', encoding='utf-8')
PYEOF
fi

# Verify inputs exist before starting
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: DATA_ROOT not found: $DATA_ROOT" >&2
  exit 1
fi
if [[ ! -f "$SPLITS_PATH" ]]; then
  echo "ERROR: SPLITS_PATH not found: $SPLITS_PATH" >&2
  exit 1
fi

# Initialize or resume run_manifest.json and capture hardware_environment.json
GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
"$PYTHON_BIN" - "$RESULTS_ROOT" "$GIT_SHA" "$SMOKE" "$DATA_ROOT" "$SPLITS_PATH" "$BODY_DIMS_PATH" "$RUN_ID" <<'PYEOF'
import datetime, hashlib, json, os, platform, shutil, subprocess, sys
from pathlib import Path
from tempfile import NamedTemporaryFile

results_root = Path(sys.argv[1])
git_sha = sys.argv[2]
smoke = bool(int(sys.argv[3]))
data_root = sys.argv[4]
splits_path = Path(sys.argv[5])
body_dims_path = Path(sys.argv[6])
req_run_id = sys.argv[7]

def atomic_json(path, obj):
    with NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write('\n')
    tmp.replace(path)

def sha256_file(p):
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

manifest_path = results_root / 'run_manifest.json'
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
if manifest_path.is_file() and not req_run_id:
    old = json.loads(manifest_path.read_text(encoding='utf-8'))
    run_id = old.get('run_id') or f"v4_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
else:
    run_id = req_run_id or f"v4_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

manifest = {
    'run_id': run_id,
    'mode': 'smoke' if smoke else 'cloud_full',
    'git_commit': git_sha,
    'updated_at_utc': now,
    'data_root': str(Path(data_root).resolve()),
    'splits_path': str(splits_path.resolve()),
    'splits_sha256': sha256_file(splits_path),
    'body_dims_sha256': sha256_file(body_dims_path),
    'runner_sha256': sha256_file(Path('scripts/run_cloud_all_v4.sh')),
    'task3_accel_unit_enforced': 'none',
}
atomic_json(manifest_path, manifest)

# Hardware & software snapshot
disk = shutil.disk_usage(results_root)
gpu_info = None
try:
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version', '--format=csv,noheader'],
        stderr=subprocess.DEVNULL, text=True, timeout=5
    ).strip()
    gpu_info = out.splitlines() if out else None
except Exception:
    gpu_info = None

mem_gb = None
try:
    with open('/proc/meminfo', encoding='utf-8') as f:
        for line in f:
            if line.startswith('MemTotal:'):
                mem_gb = round(int(line.split()[1]) / (1024 * 1024), 2)
                break
except Exception:
    pass

pkgs = {}
for mod in ('numpy', 'scipy', 'pandas', 'pyarrow', 'sklearn', 'lightgbm', 'matplotlib', 'torch'):
    try:
        m = __import__(mod)
        pkgs[mod] = getattr(m, '__version__', 'installed')
    except Exception:
        pkgs[mod] = None

hw = {
    'captured_at_utc': now,
    'hostname': platform.node(),
    'platform': platform.platform(),
    'python_version': platform.python_version(),
    'cpu_count': os.cpu_count(),
    'ram_total_gb': mem_gb,
    'disk_free_gb': round(disk.free / (1024 ** 3), 2),
    'gpu': gpu_info,
    'packages': pkgs,
}
atomic_json(results_root / 'hardware_environment.json', hw)
print(f"RUN_START run_id={run_id} mode={manifest['mode']} git={git_sha[:12]} cpu={hw['cpu_count']} disk_free_gb={hw['disk_free_gb']}")
PYEOF

# Helper: compute deterministic stage config hash (stage spec + key script hashes)
stage_hash() {
  local stage_name="$1"
  local spec="$2"
  shift 2
  "$PYTHON_BIN" - "$stage_name" "$spec" "$@" <<'PYEOF'
import hashlib, sys
from pathlib import Path
h = hashlib.sha256()
h.update(sys.argv[1].encode('utf-8'))
h.update(b'\0')
h.update(sys.argv[2].encode('utf-8'))
for p in sys.argv[3:]:
    path = Path(p)
    if path.is_file():
        h.update(b'\0' + p.encode('utf-8') + b'\0' + path.read_bytes())
print(h.hexdigest())
PYEOF
}

# Helper: check whether stage is DONE with matching hash
stage_is_done() {
  local stage_name="$1"
  local cur_hash="$2"
  if [[ "$OVERWRITE" -eq 1 ]]; then
    return 1
  fi
  "$PYTHON_BIN" - "$RESULTS_ROOT/stage_status.json" "$stage_name" "$cur_hash" <<'PYEOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    sys.exit(1)
data = json.loads(p.read_text(encoding='utf-8'))
rec = data.get(sys.argv[2], {})
if rec.get('status') == 'DONE' and rec.get('config_sha256') == sys.argv[3]:
    sys.exit(0)
sys.exit(1)
PYEOF
}

# Helper: atomically record stage status and regenerate timing_by_stage.csv
record_stage() {
  local stage_name="$1"
  local status="$2"
  local cfg_hash="$3"
  local elapsed="$4"
  local log_rel="$5"
  local note="${6:-}"
  "$PYTHON_BIN" - "$RESULTS_ROOT" "$stage_name" "$status" "$cfg_hash" "$elapsed" "$log_rel" "$note" <<'PYEOF'
import csv, datetime, json, sys
from pathlib import Path
from tempfile import NamedTemporaryFile

root = Path(sys.argv[1])
stage, status, cfg_hash, elapsed, log_rel, note = sys.argv[2:8]
status_path = root / 'stage_status.json'
data = json.loads(status_path.read_text(encoding='utf-8')) if status_path.is_file() else {}
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
prev = data.get(stage, {})
rec = {
    'stage': stage,
    'status': status,
    'config_sha256': cfg_hash,
    'started_at_utc': now if status == 'START' else prev.get('started_at_utc', now),
    'updated_at_utc': now,
    'elapsed_seconds': float(elapsed),
    'log_file': log_rel,
    'note': note,
}
data[stage] = rec
with NamedTemporaryFile('w', encoding='utf-8', dir=root, delete=False) as f:
    tmp = Path(f.name)
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write('\n')
tmp.replace(status_path)

timing_path = root / 'timing_by_stage.csv'
with NamedTemporaryFile('w', encoding='utf-8', dir=root, delete=False, newline='') as f:
    tmp = Path(f.name)
    w = csv.DictWriter(f, fieldnames=['stage', 'status', 'elapsed_seconds', 'config_sha256', 'log_file', 'note'])
    w.writeheader()
    for k, v in data.items():
        w.writerow({col: v.get(col, '') for col in w.fieldnames})
tmp.replace(timing_path)
PYEOF
}

# Run a required or optional stage with logging, hash invalidation, and timing
run_stage() {
  local stage_name="$1"
  local required="$2"
  local cfg_hash="$3"
  local fn_name="$4"
  local log_rel="logs/${stage_name}.log"
  local log_file="$RESULTS_ROOT/$log_rel"

  if stage_is_done "$stage_name" "$cfg_hash"; then
    echo "[RESUME_SKIP] $stage_name (DONE with matching config_sha256=${cfg_hash:0:12})"
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
  else
    local fail_file="$RESULTS_ROOT/failures/${stage_name}.txt"
    {
      echo "stage=$stage_name"
      echo "exit_code=$rc"
      echo "elapsed_seconds=$elapsed"
      echo "log_file=$log_rel"
      echo "--- tail of log ---"
      tail -n 60 "$log_file" || true
    } >"$fail_file"
    if [[ "$required" == "optional" ]]; then
      record_stage "$stage_name" "FAIL_ISOLATED" "$cfg_hash" "$elapsed" "$log_rel" "exit=$rc (optional stage isolated)"
      echo "[WARN]  $stage_name failed (exit=$rc), isolated; continuing." >&2
      return 0
    else
      record_stage "$stage_name" "FAIL" "$cfg_hash" "$elapsed" "$log_rel" "exit=$rc"
      echo "[FAIL]  $stage_name failed (exit=$rc). See $fail_file" >&2
      return "$rc"
    fi
  fi
}

# =============================================================================
# Stage definitions
# =============================================================================

do_stage0_p0() {
  local out="$RESULTS_ROOT/p0_diagnostics"
  mkdir -p "$out"
  local subj_args=()
  if [[ "$SMOKE" -eq 1 ]]; then
    subj_args=(--subjects 1 2 3 4)
  fi
  "$PYTHON_BIN" scripts/diagnostics/diag_accel_unit.py \
    --data-root "$DATA_ROOT" --out "$out" "${subj_args[@]}"
  "$PYTHON_BIN" scripts/diagnostics/diag_orientation_independence.py \
    --data-root "$DATA_ROOT" --out "$out" "${subj_args[@]}"
  local proto_args=(
    --data-root "$DATA_ROOT" --out "$out"
    --splits "$SPLITS_PATH" --body-dims "$BODY_DIMS_PATH"
  )
  if [[ -n "$OOF_PATH" && -f "$OOF_PATH" ]]; then
    proto_args+=(--oof "$OOF_PATH")
  fi
  "$PYTHON_BIN" scripts/diagnostics/diag_subject_protocol.py \
    "${proto_args[@]}" "${subj_args[@]}"
}

do_stage1_features() {
  local ov_args=()
  [[ "$OVERWRITE" -eq 1 ]] && ov_args=(--overwrite)
  local subj_args=()
  if [[ "$SMOKE" -eq 1 ]]; then
    subj_args=(--subjects 1 2 3 4)
    "$PYTHON_BIN" scripts/task1_v4/extract_v4_features.py \
      --data-root "$DATA_ROOT" --out-cache "$FEATURE_CACHE" \
      --window-sec 2 --families F1H F3 "${subj_args[@]}" "${ov_args[@]}"
    "$PYTHON_BIN" scripts/task1_v4/extract_v4_features.py \
      --data-root "$DATA_ROOT" --out-cache "$FEATURE_CACHE" \
      --window-sec 4 --families F3 "${subj_args[@]}" "${ov_args[@]}"
  else
    "$PYTHON_BIN" scripts/task1_v4/extract_v4_features.py \
      --data-root "$DATA_ROOT" --out-cache "$FEATURE_CACHE" \
      --window-sec 2 --families F1H F1W F1B F2 F2p F3 F4 "${ov_args[@]}"
    "$PYTHON_BIN" scripts/task1_v4/extract_v4_features.py \
      --data-root "$DATA_ROOT" --out-cache "$FEATURE_CACHE" \
      --window-sec 4 --families F1H F3 F4 "${ov_args[@]}"
  fi
}

do_stage2_task1() {
  local ov_args=()
  [[ "$OVERWRITE" -eq 1 ]] && ov_args=(--overwrite)
  local base=(
    --data-root "$DATA_ROOT"
    --feature-cache "$FEATURE_CACHE"
    --splits "$SPLITS_PATH"
    --enable-sticky-hmm
  )
  if [[ "$SMOKE" -eq 1 ]]; then
    for spec in "F1H:2" "F3:2" "F3:4"; do
      local fam="${spec%%:*}"
      local ws="${spec##*:}"
      "$PYTHON_BIN" scripts/task1_v4/train_task1_v4.py "${base[@]}" \
        --family "$fam" --window-sec "$ws" --fast-smoke \
        --out "$RESULTS_ROOT/task1_v4/${fam}_${ws}s" "${ov_args[@]}"
    done
  else
    for spec in "F1H:2" "F1W:2" "F1B:2" "F2p:2" "F3:2" "F3:4" "F4:2" "F4:4"; do
      local fam="${spec%%:*}"
      local ws="${spec##*:}"
      "$PYTHON_BIN" scripts/task1_v4/train_task1_v4.py "${base[@]}" \
        --family "$fam" --window-sec "$ws" \
        --out "$RESULTS_ROOT/task1_v4/${fam}_${ws}s" "${ov_args[@]}"
    done
  fi
}

do_stage3_task2() {
  local ov_args=()
  [[ "$OVERWRITE" -eq 1 ]] && ov_args=(--overwrite)
  local base=(
    --data-root "$DATA_ROOT"
    --feature-cache "$FEATURE_CACHE"
    --splits "$SPLITS_PATH"
    --window-sec 2
  )
  if [[ "$SMOKE" -eq 1 ]]; then
    local smoke_sensors=(LowerBack UpperBack Head LeftThigh RightShank RightFoot)
    "$PYTHON_BIN" scripts/task2_v4/run_sensor_selection.py "${base[@]}" \
      --family F1H --strategy ranking_prefix --fast-smoke --sensors "${smoke_sensors[@]}" \
      --out "$RESULTS_ROOT/task2_v4/F1H_ranking_prefix" "${ov_args[@]}"
    "$PYTHON_BIN" scripts/task2_v4/run_sensor_selection.py "${base[@]}" \
      --family F1H --strategy backward_greedy --fast-smoke --sensors "${smoke_sensors[@]}" \
      --out "$RESULTS_ROOT/task2_v4/F1H_backward" "${ov_args[@]}"
  else
    # Run fast ranking_prefix comparison first, then exact wrapper backward_greedy primary,
    # then F3 backward comparison and F1H forward sensitivity.
    "$PYTHON_BIN" scripts/task2_v4/run_sensor_selection.py "${base[@]}" \
      --family F1H --strategy ranking_prefix \
      --out "$RESULTS_ROOT/task2_v4/F1H_ranking_prefix" "${ov_args[@]}"
    "$PYTHON_BIN" scripts/task2_v4/run_sensor_selection.py "${base[@]}" \
      --family F1H --strategy backward_greedy \
      --out "$RESULTS_ROOT/task2_v4/F1H_backward" "${ov_args[@]}"
    "$PYTHON_BIN" scripts/task2_v4/run_sensor_selection.py "${base[@]}" \
      --family F3 --strategy backward_greedy \
      --out "$RESULTS_ROOT/task2_v4/F3_backward" "${ov_args[@]}"
    "$PYTHON_BIN" scripts/task2_v4/run_sensor_selection.py "${base[@]}" \
      --family F1H --strategy forward_greedy \
      --out "$RESULTS_ROOT/task2_v4/F1H_forward" "${ov_args[@]}"
  fi
}

do_stage4_task3() {
  local ov_args=()
  [[ "$OVERWRITE" -eq 1 ]] && ov_args=(--overwrite)
  local subsets_json="$RESULTS_ROOT/task2_v4/F1H_backward/per_fold_selected_subsets.json"
  if [[ ! -f "$subsets_json" ]]; then
    echo "ERROR: Expected Task 2 per-fold subsets JSON missing: $subsets_json" >&2
    return 1
  fi
  local base=(
    --data-root "$DATA_ROOT"
    --splits "$SPLITS_PATH"
    --body-dims "$BODY_DIMS_PATH"
    --out "$RESULTS_ROOT/task3_v4"
    --fold-subsets-json "$subsets_json"
    --accel-unit none
    --workers "$TASK3_WORKERS"
    --resume
  )
  if [[ "$SMOKE" -eq 1 ]]; then
    "$PYTHON_BIN" scripts/task3/run_task3_wnoj.py "${base[@]}" \
      --schemes H1 H3 --h1-sensors RightShank --h3-k 3 4 \
      --folds 0 1 --max-trials-per-activity 1 --max-seconds-per-trial 10 \
      --no-calibrate --no-ablation "${ov_args[@]}"
  else
    "$PYTHON_BIN" scripts/task3/run_task3_wnoj.py "${base[@]}" \
      --schemes H1 H3 --h3-k 3 4 5 6 "${ov_args[@]}"
  fi
}

do_stage5_optional_gpu() {
  if [[ "$SMOKE" -eq 1 ]]; then
    echo "Optional GPU Stage 5 skipped in --smoke mode."
    return 0
  fi
  if [[ ! -f "scripts/task3/train_bigru_6d.py" ]]; then
    echo "Optional GPU script scripts/task3/train_bigru_6d.py not present; skipping Stage 5."
    return 0
  fi
  "$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
  "$PYTHON_BIN" scripts/task3/train_bigru_6d.py \
    --data-root "$DATA_ROOT" --splits "$SPLITS_PATH" \
    --out "$RESULTS_ROOT/task5_bigru"
}

do_stage6_package() {
  "$PYTHON_BIN" scripts/package_cloud_v4_bundle.py \
    --results-root "$RESULTS_ROOT" \
    --out-zip "$OUT_ZIP"
  ls -lh "$OUT_ZIP"
}

# =============================================================================
# Execute pipeline
# =============================================================================
H0="$(stage_hash stage0_p0 "smoke=$SMOKE data=$DATA_ROOT splits=$SPLITS_PATH" scripts/diagnostics/diag_accel_unit.py scripts/diagnostics/diag_orientation_independence.py scripts/diagnostics/diag_subject_protocol.py)"
H1="$(stage_hash stage1_features "smoke=$SMOKE data=$DATA_ROOT cache=$FEATURE_CACHE" scripts/task1_v4/extract_v4_features.py src/v4_features/feature_families.py src/v4_features/workflow.py)"
H2="$(stage_hash stage2_task1 "smoke=$SMOKE data=$DATA_ROOT cache=$FEATURE_CACHE splits=$SPLITS_PATH" scripts/task1_v4/train_task1_v4.py src/v4_features/sticky_hmm.py src/v4_features/workflow.py)"
H3="$(stage_hash stage3_task2 "smoke=$SMOKE data=$DATA_ROOT cache=$FEATURE_CACHE splits=$SPLITS_PATH" scripts/task2_v4/run_sensor_selection.py src/v4_features/workflow.py)"
H4="$(stage_hash stage4_task3 "smoke=$SMOKE data=$DATA_ROOT splits=$SPLITS_PATH h3=$H3 accel=none" scripts/task3/run_task3_wnoj.py src/task3/wnoj_prior.py src/task3/smoother.py src/task3/withheld_eval.py)"
H5="$(stage_hash stage5_optional_gpu "smoke=$SMOKE data=$DATA_ROOT splits=$SPLITS_PATH")"

run_stage "stage0_p0_diagnostics" "required" "$H0" do_stage0_p0
run_stage "stage1_v4_features"    "required" "$H1" do_stage1_features
run_stage "stage2_task1_v4"       "required" "$H2" do_stage2_task1
run_stage "stage3_task2_v4"       "required" "$H3" do_stage3_task2
run_stage "stage4_task3_wnoj"     "required" "$H4" do_stage4_task3
run_stage "stage5_optional_gpu"   "optional" "$H5" do_stage5_optional_gpu

# Always run Stage 6 packaging at the end so final bundle reflects latest status
H6="$(stage_hash stage6_package "h0=$H0 h1=$H1 h2=$H2 h3=$H3 h4=$H4 h5=$H5 zip=$OUT_ZIP" scripts/package_cloud_v4_bundle.py)"
run_stage "stage6_package_bundle" "required" "$H6" do_stage6_package

echo "================================================================="
echo "ALL_STAGES_COMPLETE"
echo "Final bundle: $OUT_ZIP"
echo "Stage status: $RESULTS_ROOT/stage_status.json"
echo "================================================================="
