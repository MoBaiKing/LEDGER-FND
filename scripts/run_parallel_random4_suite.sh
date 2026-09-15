#!/usr/bin/env bash
# Run four matched random seeds concurrently (one process/GPU), datasets sequentially.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/dyl/sotamodelv5/.venv/bin/python}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
EPOCHS="${EPOCHS:-50}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-8}"
DATASETS=("$@")
if (( ${#DATASETS[@]} == 0 )); then
  DATASETS=(gossipcop weibo)
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! "$EPOCHS" =~ ^[1-9][0-9]*$ || ! "$EARLY_STOP_PATIENCE" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPOCHS and EARLY_STOP_PATIENCE must be positive integers" >&2
  exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS_CSV"
if (( ${#GPU_ARRAY[@]} != 4 )); then
  echo "GPU_IDS must contain exactly four comma-separated GPU IDs" >&2
  exit 2
fi
declare -A SEEN_GPU=()
for gpu in "${GPU_ARRAY[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ || -n "${SEEN_GPU[$gpu]:-}" ]]; then
    echo "GPU_IDS must contain four distinct non-negative integers" >&2
    exit 2
  fi
  SEEN_GPU[$gpu]=1
done
for dataset in "${DATASETS[@]}"; do
  case "$dataset" in
    weibo21|gossipcop|weibo) ;;
    *) echo "This suite accepts only weibo21, gossipcop and weibo: $dataset" >&2; exit 2 ;;
  esac
  [[ -f "configs/datasets/$dataset.json" ]] || { echo "Missing config: $dataset" >&2; exit 2; }
  [[ -f "datasets/$dataset/ready/dataset_manifest.json" ]] || {
    echo "Dataset is not ready: $dataset" >&2; exit 2;
  }
done

# One OS-CSPRNG seed set is shared across datasets for a matched comparison.
# An explicit set allows an interrupted launch to resume the same experiment.
SEEDS_CSV="${SEEDS_CSV:-$($PYTHON_BIN - <<'PY'
import secrets
print(",".join(map(str, secrets.SystemRandom().sample(range(1, 2**31), 4))))
PY
)}"
IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS_CSV"
if (( ${#SEED_ARRAY[@]} != 4 )); then
  echo "SEEDS_CSV must contain exactly four comma-separated seeds" >&2
  exit 2
fi
declare -A SEEN_SEED=()
for seed in "${SEED_ARRAY[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ || "$seed" -ge 2147483648 || -n "${SEEN_SEED[$seed]:-}" ]]; then
    echo "SEEDS_CSV must contain four distinct integers in [0, 2^31)" >&2
    exit 2
  fi
  SEEN_SEED[$seed]=1
done

SUITE="${SUITE_GROUP:-lgled_parallel4_random_$(date +%Y%m%d_%H%M%S)}"
if [[ ! "$SUITE" =~ ^[A-Za-z0-9_.-]+$ || "$SUITE" == . || "$SUITE" == .. ]]; then
  echo "Invalid SUITE_GROUP: $SUITE" >&2
  exit 2
fi
SUITE_DIR="workspaces/multiseed_suites/$SUITE"
mkdir -p "$(dirname "$SUITE_DIR")"
mkdir "$SUITE_DIR"

"$PYTHON_BIN" - "$SUITE_DIR/seeds.json" "$SEEDS_CSV" "$GPU_IDS_CSV" "$EPOCHS" "$EARLY_STOP_PATIENCE" "${DATASETS[@]}" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path
output, seeds, gpus, epochs, patience, *datasets = sys.argv[1:]
payload = {
    "created_at": datetime.now().astimezone().isoformat(),
    "seed_source": "OS CSPRNG via secrets.SystemRandom",
    "seeds": [int(value) for value in seeds.split(",")],
    "matched_across_datasets": True,
    "datasets": datasets,
    "gpu_ids": [int(value) for value in gpus.split(",")],
    "epochs": int(epochs),
    "early_stop_patience": int(patience),
    "parallelism": "four independent single-GPU runs per dataset; datasets sequential",
    "standard_deviation": "sample (n-1)",
}
Path(output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY

export PYTHON_BIN PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
ACTIVE_PIDS=()
stop_children() {
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap stop_children INT TERM

echo "[$(date --iso-8601=seconds)] suite=$SUITE seeds=$SEEDS_CSV GPUs=$GPU_IDS_CSV"
for dataset in "${DATASETS[@]}"; do
  case "$dataset" in
    weibo21)
      batch_size="${WEIBO21_BATCH_SIZE:-16}"
      accumulation="${WEIBO21_GRAD_ACCUM_STEPS:-1}"
      ;;
    gossipcop)
      batch_size="${GOSSIPCOP_BATCH_SIZE:-32}"
      accumulation="${GOSSIPCOP_GRAD_ACCUM_STEPS:-1}"
      ;;
    weibo)
      batch_size="${WEIBO_BATCH_SIZE:-16}"
      accumulation="${WEIBO_GRAD_ACCUM_STEPS:-1}"
      ;;
  esac
  if [[ ! "$batch_size" =~ ^[1-9][0-9]*$ || ! "$accumulation" =~ ^[1-9][0-9]*$ ]]; then
    echo "Batch size and accumulation must be positive integers" >&2
    exit 2
  fi
  GROUP="${SUITE}_${dataset}"
  GROUP_DIR="workspaces/$dataset/runs/multiseed/$GROUP"
  mkdir -p "$(dirname "$GROUP_DIR")"
  mkdir "$GROUP_DIR"
  cp "$SUITE_DIR/seeds.json" "$GROUP_DIR/seeds.json"

  OUTPUT_ROOT="$($PYTHON_BIN - "configs/datasets/$dataset.json" <<'PY'
import json, sys
from pathlib import Path
root = Path.cwd()
value = Path(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["train"]["output_dir"])
print(value if value.is_absolute() else root / value)
PY
)"
  RUN_DIRS=()
  ACTIVE_PIDS=()
  RUN_NAMES=()
  echo "[$(date --iso-8601=seconds)] starting dataset=$dataset (4 seeds in parallel) batch=$batch_size accumulation=$accumulation effective_batch=$((batch_size * accumulation))"
  for index in 0 1 2 3; do
    seed="${SEED_ARRAY[$index]}"
    gpu="${GPU_ARRAY[$index]}"
    run_name="${GROUP}_seed${seed}"
    log_path="$GROUP_DIR/seed_${seed}_gpu${gpu}.log"
    RUN_DIRS+=("$OUTPUT_ROOT/$run_name")
    RUN_NAMES+=("$run_name")
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export NPROC_PER_NODE=1
      bash scripts/run_dataset.sh "$dataset" --seed "$seed" --run-name "$run_name" \
        --epochs "$EPOCHS" --early-stop-patience "$EARLY_STOP_PATIENCE" \
        --per-gpu-batch-size "$batch_size" --grad-accum-steps "$accumulation"
    ) >"$log_path" 2>&1 < /dev/null &
    pid=$!
    ACTIVE_PIDS+=("$pid")
    printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "$seed" "$gpu" "$pid" "$log_path" \
      | tee -a "$SUITE_DIR/launched.tsv"
  done

  failed=0
  for index in 0 1 2 3; do
    pid="${ACTIVE_PIDS[$index]}"
    if wait "$pid"; then
      status=0
    else
      status=$?
      failed=1
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$dataset" "${SEED_ARRAY[$index]}" \
      "${GPU_ARRAY[$index]}" "$status" "$(date --iso-8601=seconds)" \
      | tee -a "$SUITE_DIR/status.tsv"
  done
  ACTIVE_PIDS=()
  if (( failed != 0 )); then
    echo "At least one $dataset seed failed; inspect $GROUP_DIR" >&2
    exit 1
  fi
  "$PYTHON_BIN" aggregate_seeds.py --output "$GROUP_DIR/aggregate.json" "${RUN_DIRS[@]}"
  echo "[$(date --iso-8601=seconds)] completed dataset=$dataset aggregate=$GROUP_DIR/aggregate.json"
done

trap - INT TERM
echo "[$(date --iso-8601=seconds)] completed suite=$SUITE"
