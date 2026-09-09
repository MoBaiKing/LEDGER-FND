#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

DATASET=""
SEEDS="42,3407,2024,2025,200408"
EPOCHS=""
PATIENCE=""
GROUP=""
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --early-stop-patience) PATIENCE="$2"; shift 2 ;;
    --group) GROUP="$2"; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
    -h|--help)
      echo "usage: bash run_5seeds.sh --dataset NAME [--epochs N] [--early-stop-patience N]"
      exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$DATASET" ]]; then
  echo "--dataset is required" >&2
  exit 2
fi
CONFIG="configs/datasets/${DATASET}.json"
MANIFEST_DIR="datasets/${DATASET}/ready"
IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
if [[ ${#SEED_ARRAY[@]} -ne 5 ]]; then
  echo "exactly five comma-separated seeds are required" >&2
  exit 2
fi
GROUP="${GROUP:-${DATASET}_lgled_5seeds_$(date +%Y%m%d_%H%M%S)}"
GROUP_DIR="workspaces/${DATASET}/runs/multiseed/${GROUP}"
mkdir -p "$GROUP_DIR"

export PYTHON_BIN NPROC_PER_NODE
RUN_DIRS=()
for seed in "${SEED_ARRAY[@]}"; do
  RUN_NAME="${GROUP}_seed${seed}"
  EXTRA=(--seed "$seed" --run-name "$RUN_NAME")
  if [[ -n "$EPOCHS" ]]; then EXTRA+=(--epochs "$EPOCHS"); fi
  if [[ -n "$PATIENCE" ]]; then EXTRA+=(--early-stop-patience "$PATIENCE"); fi
  echo "[$(date '+%F %T')] dataset=$DATASET seed=$seed"
  bash scripts/run_dataset.sh "$DATASET" "${EXTRA[@]}" \
    2>&1 | tee "$GROUP_DIR/seed_${seed}.log"
  OUTPUT_ROOT="$($PYTHON_BIN - "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
root = Path.cwd()
cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = Path(cfg["train"]["output_dir"])
print(value if value.is_absolute() else root / value)
PY
)"
  RUN_DIRS+=("$OUTPUT_ROOT/$RUN_NAME")
done

"$PYTHON_BIN" aggregate_seeds.py \
  --output "$GROUP_DIR/aggregate.json" "${RUN_DIRS[@]}"
echo "completed_group=$GROUP_DIR"
