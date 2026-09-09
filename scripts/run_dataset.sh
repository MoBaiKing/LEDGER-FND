#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATASET="${1:-}"
if [[ -z "$DATASET" ]]; then
  echo "usage: bash scripts/run_dataset.sh {gossipcop|weibo21|twitter|weibo} [extra train.py args]" >&2
  exit 2
fi
shift
CONFIG="configs/datasets/${DATASET}.json"
MANIFEST_DIR="datasets/${DATASET}/ready"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python is not executable: $PYTHON_BIN; run scripts/setup_environment.sh" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST_DIR/dataset_manifest.json" ]]; then
  echo "Dataset is not ready: $MANIFEST_DIR; run scripts/prepare_v2_datasets.py" >&2
  exit 2
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node="$NPROC_PER_NODE" \
  train.py --dataset "$DATASET" --config "$CONFIG" \
  --manifest-dir "$MANIFEST_DIR" "$@"
