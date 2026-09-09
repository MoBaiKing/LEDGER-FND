#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-$ROOT/.venv}"
export ENV_PREFIX

if [[ -z "${DATASET_SOURCE_ROOT:-}" ]]; then
  echo "Set DATASET_SOURCE_ROOT to the directory containing the prepared datasets." >&2
  exit 2
fi

bash "$ROOT/scripts/setup_environment.sh"
"$ENV_PREFIX/bin/python" "$ROOT/scripts/download_pretrained_models.py"
"$ENV_PREFIX/bin/python" "$ROOT/scripts/prepare_v2_datasets.py" \
  --source-root "$DATASET_SOURCE_ROOT"

echo "CUTE-FND v3 environment, backbones and four v2 datasets are ready."
