#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-$ROOT/.venv}"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
  if [[ -z "$CONDA_BIN" ]]; then
    echo "conda was not found; set CONDA_BIN to the conda executable" >&2
    exit 127
  fi
  "$CONDA_BIN" create -y -p "$ENV_PREFIX" python=3.10
fi

PYTHON_BIN="$ENV_PREFIX/bin/python"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.4.1 torchvision==0.19.1
"$PYTHON_BIN" -m pip install -r "$ROOT/requirements.txt"

"$PYTHON_BIN" - <<'PY'
import peft, torch, transformers
print("torch=", torch.__version__)
print("transformers=", transformers.__version__)
print("peft=", peft.__version__)
print("cuda_available=", torch.cuda.is_available())
print("cuda_devices=", torch.cuda.device_count())
PY
