#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec "${R1_PYTHON:-$ROOT/.venv/bin/python}" "$ROOT/scripts/masked_r1_cli.py" "$@"
