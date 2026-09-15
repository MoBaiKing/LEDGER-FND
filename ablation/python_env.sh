#!/usr/bin/env bash
set -euo pipefail

ABLATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_PYTHON="${ABLATION_BASE_PYTHON:-/data/dyl/conda-envs/dyl_reproduce/bin/python}"

export PYTHONPATH="${ABLATION_ROOT}/.runtime/python-packages${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPYCACHEPREFIX="${ABLATION_ROOT}/.cache/pycache"
export PIP_CACHE_DIR="${ABLATION_ROOT}/.cache/pip"
export TMPDIR="${ABLATION_ROOT}/.cache/tmp"
export HF_HOME="${ABLATION_ROOT}/.cache/huggingface"
export TORCH_HOME="${ABLATION_ROOT}/.cache/torch"
export XDG_CACHE_HOME="${ABLATION_ROOT}/.cache/xdg"
export PYTHONNOUSERSITE=1

mkdir -p "${TMPDIR}" "${PYTHONPYCACHEPREFIX}" "${PIP_CACHE_DIR}" \
  "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"
exec "${ABLATION_PYTHON}" "$@"
