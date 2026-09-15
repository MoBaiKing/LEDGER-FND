#!/usr/bin/env bash
# Weibo21 then GossipCop: four matched random seeds, one single-GPU run per card.
# Each completed dataset is aggregated as mean +/- sample std (n-1).
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHON_BIN="${PYTHON_BIN:-/data/dyl/sotamodelv5/.venv/bin/python}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export EPOCHS=30
export EARLY_STOP_PATIENCE=8

# These preserve the original v3 effective batch while eliminating avoidable
# microbatch accumulation. Both settings have a GPU smoke-test entry point in
# train.py; Weibo21's flattened multi-image batches are the limiting case.
export WEIBO21_BATCH_SIZE="${WEIBO21_BATCH_SIZE:-16}"
export WEIBO21_GRAD_ACCUM_STEPS="${WEIBO21_GRAD_ACCUM_STEPS:-1}"
export GOSSIPCOP_BATCH_SIZE="${GOSSIPCOP_BATCH_SIZE:-32}"
export GOSSIPCOP_GRAD_ACCUM_STEPS="${GOSSIPCOP_GRAD_ACCUM_STEPS:-1}"

export SUITE_GROUP="${SUITE_GROUP:-lgled_weibo21_gossip_random4_max_$(date +%Y%m%d_%H%M%S)}"

exec bash "$ROOT/scripts/run_parallel_random4_suite.sh" weibo21 gossipcop
