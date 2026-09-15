#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -ne 8 ]]; then
  echo "usage: $0 WAIT_PID WAIT_GROUP QUEUE_DIR GOSSIP_GROUP GOSSIP_SEEDS WEIBO_GROUP WEIBO_SEEDS PYTHON_BIN" >&2
  exit 2
fi

WAIT_PID="$1"
WAIT_GROUP="$2"
QUEUE_DIR="$3"
GOSSIP_GROUP="$4"
GOSSIP_SEEDS="$5"
WEIBO_GROUP="$6"
WEIBO_SEEDS="$7"
PYTHON_BIN="$8"
STATUS_FILE="$QUEUE_DIR/status.tsv"

mkdir -p "$QUEUE_DIR"

waiting_for_target() {
  [[ -r "/proc/$WAIT_PID/cmdline" ]] || return 1
  local command_line
  command_line="$(tr '\0' ' ' < "/proc/$WAIT_PID/cmdline")"
  [[ "$command_line" == *"$WAIT_GROUP"* ]]
}

echo "[$(date '+%F %T')] queued; waiting for pid=$WAIT_PID group=$WAIT_GROUP"
while waiting_for_target; do
  sleep 30
done
echo "[$(date '+%F %T')] prerequisite ended; starting gossipcop"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHON_BIN
export NPROC_PER_NODE=4
export TOKENIZERS_PARALLELISM=false

bash run_5seeds.sh \
  --dataset gossipcop \
  --seeds "$GOSSIP_SEEDS" \
  --group "$GOSSIP_GROUP"
gossip_status=$?
printf 'gossipcop\t%s\t%s\n' "$gossip_status" "$(date --iso-8601=seconds)" >> "$STATUS_FILE"

echo "[$(date '+%F %T')] gossipcop exit=$gossip_status; starting weibo"
bash run_5seeds.sh \
  --dataset weibo \
  --seeds "$WEIBO_SEEDS" \
  --group "$WEIBO_GROUP"
weibo_status=$?
printf 'weibo\t%s\t%s\n' "$weibo_status" "$(date --iso-8601=seconds)" >> "$STATUS_FILE"

echo "[$(date '+%F %T')] queue finished; gossipcop=$gossip_status weibo=$weibo_status"
if (( gossip_status != 0 || weibo_status != 0 )); then
  exit 1
fi
