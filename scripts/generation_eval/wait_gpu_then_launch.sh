#!/usr/bin/env bash
# Wait for a GPU to become free, then launch a generation_eval pipeline on it.
# Used to queue a follow-up run behind a still-running pipeline (e.g. launch
# popqa non-thinking on GPU1 only after the previous popqa thinking run exits
# and the engine releases memory). Detached and SSH-disconnect-proof: run via
# setsid nohup.
#
# Usage:
#   wait_gpu_then_launch.sh [gpu_index] [wait_pattern] [max_wait_s] \
#        [config] [model] [backend] [n_samples] [dataset] [run_id]
#
#   wait_pattern   : pgrep -f pattern that must DISAPPEAR before launching
#                    (empty string skips the process wait)
#   max_wait_s     : give up after this many seconds (exit 1, nothing launched)
#   run_id         : optional RUN_ID_OVERRIDE; default geneval-<ts>-gpu<index>
set -uo pipefail
CATEGORY="generation_eval"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/log/$CATEGORY"
TS=$(date +%Y%m%d-%H%M%S)
GPU=${1:-0}
WAIT_PATTERN=${2:-}
MAX_WAIT=${3:-21600}
CONFIG=${4:-config/generation_eval.yaml}
MODEL=${5:-qwen3_5_4b}
BACKEND=${6:-vllm}
NSAMPLES=${7:-16}
DATASET=${8:-}
RUN_ID=${9:-"geneval-$TS-gpu$GPU"}
LOG="$ROOT/log/$CATEGORY/wait_gpu_then_launch-$RUN_ID.log"
exec >> "$LOG" 2>&1
echo "[$(date '+%F %T')] waiting: gpu=$GPU pattern='${WAIT_PATTERN}' max_wait=${MAX_WAIT}s -> $DATASET x$NSAMPLES ($CONFIG) run_id=$RUN_ID"

start=$(date +%s)
if [ -n "$WAIT_PATTERN" ]; then
  while pgrep -f "$WAIT_PATTERN" > /dev/null; do
    if [ $(( $(date +%s) - start )) -gt "$MAX_WAIT" ]; then
      echo "[$(date '+%F %T')] TIMEOUT waiting for pattern; nothing launched (exit 1)"
      exit 1
    fi
    sleep 60
  done
  echo "[$(date '+%F %T')] pattern gone after $(( $(date +%s) - start ))s"
fi

# engine core exit lags behind process exit (vLLM 0.30 constraint): wait until
# the GPU actually releases memory, then a fixed settle delay
mem_free=0
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
  if [ "${used:-99999}" -lt 2000 ]; then mem_free=1; break; fi
  sleep 30
done
echo "[$(date '+%F %T')] gpu${GPU} mem_used=${used:-unknown}MiB mem_free=$mem_free; settle 120s"
sleep 120

COMMON=(bash "$ROOT/scripts/generation_eval/run_pipeline.sh" "$CONFIG" "$MODEL" "$BACKEND" "$NSAMPLES")
[ -n "$DATASET" ] && COMMON+=("$DATASET")
echo "[$(date '+%F %T')] launching: CUDA_VISIBLE_DEVICES=$GPU RUN_ID_OVERRIDE=$RUN_ID ${COMMON[*]}"
CUDA_VISIBLE_DEVICES="$GPU" RUN_ID_OVERRIDE="$RUN_ID" \
  setsid nohup "${COMMON[@]}" > "$ROOT/log/$CATEGORY/queue-gpu$GPU-$RUN_ID.log" 2>&1 < /dev/null &
echo "[$(date '+%F %T')] launched pid=$!; log: log/$CATEGORY/queue-gpu$GPU-$RUN_ID.log"
