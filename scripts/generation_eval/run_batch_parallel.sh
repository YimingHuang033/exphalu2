#!/usr/bin/env bash
# Parallel GPU-pinned batch: run several (dataset, n) pipelines concurrently,
# one sequential queue per GPU (CUDA_VISIBLE_DEVICES pinning).
# Usage: run_batch_parallel.sh [config] [backend] [default_model] <GPU:DATASET:NUM[:MODEL]> ...
#   e.g. run_batch_parallel.sh config/generation_eval.yaml vllm qwen2_5_1_5b \
#          0:mmlu_pro:500 1:hle_text:500:qwen3_1_7b
set -uo pipefail
CATEGORY="generation_eval"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/log/$CATEGORY" "$ROOT/results/$CATEGORY"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$ROOT/log/$CATEGORY/run_batch_parallel-$TS.log"
exec > >(tee -a "$LOG") 2>&1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tim
cd "$ROOT"

CONFIG=${1:-config/generation_eval.yaml}
BACKEND=${2:-vllm}
DEFAULT_MODEL=${3:-qwen2_5_1_5b}
shift 3 2>/dev/null || shift $#

RESULTS_FILE="$ROOT/results/$CATEGORY/BATCHPAR-$TS-summary.csv"
echo "gpu,model,dataset,n,run_id,status" > "$RESULTS_FILE"

declare -A QUEUES
declare -A QUEUE_GPUS
for SPEC in "$@"; do
  IFS=':' read -r GPU DATASET NUM MODEL <<< "$SPEC"
  MODEL=${MODEL:-$DEFAULT_MODEL}
  QUEUES["$GPU"]+="$DATASET,$NUM,$MODEL;"
  QUEUE_GPUS["$GPU"]=1
done

for GPU in "${!QUEUE_GPUS[@]}"; do
  (
    export CUDA_VISIBLE_DEVICES=$GPU
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-}
    IFS=';' read -ra JOBS <<< "${QUEUES[$GPU]}"
    for JOB in "${JOBS[@]}"; do
      [ -z "$JOB" ] && continue
      IFS=',' read -r DATASET NUM MODEL <<< "$JOB"
      RID="geneval-$(date +%Y%m%d-%H%M%S)-g$GPU"
      echo "=== [gpu$GPU] START $MODEL x $DATASET (n=$NUM) run_id=$RID $(date +%H:%M:%S)"
      if RUN_ID_OVERRIDE="$RID" bash scripts/generation_eval/run_pipeline.sh \
           "$CONFIG" "$MODEL" "$BACKEND" "$NUM" "$DATASET"; then
        echo "gpu$GPU,$MODEL,$DATASET,$NUM,$RID,ok" >> "$RESULTS_FILE"
        echo "=== [gpu$GPU] OK $DATASET run_id=$RID"
      else
        echo "gpu$GPU,$MODEL,$DATASET,$NUM,$RID,FAILED" >> "$RESULTS_FILE"
        echo "=== [gpu$GPU] FAILED $DATASET run_id=$RID (see log above); continuing queue"
      fi
    done
    echo "=== [gpu$GPU] QUEUE DONE $(date +%H:%M:%S)"
  ) > "$ROOT/log/$CATEGORY/queue-gpu$GPU-$TS.log" 2>&1 &
  echo "launched queue gpu$GPU -> log/$CATEGORY/queue-gpu$GPU-$TS.log (pid $!)"
  sleep 3   # stagger launches so run_ids/timestamps do not collide
done

wait
echo
echo "== parallel batch summary: $RESULTS_FILE =="
column -s, -t < "$RESULTS_FILE"
if grep -q FAILED "$RESULTS_FILE"; then echo "BATCH FINISHED WITH FAILURES"; exit 1; fi
echo "BATCH OK"
