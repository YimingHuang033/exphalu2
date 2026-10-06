#!/usr/bin/env bash
# Batch hallucination-eval matrix: runs the full generation_eval pipeline
# (generate -> detect -> judge(gpt-oss-20b) -> evaluate) sequentially for every
# (model, dataset) pair. Each pair gets its own run_id under results/generation_eval/.
# Usage: run_batch.sh [config] [backend] [num_samples] [models_csv] [datasets_csv] [judge_provider]
set -uo pipefail
CATEGORY="generation_eval"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/log/$CATEGORY" "$ROOT/results/$CATEGORY"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$ROOT/log/$CATEGORY/run_batch-$TS.log"
exec > >(tee -a "$LOG") 2>&1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tim
cd "$ROOT"

CONFIG=${1:-config/generation_eval.yaml}
BACKEND=${2:-vllm}
NSAMPLES=${3:-16}
MODELS=${4:-qwen2_5_1_5b,qwen3_1_7b}
DATASETS=${5:-mmlu_pro,hle_text,competition_math_level5,gsm8k,math500}
JUDGE_PROVIDER=${6:-}

RESULTS_FILE="$ROOT/results/$CATEGORY/BATCH-$TS-summary.csv"
echo "model,dataset,run_id,status,log" > "$RESULTS_FILE"

IFS=',' read -ra MODEL_ARR <<< "$MODELS"
IFS=',' read -ra DS_ARR <<< "$DATASETS"

FAILED=0
for M in "${MODEL_ARR[@]}"; do
  for D in "${DS_ARR[@]}"; do
    RUN_TS=$(date +%Y%m%d-%H%M%S)
    PAIR_LOG="$ROOT/log/$CATEGORY/run_pipeline-$RUN_TS-$M-$D.log"
    echo "=== BATCH [$M x $D] $(date +%H:%M:%S) ==="
    if bash scripts/generation_eval/run_pipeline.sh "$CONFIG" "$M" "$BACKEND" \
         "$NSAMPLES" "$D" "$JUDGE_PROVIDER" > >(tee -a "$PAIR_LOG") 2>&1; then
      RUN_ID=$(grep -o 'geneval-[0-9]*-[0-9]*' "$PAIR_LOG" | tail -1)
      echo "$M,$D,${RUN_ID:-unknown},ok,$PAIR_LOG" >> "$RESULTS_FILE"
      echo "=== OK [$M x $D] run_id=$RUN_ID"
    else
      echo "$M,$D,unknown,FAILED,$PAIR_LOG" >> "$RESULTS_FILE"
      FAILED=$((FAILED+1))
      echo "=== FAILED [$M x $D] (log: $PAIR_LOG); continuing with next pair"
    fi
  done
done

echo
echo "== batch summary: $RESULTS_FILE =="
column -s, -t < "$RESULTS_FILE"
if [ "$FAILED" -gt 0 ]; then
  echo "BATCH FINISHED WITH $FAILED FAILED PAIRS"
  exit 1
fi
echo "BATCH OK"
