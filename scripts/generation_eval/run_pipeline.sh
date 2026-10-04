#!/usr/bin/env bash
# Full generation_eval pipeline on real data: generate -> detect -> judge -> evaluate.
# Usage: run_pipeline.sh [config] [model_key] [backend] [num_samples]
set -uo pipefail
CATEGORY="generation_eval"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/log/$CATEGORY" "$ROOT/results/$CATEGORY"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$ROOT/log/$CATEGORY/run_pipeline-$TS.log"
exec > >(tee -a "$LOG") 2>&1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tim
cd "$ROOT"

CONFIG=${1:-config/generation_eval.yaml}
MODEL=${2:-qwen3_5_4b}
BACKEND=${3:-vllm}
NSAMPLES=${4:-16}
RUN_ID="geneval-$TS"
COMMON=(--config "$CONFIG" --category generation_eval --run-id "$RUN_ID" --model "$MODEL" --backend "$BACKEND" --num-samples "$NSAMPLES" --strict-env)

python -m reppl2.cli verify-backend "${COMMON[@]}" || exit 1
python -m reppl2.cli generate "${COMMON[@]}" || exit 1
python -m reppl2.cli detect "${COMMON[@]}" || exit 1
python -m reppl2.cli judge "${COMMON[@]}" || echo "judge failed; evaluation uses em_gold labels"
python -m reppl2.cli evaluate "${COMMON[@]}" --label-source auto || exit 1

echo "PIPELINE OK: run_id=$RUN_ID"
echo "results: $ROOT/results/generation_eval/$RUN_ID (eval.csv, detection.json, judge.json, trajectory.json)"
echo "log:     $LOG"
