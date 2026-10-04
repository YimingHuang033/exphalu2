#!/usr/bin/env bash
# Interpretability exports: input mu/r/p_hat, B edit impacts, per-token NLL (interp category).
set -uo pipefail
CATEGORY="interp"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/log/$CATEGORY" "$ROOT/results/$CATEGORY"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$ROOT/log/$CATEGORY/run_interp-$TS.log"
exec > >(tee -a "$LOG") 2>&1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tim
cd "$ROOT"

CONFIG=${1:-config/interp.yaml}
MODEL=${2:-qwen3_5_4b}
BACKEND=${3:-vllm}
NSAMPLES=${4:-8}
RUN_ID="interp-$TS"
COMMON=(--config "$CONFIG" --category interp --run-id "$RUN_ID" --model "$MODEL" --backend "$BACKEND" --num-samples "$NSAMPLES" --strict-env)

python -m reppl2.cli generate "${COMMON[@]}" || exit 1
python -m reppl2.cli detect "${COMMON[@]}" || exit 1

echo "INTERP OK: run_id=$RUN_ID"
echo "results: $ROOT/results/interp/$RUN_ID (detection.json holds per-sample interp blocks)"
echo "log:     $LOG"
