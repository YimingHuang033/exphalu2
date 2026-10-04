#!/usr/bin/env bash
# Real-GPU backend acceptance: generation + token logprobs + token_embed last-hidden states.
# Usage: verify_backend.sh [config] [model_key] [backend]   (defaults: smoke.yaml, qwen3_5_4b, vllm)
set -uo pipefail
CATEGORY="smoke"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/log/$CATEGORY"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$ROOT/log/$CATEGORY/verify_backend-$TS.log"
exec > >(tee -a "$LOG") 2>&1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tim
cd "$ROOT"

CONFIG=${1:-config/smoke.yaml}
MODEL=${2:-qwen3_5_4b}
BACKEND=${3:-vllm}
ARGS=(--config "$CONFIG" --category smoke --model "$MODEL" --backend "$BACKEND" --strict-env)
if [ -n "${REPPLE_NUM_SAMPLES:-}" ]; then ARGS+=(--num-samples "$REPPLE_NUM_SAMPLES"); fi

python -m reppl2.cli verify-backend "${ARGS[@]}"
STATUS=$?
echo "verify-backend exit=$STATUS; log=$LOG"
exit $STATUS
