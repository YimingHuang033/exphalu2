#!/usr/bin/env bash
# Performance benchmark: per-stage latency, tokens/s, peak VRAM (perf category).
set -uo pipefail
CATEGORY="perf"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/log/$CATEGORY" "$ROOT/results/$CATEGORY"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$ROOT/log/$CATEGORY/run_perf-$TS.log"
exec > >(tee -a "$LOG") 2>&1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tim
cd "$ROOT"

CONFIG=${1:-config/perf.yaml}
MODEL=${2:-qwen3_5_4b}
BACKEND=${3:-vllm}
RUN_ID="perf-$TS"

python -m reppl2.cli perf --config "$CONFIG" --category perf --run-id "$RUN_ID" \
  --model "$MODEL" --backend "$BACKEND" --strict-env || exit 1
echo "PERF OK: run_id=$RUN_ID"
echo "results: $ROOT/results/perf/$RUN_ID (perf.csv, perf_summary.json)"
echo "log:     $LOG"
