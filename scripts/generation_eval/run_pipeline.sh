#!/usr/bin/env bash
# Full generation_eval pipeline on real data: generate -> detect -> judge -> evaluate.
# Usage: run_pipeline.sh [config] [model_key] [backend] [num_samples] [dataset] [judge_provider]
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
DATASET=${5:-}
JUDGE_PROVIDER=${6:-}
RUN_ID=${RUN_ID_OVERRIDE:-"geneval-$TS"}
COMMON=(--config "$CONFIG" --category generation_eval --run-id "$RUN_ID" --model "$MODEL" --backend "$BACKEND" --num-samples "$NSAMPLES" --strict-env)
if [ -n "$DATASET" ]; then
  COMMON+=(--dataset "$DATASET")
fi
if [ -n "$JUDGE_PROVIDER" ]; then
  COMMON+=(--judge-provider "$JUDGE_PROVIDER")
fi

# Isolated judge service (adapter systemone) is managed around the judge stage:
# llama.cpp + startlux_decision run on both GPUs, so they must not overlap vLLM stages.
PROVIDER=${JUDGE_PROVIDER:-$(python -c "
import sys; sys.path.insert(0, '.')
from reppl2.config_loader import load_config
print(load_config(sys.argv[1]).get('judge', {}).get('provider', 'local_self'))" "$CONFIG")}

# A pre-existing service occupies the same GPUs and is not owned by this run.
if [ "$PROVIDER" = "startlux_local" ] && bash scripts/setup/startlux_service.sh status; then
  echo "StartLux is already running; stop it before this GPU pipeline."
  exit 1
fi

python -m reppl2.cli verify-backend "${COMMON[@]}" || exit 1
python -m reppl2.cli generate "${COMMON[@]}" || exit 1
python -m reppl2.cli detect "${COMMON[@]}" || exit 1

SVC_STARTED=0
if [ "$PROVIDER" = "startlux_local" ]; then
  if bash scripts/setup/startlux_service.sh start; then SVC_STARTED=1; else
    echo "startlux service failed to start; judge will fail honestly"; fi
  trap '[ "$SVC_STARTED" = "1" ] && bash scripts/setup/startlux_service.sh stop' EXIT
fi

python -m reppl2.cli judge "${COMMON[@]}" || echo "judge failed; evaluation uses em_gold labels"
python -m reppl2.cli evaluate "${COMMON[@]}" --label-source auto || exit 1

echo "PIPELINE OK: run_id=$RUN_ID"
echo "results: $ROOT/results/generation_eval/$RUN_ID (eval.csv, detection.json, judge.json, trajectory.json)"
echo "log:     $LOG"
