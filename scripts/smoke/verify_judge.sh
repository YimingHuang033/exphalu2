#!/usr/bin/env bash
# Acceptance of the configured judge provider on canned ground-truth cases
# (default provider per config; e.g. gpt-oss-20b or startlux_local).
# Writes results/smoke/<run_id>/verify_judge.json.
# Usage: verify_judge.sh [config] [backend] [judge_provider]
set -uo pipefail
CATEGORY="smoke"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/log/$CATEGORY" "$ROOT/results/$CATEGORY"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$ROOT/log/$CATEGORY/verify_judge-$TS.log"
exec > >(tee -a "$LOG") 2>&1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tim
cd "$ROOT"

CONFIG=${1:-config/generation_eval.yaml}
BACKEND=${2:-vllm}
JUDGE_PROVIDER=${3:-}
RUN_ID="verifyjudge-$TS"
EXTRA=()
if [ -n "$JUDGE_PROVIDER" ]; then EXTRA+=(--judge-provider "$JUDGE_PROVIDER"); fi

PROVIDER=${JUDGE_PROVIDER:-$(python -c "
import sys; sys.path.insert(0, '.')
from reppl2.config_loader import load_config
print(load_config('$CONFIG').get('judge', {}).get('provider', 'local_self'))")}
SVC_STARTED=0
if [ "$PROVIDER" = "startlux_local" ]; then
  if bash scripts/setup/startlux_service.sh start; then SVC_STARTED=1; else
    echo "startlux service failed to start"; exit 1; fi
  trap '[ "$SVC_STARTED" = "1" ] && bash scripts/setup/startlux_service.sh stop' EXIT
fi

python -m reppl2.cli verify-judge --config "$CONFIG" --category smoke --run-id "$RUN_ID" \
  --backend "$BACKEND" --strict-env "${EXTRA[@]}"
RC=$?
echo "verify-judge exit=$RC  run_id=$RUN_ID"
echo "log: $LOG"
exit $RC
