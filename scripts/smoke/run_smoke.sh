#!/usr/bin/env bash
# End-to-end smoke: unit tests -> backend verify -> generate -> detect -> judge -> evaluate.
# Small synthetic dataset, small K. Artifacts in results/smoke/<run_id>/.
set -uo pipefail
CATEGORY="smoke"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/log/$CATEGORY" "$ROOT/results/$CATEGORY"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$ROOT/log/$CATEGORY/run_smoke-$TS.log"
exec > >(tee -a "$LOG") 2>&1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tim
cd "$ROOT"

CONFIG=${1:-config/smoke.yaml}
MODEL=${2:-qwen3_5_4b}
BACKEND=${3:-vllm}
RUN_ID="smoke-$TS"
COMMON=(--config "$CONFIG" --category smoke --run-id "$RUN_ID" --model "$MODEL" --backend "$BACKEND" --strict-env)

echo "=== [1/5] unit tests ==="
bash scripts/smoke/unit_tests.sh || exit 1

echo "=== [2/5] verify backend ($MODEL / $BACKEND) ==="
python -m reppl2.cli verify-backend "${COMMON[@]}" || exit 1

echo "=== [3/5] generate + detect ==="
python -m reppl2.cli generate "${COMMON[@]}" || exit 1
python -m reppl2.cli detect "${COMMON[@]}" || exit 1

echo "=== [4/5] judge ==="
python -m reppl2.cli judge "${COMMON[@]}" || echo "judge failed; evaluation falls back to em_gold labels"

echo "=== [5/5] evaluate ==="
python -m reppl2.cli evaluate "${COMMON[@]}" --label-source auto || exit 1

echo "SMOKE OK: run_id=$RUN_ID"
echo "results: $ROOT/results/smoke/$RUN_ID"
echo "log:     $LOG"
