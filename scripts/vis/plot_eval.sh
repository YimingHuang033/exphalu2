#!/usr/bin/env bash
# Generate ROC / comparison plots from a finished run's eval.csv into vis/<category>/.
# Usage: plot_eval.sh <category> <run_id>
set -uo pipefail
CATEGORY=${1:-smoke}
RUN_ID=${2:?run_id required}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/log/vis" "$ROOT/vis/$CATEGORY"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$ROOT/log/vis/plot_eval-$TS.log"
exec > >(tee -a "$LOG") 2>&1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tim
cd "$ROOT"
python scripts/vis/plot_eval.py --run-dir "results/$CATEGORY/$RUN_ID" --out-dir "vis/$CATEGORY" || exit 1
echo "vis OK: $ROOT/vis/$CATEGORY; log=$LOG"
