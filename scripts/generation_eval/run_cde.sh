#!/usr/bin/env bash
# Usage: bash scripts/generation_eval/run_cde.sh prepare config/cde.yaml [--force]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="${1:?stage: prepare|detect|train-e|evaluate}"
CONFIG="${2:-config/cde.yaml}"
if [ "$#" -ge 2 ]; then shift 2; else shift; fi
cd "$ROOT"
mkdir -p log/generation_eval
LOG="log/generation_eval/cde-${STAGE}-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
if [ -z "${CDE_PYTHON:-}" ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate tim
  CDE_PYTHON=python
fi
"$CDE_PYTHON" -m reppl2.cde "$STAGE" --config "$CONFIG" "$@"
