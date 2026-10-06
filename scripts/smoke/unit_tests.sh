#!/usr/bin/env bash
# Fixed-array math tests (DESIGN.md M1 acceptance).
set -euo pipefail
CATEGORY="smoke"
mkdir -p "$(dirname "$0")/../../log/$CATEGORY"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$(dirname "$0")/../../log/$CATEGORY/unit_tests-$TS.log"
exec > >(tee -a "$LOG") 2>&1
if [ -z "${TEST_PYTHON:-}" ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate tim
  TEST_PYTHON=python
fi
cd "$(dirname "$0")/../.."
"$TEST_PYTHON" -m pytest tests/ -q "$@"
