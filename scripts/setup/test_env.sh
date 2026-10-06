#!/usr/bin/env bash
# Optional isolated CPU test environment; does not modify the server's conda env.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/log/setup"
exec > >(tee -a "$ROOT/log/setup/test_env-$(date +%Y%m%d-%H%M%S).log") 2>&1
cd "$ROOT"
"${1:-python3}" -m venv .venv
.venv/bin/python -m pip install -r config/requirements-test.txt
TEST_PYTHON="$ROOT/.venv/bin/python" bash scripts/smoke/unit_tests.sh
