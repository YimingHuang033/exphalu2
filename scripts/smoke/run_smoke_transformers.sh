#!/usr/bin/env bash
# Legacy-port smoke: same pipeline on the Transformers compatibility backend.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec bash "$ROOT/scripts/smoke/run_smoke.sh" config/smoke_transformers.yaml "${1:-qwen3_5_4b}" transformers
