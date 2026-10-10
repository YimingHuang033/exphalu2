#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/run_cde.sh" train-e "${1:-config/cde.yaml}"
