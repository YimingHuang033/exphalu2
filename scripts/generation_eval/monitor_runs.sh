#!/usr/bin/env bash
# Snapshot of running/finished generation_eval work: tails every pipeline/queue
# log touched within the last N minutes (default 30), plus GPU status.
# Usage: monitor_runs.sh [minutes]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MIN=${1:-30}
echo "== monitor snapshot $(date '+%F %T') (logs touched in last ${MIN} min) =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo "-- active pipelines --"
pgrep -af "reppl2.cli" | sed 's/ /  # /2' | cut -c1-160 || true

echo "-- log tails (modified < ${MIN} min) --"
FOUND=0
while IFS= read -r f; do
  FOUND=1
  echo "[$(date -r "$f" '+%T')] ${f#$ROOT/}"
  grep -E "\[gen\]|\[detect\]|\[judge\]|\[eval\] labels|PIPELINE|FAILED|Traceback|Error" "$f" | tail -3 | sed 's/^/    /'
done < <(find "$ROOT/log/generation_eval" -name "*.log" -mmin -"$MIN" -not -name "monitor-*" | sort)

[ "$FOUND" = "0" ] && echo "  (no pipeline logs touched in the last ${MIN} min)"
echo "-- completed runs today (eval.csv mtime today) --"
find "$ROOT/results/generation_eval" -name "eval.csv" -newermt "$(date +%F)" | while read -r c; do
  rid=$(basename "$(dirname "$c")")
  line=$(grep -E "labels:|judge-continuous" "$c" 2>/dev/null | head -0)
  echo "  $rid  $(stat -c %y "$c" | cut -d. -f1)"
done
echo "== snapshot end =="
