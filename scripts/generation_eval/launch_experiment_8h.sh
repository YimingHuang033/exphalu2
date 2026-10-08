#!/usr/bin/env bash
# ONE-COMMAND 8-hour monitored experiment:
#   bash scripts/generation_eval/launch_experiment_8h.sh
#   bash scripts/generation_eval/launch_experiment_8h.sh [config] [model] [backend] \
#          [n_samples] [ds_gpu0] [ds_gpu1] [monitor_hours]
#
# What it does (all SSH-disconnect-proof):
#   1. launches two GPU-pinned generation_eval pipelines detached (setsid), one
#      dataset per GPU (default: config/generation_eval_reasoning.yaml, qwen3_5_4b,
#      supergpqa@GPU0 + popqa@GPU1, n=2000 each),
#   2. writes the monitor control file (runs + deadline = now + monitor_hours),
#   3. installs/refreshes ONE crontab entry (marked EXPHALU2-MONITOR) that wakes
#      `opencode run` every 30 minutes with a read-only status-check prompt;
#      the entry removes itself when the window expires,
#   4. verifies: crontab content + one synchronous monitor check.
set -uo pipefail
CATEGORY="generation_eval"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/log/$CATEGORY"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$ROOT/log/$CATEGORY/launch_experiment-$TS.log"
CONTROL="$ROOT/log/$CATEGORY/cron-monitor.control"
MONLOG="$ROOT/log/$CATEGORY/cron-monitor.log"
MARKER="EXPHALU2-MONITOR"
OCCMD="${OPENCODE_BIN:-$HOME/.opencode/bin/opencode}"
MODEL_AGENT="${OPENCODE_MODEL:-coding-plan/glm-5-3-flash}"

CONFIG=${1:-config/generation_eval_reasoning.yaml}
MODEL=${2:-qwen3_5_4b}
BACKEND=${3:-vllm}
NSAMPLES=${4:-2000}
DS0=${5:-supergpqa}     # runs on GPU0
DS1=${6:-popqa}         # runs on GPU1
HOURS=${7:-8}

exec > >(tee -a "$LOG") 2>&1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tim
cd "$ROOT"

RUN0="geneval-$TS-g0"
RUN1="geneval-$TS-g1"
echo "== [1/4] launching detached pipelines (SSH-disconnect-proof, setsid)"
CUDA_VISIBLE_DEVICES=0 RUN_ID_OVERRIDE="$RUN0" \
  setsid nohup bash scripts/generation_eval/run_pipeline.sh \
    "$CONFIG" "$MODEL" "$BACKEND" "$NSAMPLES" "$DS0" \
    > "$ROOT/log/$CATEGORY/queue-gpu0-$TS.log" 2>&1 < /dev/null &
PID0=$!
sleep 3   # stagger: run_ids + engine starts must not collide
CUDA_VISIBLE_DEVICES=1 RUN_ID_OVERRIDE="$RUN1" \
  setsid nohup bash scripts/generation_eval/run_pipeline.sh \
    "$CONFIG" "$MODEL" "$BACKEND" "$NSAMPLES" "$DS1" \
    > "$ROOT/log/$CATEGORY/queue-gpu1-$TS.log" 2>&1 < /dev/null &
PID1=$!
disown 2>/dev/null || true
echo "   GPU0: $DS0 x$NSAMPLES ($MODEL) run_id=$RUN0 pid=$PID0 -> log/$CATEGORY/queue-gpu0-$TS.log"
echo "   GPU1: $DS1 x$NSAMPLES ($MODEL) run_id=$RUN1 pid=$PID1 -> log/$CATEGORY/queue-gpu1-$TS.log"

now=$(date +%s)
deadline=$(( now + HOURS * 3600 ))
"$HOME/miniconda3/envs/tim/bin/python" - "$CONTROL" "$now" "$deadline" "$RUN0" "$RUN1" <<'EOF'
import json, sys
path, now, deadline, run0, run1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
json.dump({"started_epoch": now, "deadline_epoch": deadline,
           "runs": [run0, run1],
           "note": "GPU0/gpu0-queue log = run0; GPU1 = run1; cron disarms itself at deadline"},
          open(path, "w"), indent=1)
EOF
echo "== [2/4] control file -> $CONTROL (window: ${HOURS}h, until $(date -d "@$deadline" '+%F %T'))"

CRON_LINE="*/30 * * * * /usr/bin/flock -n /tmp/exphalu2-opencode-monitor.lock bash $ROOT/scripts/generation_eval/cron_check.sh >> $MONLOG 2>&1 $MARKER"
(crontab -l 2>/dev/null | grep -v "$MARKER"; echo "$CRON_LINE") | crontab -
echo "== [3/4] crontab installed (every 30 min; auto-removes at deadline):"
crontab -l | grep "$MARKER" | sed 's/^/   /'

if [ ! -x "$OCCMD" ]; then
  echo "WARN: opencode binary not found at $OCCMD; cron checks will log and skip."
fi
echo "== [4/4] running one synchronous monitor check to verify the wiring:"
bash "$ROOT/scripts/generation_eval/cron_check.sh"
echo
echo "DONE. One-command summary:"
echo "  - experiment: 2 detached pipelines (survive SSH close); logs log/$CATEGORY/queue-gpu{0,1}-$TS.log"
echo "  - monitoring: crontab wakes opencode every 30 min for ${HOURS}h -> $MONLOG"
echo "  - check anytime:   tail -n 40 $MONLOG"
echo "  - stop monitoring: crontab -l | grep -v $MARKER | crontab -"
echo "  - stop experiment: kill $PID0 $PID1 (and their engine children)"
