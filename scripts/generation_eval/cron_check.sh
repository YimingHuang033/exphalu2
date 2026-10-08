#!/usr/bin/env bash
# Cron-driven opencode wake-up for exphalu2 experiment monitoring.
# Installed/removed automatically by launch_experiment_8h.sh; every invocation:
#   1. exits silently if the control file is gone or the monitoring window expired
#      (and removes its own crontab entry once expired),
#   2. otherwise wakes `opencode run` with a read-only status-check prompt and
#      appends the report to log/generation_eval/cron-monitor.log.
# This file runs from cron with a minimal environment: no conda needed, absolute
# paths only, HOME exported for opencode state access.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONTROL="$ROOT/log/generation_eval/cron-monitor.control"
MONLOG="$ROOT/log/generation_eval/cron-monitor.log"
OCCMD="${OPENCODE_BIN:-$HOME/.opencode/bin/opencode}"
MODEL="${OPENCODE_MODEL:-coding-plan/glm-5-3-flash}"
MARKER="EXPHALU2-MONITOR"
export HOME="${HOME:-/home/tim}"

log() { echo "[$(date '+%F %T')] $*" >> "$MONLOG"; }

disarm() {
  (crontab -l 2>/dev/null | grep -v "$MARKER") | crontab - 2>/dev/null || true
  log "monitoring window closed; crontab entry removed (experiment itself keeps running)."
}

mkdir -p "$(dirname "$MONLOG")"
if [ ! -f "$CONTROL" ]; then
  log "no control file; disarming."
  disarm
  exit 0
fi
deadline=$("$HOME/miniconda3/envs/tim/bin/python" -c "import json;print(json.load(open('$CONTROL'))['deadline_epoch'])" 2>/dev/null \
  || python3 -c "import json;print(json.load(open('$CONTROL'))['deadline_epoch'])" 2>/dev/null) || { log "unreadable control file; disarming."; disarm; exit 0; }
now=$(date +%s)
if [ "${now:-0}" -ge "${deadline:-0}" ]; then
  log "deadline reached ($deadline)."
  disarm
  exit 0
fi

{
  echo ""
  echo "===== opencode check @ $(date '+%F %T') (window ends @ $(date -d "@$deadline" '+%F %T' 2>/dev/null || date '+%F %T')) ====="
} >> "$MONLOG"
cd "$ROOT" || exit 0
PROMPT=$'这是 exphalu2 仓库的定时自动监控任务（cron 每 30 分钟唤醒一次，只读观察，禁止修改代码、禁止重启或新跑任何实验、禁止 commit）。\n\
执行顺序：\n\
1. 读 log/generation_eval/cron-monitor.control，取出 runs 列表（形如 geneval-*-g0 / -g1）。\n\
2. 对每个 run_id：查看 log/generation_eval/run_pipeline-<run_id>.log 的尾部，报告：当前阶段(generate/detect/judge/eval)、"[gen] <dataset>-..." 已完成的条数（统计行数）、是否出现 PIPELINE OK / FAILED。若日志未见进展或进程消失，如实报告。\n\
3. 出现 FAILED 时：读失败上下文报告原因，绝不自行重试。\n\
4. 对已 PIPELINE OK 的 run：读 results/generation_eval/<run_id>/eval.csv，汇报 positive_rate 和按 AUROC 排序前 3 的方法。\n\
5. nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 看双卡状态。\n\
6. 输出不超过 12 行的中文状态快照，最后一行给结论：正常推进 / 已完成 / 需人工介入（说明原因）。'
"$OCCMD" run -m "$MODEL" "$PROMPT" >> "$MONLOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  log "opencode run exited rc=$rc (non-fatal; next cron tick retries)."
fi
exit 0
