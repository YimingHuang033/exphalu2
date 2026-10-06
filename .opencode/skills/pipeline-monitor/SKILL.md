---
name: pipeline-monitor
description: Monitor running exphalu2 generation_eval pipelines (GPU status, per-queue progress, completed run results). Use when the user asks to 监控/查看实验状态, watch pipeline progress, or when long-running run_pipeline/run_batch jobs are active and periodic status reports are needed.
---

# Pipeline Monitor

Use ONLY for the exphalu2 repo's generation_eval category jobs.

## What to check, in order

1. Queues/logs: `log/generation_eval/queue-gpu*.log` (parallel driver) and
   `log/generation_eval/run_pipeline-*.log` (single runs). Tail the last lines.
2. Stage progress markers inside a log: `[gen]` (generate), `[detect]` (detect),
   `[judge]` (judge), `[eval] labels:` (evaluate), `PIPELINE OK` (done),
   `FAILED` (honest failure — never re-run without reading the reason).
3. Results: `results/generation_eval/geneval-*/eval.csv` — read
   `positive_rate`, per-method `auroc`/`auprc`; judge rows are named
   `judge-continuous` (label source) and come from the gpt-oss-20b judge
   (reference-aware, judge-yesno-v2).
4. GPU: `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv` —
   each queue is pinned via CUDA_VISIBLE_DEVICES; expect ~20.8 GiB vLLM engine
   while generate/detect runs, gpt-oss-20b (~14.7 GiB) during judge.
5. Snapshot command: `bash scripts/generation_eval/monitor_runs.sh [minutes]`
   tails every pipeline log touched within the window.

## Cadence

Default: report every 20 minutes (`sleep 1200` between checks) until all queues
print `PIPELINE OK` / `BATCH OK` or a FAILED reason. Report per queue: current
stage, samples done / total, elapsed time, and any errors verbatim.

## Rules

- Never restart failed jobs without quoting the failure reason from the log.
- Missing values in eval.csv are `null` by design; do not treat them as 0.
- Scores direction: higher = more hallucination.
