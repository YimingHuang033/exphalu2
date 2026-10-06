#!/usr/bin/env bash
# Manage the isolated local StartLux-Decision judge service:
#   llama-server (llama.cpp, GGUF weights on both GPUs) + the official
#   startlux_decision.gguf_server (TypeSafe /v1/systemone on 127.0.0.1:8090).
# Service settings come from config/base.yaml (judge_providers.startlux_local.service).
# Usage: startlux_service.sh [start|stop|status|check]
set -uo pipefail
CATEGORY="setup"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/log/$CATEGORY"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$ROOT/log/$CATEGORY/startlux_service-$TS.log"
PID_FILE=/tmp/opencode/startlux-decision-service.pids
ACTION=${1:-status}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tim
cd "$ROOT"

read_service_cfg () {
python - <<'EOF'
import json
from reppl2.config_loader import load_config
cfg = load_config("config/base.yaml")
svc = ((cfg.get("judge_providers") or {}).get("startlux_local") or {}).get("service") or {}
svc["base_url"] = ((cfg.get("judge_providers") or {}).get("startlux_local") or {}).get("base_url")
print(json.dumps(svc))
EOF
}

SVC_JSON=$(read_service_cfg)
MODEL_DIR=$(python -c "import json;print(json.loads('''$SVC_JSON''')['model_dir'])")
LLAMA_BIN=$(python -c "import json;print(json.loads('''$SVC_JSON''')['llama_binary'])")
LLAMA_PORT=$(python -c "import json;print(json.loads('''$SVC_JSON''')['llama_port'])")
PORT=$(python -c "import json;print(json.loads('''$SVC_JSON''')['port'])")
BASE_URL=$(python -c "import json;print(json.loads('''$SVC_JSON''')['base_url'])")
CTX=$(python -c "import json;print(json.loads('''$SVC_JSON''').get('ctx', 16384))")
PARALLEL=$(python -c "import json;print(json.loads('''$SVC_JSON''').get('parallel', 4))")

GGUF=$(ls "$MODEL_DIR"/*.gguf 2>/dev/null | head -1)

health () {
  curl -s -m 3 "$BASE_URL" > /dev/null 2>&1
}
llama_health () {
  curl -s -m 3 "http://127.0.0.1:$LLAMA_PORT/health" 2>/dev/null | grep -q '"ok"'
}

is_running () { health; }

do_start () {
  if is_running; then echo "startlux service already running at $BASE_URL"; return 0; fi
  echo "== starting StartLux-Decision service (log: $LOG) =="
  if [ ! -x "$LLAMA_BIN" ]; then
    echo "llama-server not found at $LLAMA_BIN (see README: llama.cpp CUDA build)"; return 1
  fi
  if [ -z "$GGUF" ]; then echo "no .gguf under $MODEL_DIR"; return 1; fi
  nohup "$LLAMA_BIN" -m "$GGUF" -ngl 99 -c "$CTX" --parallel "$PARALLEL" \
      --port "$LLAMA_PORT" --host 127.0.0.1 > "$LOG" 2>&1 &
  LLAMA_PID=$!
  echo "$LLAMA_PID llama-server" > "$PID_FILE"
  # llama.cpp cold start with -ngl 99 on 28.6 GB takes a while
  for i in $(seq 1 120); do
    if llama_health; then echo "llama-server healthy (pid $LLAMA_PID, port $LLAMA_PORT) after ${i}0s"; break; fi
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then echo "llama-server died; see $LOG"; return 1; fi
    sleep 10
  done
  if ! llama_health; then echo "llama-server not healthy after timeout; see $LOG"; return 1; fi
  (cd "$MODEL_DIR" && nohup python -m startlux_decision.gguf_server \
      --model-dir . --llama "http://127.0.0.1:$LLAMA_PORT" --port "$PORT" \
      >> "$LOG" 2>&1 & echo $! > /tmp/opencode/startlux-decision-server.pid)
  DECISION_PID=$(cat /tmp/opencode/startlux-decision-server.pid)
  echo "$LLAMA_PID llama-server" > "$PID_FILE"; echo "$DECISION_PID startlux_decision" >> "$PID_FILE"
  for i in $(seq 1 30); do
    if health; then echo "startlux service ready at $BASE_URL (decision pid $DECISION_PID)"; return 0; fi
    sleep 2
  done
  echo "startlux decision server not ready; see $LOG"; return 1
}

do_stop () {
  if [ -f "$PID_FILE" ]; then
    while read -r PID NAME; do
      kill "$PID" 2>/dev/null && echo "stopped $NAME (pid $PID)" || echo "$NAME (pid $PID) already gone"
    done < "$PID_FILE"
    rm -f "$PID_FILE" /tmp/opencode/startlux-decision-server.pid
  else
    pkill -f startlux_decision.gguf_server 2>/dev/null && echo "stopped gguf_server (pkill)"
    pkill -f "llama-server.*$GGUF" 2>/dev/null && echo "stopped llama-server (pkill)"
  fi
  # VRAM release is not instantaneous; wait briefly
  for i in $(seq 1 12); do
    FREE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "$FREE" -lt 2000 ] && break
    sleep 5
  done
  echo "stop done"
}

do_status () {
  if is_running; then echo "RUNNING: $BASE_URL"; else echo "NOT RUNNING: $BASE_URL"; return 1; fi
}

do_check () {
  if ! is_running; then echo "service not running"; return 1; fi
  curl -s "$BASE_URL/v1/systemone" -H 'Content-Type: application/json' -d '{
    "state": {"question": "What is the capital of France?", "reference_ground_truth": "Paris", "answer": "Berlin"},
    "questions": {"factuality": {"type": "choice", "instructions": "Is the Answer correct? Choose exactly one option.",
                  "criteria": {"correct": "The Answer is correct and consistent.",
                               "hallucinated": "The Answer is wrong or unsupported."}}}}'
  echo
}

case "$ACTION" in
  start) do_start ;;
  stop)  do_stop ;;
  status) do_status ;;
  check) do_check ;;
  *) echo "usage: $0 [start|stop|status|check]"; exit 2 ;;
esac
