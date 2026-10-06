#!/usr/bin/env bash
# Environment check: conda env, GPU, model file readability, dataset readability.
set -uo pipefail
CATEGORY="setup"
mkdir -p "$(dirname "$0")/../../log/$CATEGORY"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$(dirname "$0")/../../log/$CATEGORY/env_check-$TS.log"
exec > >(tee -a "$LOG") 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tim
cd "$(dirname "$0")/../.."
echo "== python: $(python --version) =="
python - <<'EOF'
import json
fp = {}
for mod in ("vllm", "transformers", "torch", "numpy", "pandas", "sklearn"):
    try:
        fp[mod] = __import__(mod).__version__
    except Exception as e:
        fp[mod] = f"MISSING ({e})"
print(json.dumps(fp, indent=2))
import torch
print("cuda:", torch.cuda.is_available(), "| gpus:", torch.cuda.device_count())
EOF

echo "== model file readability (bad-sector guard) =="
for f in \
  /home/tim/Proj/resource/Qwen3.5-4B/model.safetensors-00001-of-00002.safetensors \
  /mnt/data/Qwen2.5-0.5B-Instruct/model.safetensors \
  /mnt/data/Qwen2.5-1.5B-Instruct/model.safetensors \
  /mnt/data/Qwen2.5-7B-Instruct/model-00001-of-00004.safetensors \
  /mnt/data/Qwen3-1.7B/model-00001-of-00002.safetensors \
  /mnt/data/Qwen3-4B/model-00001-of-00003.safetensors \
  /mnt/data/Qwen3-8B/model-00001-of-00005.safetensors \
  /mnt/data/gpt-oss-20b/model-00000-of-00002.safetensors \
  /mnt/data/deberta-v2-xlarge-mnli/pytorch_model.bin ; do
  if [ -f "$f" ]; then
    if dd if="$f" of=/dev/null bs=8M count=1 status=none 2>/dev/null; then
      echo "READABLE(head): $f"
    else
      echo "UNREADABLE:     $f  <-- record as blocked, do not use"
    fi
  else
    echo "MISSING:        $f"
  fi
done

echo "== dataset readability =="
python - <<'EOF'
import json
import pandas as pd

def parquet(p, name):
    try:
        df = pd.read_parquet(p)
        print(f"{name} OK: {df.shape[0]} rows")
    except Exception as e:
        print(f"{name} FAILED: {e}")

def jsonl(p, name):
    try:
        n = sum(1 for line in open(p) if line.strip())
        print(f"{name} OK: {n} lines")
    except Exception as e:
        print(f"{name} FAILED: {e}")

def jsonfile(p, name):
    try:
        d = json.load(open(p))
        print(f"{name} OK: {len(d)} items")
    except Exception as e:
        print(f"{name} FAILED: {e}")

parquet("/mnt/data/trivia_qa/rc.nocontext/validation-00000-of-00001.parquet", "triviaqa")
parquet("/mnt/data/gsm8k/main/test-00000-of-00001.parquet", "gsm8k")
jsonl("/mnt/data/MATH-500/test.jsonl", "math500")
jsonfile("/mnt/data/datasets/competition_math_level_balanced_200.json", "competition_math")
jsonfile("/mnt/data/datasets/GPQA-Diamond.json", "gpqa_diamond")
jsonfile("/mnt/data/datasets/MMLU_College_Chemistry_full.json", "mmlu_college_chemistry")
jsonfile("/mnt/data/datasets/MMLU_College_Computer_Science_full.json", "mmlu_college_computer_science")
jsonfile("/mnt/data/datasets/MMLU_College_Math_full.json", "mmlu_college_mathematics")
EOF
echo "env check done; log at $LOG"
