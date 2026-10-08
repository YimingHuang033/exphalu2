#!/usr/bin/env bash
# Download DESIGN.md §9.4 dataset candidates to /mnt/data via hf-mirror.
# Freezes the resolved revision per dataset into a manifest (source_url/repo_id/
# revision/license/split file counts). Run: download_design94_datasets.sh [name ...]
set -uo pipefail
CATEGORY="setup"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$ROOT/log/$CATEGORY" "$ROOT/results/.."
TS=$(date +%Y%m%d-%H%M%S)
LOG="$ROOT/log/$CATEGORY/download_design94-$TS.log"
exec > >(tee -a "$LOG") 2>&1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tim
export HF_ENDPOINT=https://hf-mirror.com
cd /mnt/data

MANIFEST_DIR="$ROOT/config/manifests"
mkdir -p "$MANIFEST_DIR"

# name|repo_id|local_dir|repo_type
DATASETS=(
  "supergpqa|m-a-p/SuperGPQA|/mnt/data/SuperGPQA|dataset"
  "bigcodebench|bigcode/bigcodebench|/mnt/data/bigcodebench|dataset"
  "bfcl_hf|gorilla-llm/Berkeley-Function-Calling-Leaderboard|/mnt/data/BFCL-hf|dataset"
  "livecodebench|livecodebench/code_generation_lite|/mnt/data/LiveCodeBench|dataset"
  "multichallenge|ScaleAI/MultiChallenge|/mnt/data/MultiChallenge|dataset"
  "popqa|akariasai/PopQA|/mnt/data/PopQA|dataset"
  "cruxeval|cruxeval-org/cruxeval|/mnt/data/CRUXEval|dataset"
)

WANT=("$@")
if [ ${#WANT[@]} -eq 0 ]; then WANT=("${DATASETS[@]//|*/}"); fi

manifest_one () {  # name repo_id local_dir repo_type
  local name=$1 repo=$2 dir=$3 rtype=$4
  python - "$name" "$repo" "$dir" "$rtype" <<'EOF'
import json, os, sys
from datetime import datetime, timezone
from huggingface_hub import HfApi
name, repo, d, rtype = sys.argv[1:5]
info = (HfApi().dataset_info(repo, files_metadata=True) if rtype == "dataset"
        else HfApi().model_info(repo, files_metadata=True))
files = [(s.rfilename, s.size or 0) for s in info.siblings]
manifest = {
    "name": name, "repo_id": repo, "repo_type": rtype,
    "revision_sha": info.sha, "license": getattr(info, "license", None)
    or (info.cardData or {}).get("license"),
    "source_url": f"https://huggingface.co/{'datasets/' if rtype=='dataset' else ''}{repo}/tree/{info.sha}",
    "endpoint": os.environ.get("HF_ENDPOINT"), "design_section": "9.4",
    "downloaded_at": datetime.now(timezone.utc).isoformat(),
    "local_dir": d, "n_files": len(files),
    "total_bytes": sum(sz for _, sz in files),
    "files": [{"path": p, "size": sz} for p, sz in files],
}
out = sys.argv[1]
cfg = os.path.join(os.environ["MANIFEST_DIR"], out + ".manifest.json")
json.dump(manifest, open(cfg, "w"), indent=1, ensure_ascii=False)
print(f"manifest -> {cfg} (revision {info.sha[:12]}, {len(files)} files, "
      f"{sum(sz for _, sz in files)/1e6:.1f} MB)")
EOF
}
export MANIFEST_DIR="$MANIFEST_DIR"
export -f manifest_one 2>/dev/null || true

for SPEC in "${DATASETS[@]}"; do
  IFS='|' read -r NAME REPO DIR RTYPE <<< "$SPEC"
  SKIP=1; for W in "${WANT[@]}"; do [ "$W" = "$NAME" ] && SKIP=0; done
  [ "$SKIP" = "1" ] && continue
  # hf download is idempotent and resumes interrupted transfers; leftover .cache
  # state from a stalled download does not need special-casing
  echo "== [$NAME] hf download $REPO -> $DIR"
  if hf download --repo-type "$RTYPE" --max-workers 4 "$REPO" --local-dir "$DIR"; then
    manifest_one "$NAME" "$REPO" "$DIR" "$RTYPE"
    echo "== [$NAME] OK"
  else
    echo "== [$NAME] FAILED (recorded; no silent substitution) — rerun this script to resume"
  fi
done

# TruthfulQA comes from the author GitHub repo (a CSV, not a HF dataset)
if printf '%s\n' "${WANT[@]}" | grep -qx truthfulqa; then
  TDIR=/mnt/data/TruthfulQA
  if [ -f "$TDIR/TruthfulQA.csv" ]; then
    echo "== [truthfulqa] already present"
  else
    mkdir -p "$TDIR"
    if curl -sL --retry 3 -o "$TDIR/TruthfulQA.csv" \
        "https://ghproxy.net/https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"; then
      head -1 "$TDIR/TruthfulQA.csv" | grep -q Question && echo "== [truthfulqa] OK ($(wc -l < "$TDIR/TruthfulQA.csv") lines)" \
        || echo "== [truthfulqa] downloaded but content unexpected; inspect $TDIR"
    else
      echo "== [truthfulqa] download FAILED (recorded)"
    fi
  fi
  if [ -f "$TDIR/TruthfulQA.csv" ]; then
    python - "$TDIR" <<'EOF'
import csv, hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
d = sys.argv[1]
csv_path = os.path.join(d, "TruthfulQA.csv")
rows = list(csv.DictReader(open(csv_path)))
sha = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()
try:
    rev = subprocess.run(["git", "ls-remote", "https://github.com/sylinrl/TruthfulQA",
                          "HEAD"], capture_output=True, text=True, timeout=30
                         ).stdout.split()[0]
except Exception:
    rev = None
manifest = {
    "name": "truthfulqa", "repo_id": "sylinrl/TruthfulQA (GitHub)", "repo_type": "github",
    "revision_sha": rev, "license": "Apache-2.0 (code); dataset per repo README",
    "source_url": "https://github.com/sylinrl/TruthfulQA",
    "design_section": "9.4.2.G",
    "downloaded_at": datetime.now(timezone.utc).isoformat(),
    "local_dir": d, "sha256": sha, "n_rows": len(rows),
    "files": [{"path": "TruthfulQA.csv",
               "size": os.path.getsize(csv_path)}],
}
out = os.path.join(os.environ["MANIFEST_DIR"], "truthfulqa.manifest.json")
json.dump(manifest, open(out, "w"), indent=1, ensure_ascii=False)
print(f"manifest -> {out} ({len(rows)} rows, sha256 {sha[:12]})")
EOF
  fi
fi

echo "download script done; log: $LOG"
