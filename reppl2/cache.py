"""Run-level manifest + stage-file resumability. Content-hash cache for repeated replays."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from .logging_utils import load_json, save_json


def content_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class RunStore:
    """File layout per run: results/<category>/<run_id>/ with per-stage JSON files.

    A stage is skipped when its file exists AND the config hash matches the snapshot;
    otherwise the stage reruns and overwrites (config-change guard).
    """

    STAGES = ("config_snapshot", "generation", "trajectory", "detection", "judge", "evaluation", "interp")

    def __init__(self, run_dir: Path, config_hash: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config_hash = config_hash

    def path(self, stage: str) -> Path:
        if stage not in self.STAGES:
            raise ValueError(f"unknown stage '{stage}'")
        if stage == "config_snapshot":
            return self.run_dir / "config_snapshot.json"
        return self.run_dir / f"{stage}.json"

    def up_to_date(self, stage: str) -> bool:
        p = self.path(stage)
        if not p.exists():
            return False
        try:
            data = load_json(p)
        except Exception:
            return False
        return data.get("_config_hash_") == self.config_hash

    def save_stage(self, stage: str, payload: dict) -> Path:
        return save_json(self.path(stage), {"_config_hash_": self.config_hash, **payload})

    def load_stage(self, stage: str) -> dict:
        return load_json(self.path(stage))

    def snapshot_config(self, cfg: dict, extra_env: Optional[dict] = None) -> Path:
        snap = {k: v for k, v in cfg.items()}
        snap["_env_fingerprint_"] = extra_env or {}
        return save_json(self.run_dir / "config_snapshot.json", snap)


def replay_cache_key(backend_name: str, model_path: str, context_ids: list[int],
                     output_ids: list[int]) -> str:
    return content_hash({
        "backend": backend_name,
        "model": model_path,
        "ctx": list(context_ids),
        "out": list(output_ids),
    })
