from __future__ import annotations

import copy
import os
import re
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
# Bump when scoring/label semantics change so legacy artifacts cannot be reused.
CACHE_VERSION = 2

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(RuntimeError):
    pass


def _interpolate(node: Any, strict: bool) -> Any:
    if isinstance(node, dict):
        return {k: _interpolate(v, strict) for k, v in node.items()}
    if isinstance(node, list):
        return [_interpolate(v, strict) for v in node]
    if isinstance(node, str):
        def _sub(m):
            name = m.group(1)
            val = os.environ.get(name)
            if val is None:
                if strict:
                    raise ConfigError(f"environment variable '{name}' required by config but not set")
                return ""
            return val
        return _ENV_RE.sub(_sub, node)
    return node


def load_config(path: str | Path, strict_env: bool = False) -> dict:
    path = Path(path)
    if not path.is_absolute():
        if not path.exists():
            alt = CONFIG_DIR / path
            if alt.exists():
                path = alt
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = _interpolate(raw, strict_env)
    base_ref = cfg.pop("_include_base_", True)
    if base_ref:
        base_path = CONFIG_DIR / "base.yaml"
        if path.resolve() != base_path.resolve() and base_path.exists():
            base = load_config(base_path, strict_env)
            base = _deep_merge(base, cfg)
            cfg = base
    cfg["_config_path_"] = str(path)
    cfg["_config_hash_"] = config_hash(cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def config_hash(cfg: dict) -> str:
    payload = json.dumps({"cache_version": CACHE_VERSION,
                          "config": {k: v for k, v in cfg.items() if not k.startswith("_")}},
                         sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def require(cfg: dict, key_path: str):
    node: Any = cfg
    for part in key_path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"config key '{key_path}' missing")
        node = node[part]
    if node is None or (isinstance(node, str) and node.strip() == ""):
        raise ConfigError(f"config key '{key_path}' is empty; fill it in config/ or set its env var")
    return node
