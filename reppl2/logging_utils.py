from __future__ import annotations

import json
import logging
import math
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = PROJECT_ROOT / "log"
RESULTS_ROOT = PROJECT_ROOT / "results"
CATEGORIES = ("smoke", "generation_eval", "perf", "interp", "setup", "vis")


def setup_logging(category: str, name: str, run_id: str | None = None) -> tuple[logging.Logger, Path, str]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_dir = LOG_ROOT / category
    log_dir.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"{name}-{run_id}.log"
    logger = logging.getLogger(f"exphalu2.{category}.{name}.{run_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger, log_file, run_id


def results_dir(category: str, run_id: str) -> Path:
    d = RESULTS_ROOT / category / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as f:
            tmp_path = Path(f.name)
            json.dump(_json_safe(obj), f, indent=2, ensure_ascii=False,
                      allow_nan=False, default=_default_json)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return path


def _json_safe(obj):
    """Convert non-finite Python/numpy values to standard JSON null recursively."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _default_json(o):
    try:
        import numpy as np

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return v if v == v else None
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)
