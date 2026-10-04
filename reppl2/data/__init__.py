from __future__ import annotations

from pathlib import Path

from ..config_loader import ConfigError

DATASET_STATUS = {
    "triviaqa": "implemented",
    "synthetic": "implemented",
    "squad": "blocked:raw-json-not-mounted",
    "coqa": "blocked:raw-json-not-mounted",
    "nq": "blocked:no-local-file",
}
