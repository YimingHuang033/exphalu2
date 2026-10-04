from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

from ..types import Example
from ..config_loader import ConfigError

DATASET_STATUS = {
    "triviaqa": "implemented",
    "nq": "blocked:no-local-file",
    "squad": "blocked:raw-json-not-mounted",
    "coqa": "blocked:raw-json-not-mounted",
    "synthetic": "implemented",
}


def _short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:n]


def iter_triviaqa(cfg: dict, limit: int) -> Iterator[Example]:
    path = cfg.get("datasets", {}).get("triviaqa", {}).get("path")
    if not path:
        raise ConfigError("datasets.triviaqa.path missing in config")
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("validation-*.parquet"))
        if not files:
            raise ConfigError(f"no validation-*.parquet under {p}")
    else:
        files = [p]
    import pandas as pd

    df = pd.read_parquet(files[0])
    df = df.reset_index(drop=True)
    if limit and limit > 0:
        df = df.head(limit)
    system = cfg.get("datasets", {}).get("triviaqa", {}).get("system_prompt",
        "You are a helpful AI assistant. Answer the user's question concisely. Avoid full sentences.")
    for _, row in df.iterrows():
        q = str(row["question"]).strip()
        if not q.endswith("?"):
            q = q + "?"
        ans = row["answer"]
        gold = []
        if isinstance(ans, dict):
            if ans.get("value"):
                gold.append(str(ans["value"]))
            aliases = ans.get("aliases")
            if aliases is not None:
                gold.extend([str(a) for a in list(aliases)[:5]])
        elif ans:
            gold = [str(ans)]
        yield Example(
            sample_id=f"triviaqa-{_short_hash(str(row.get('question_id', q)))}",
            task="open_qa",
            question=q,
            context="",
            gold_answers=gold,
            system_prompt=system,
            split="validation",
            meta={"dataset": "triviaqa"},
        )


def iter_synthetic(cfg: dict, limit: int) -> Iterator[Example]:
    """Deterministic tiny fixture set for smoke tests (no external files)."""
    items = [
        ("What is the capital of France?", "Paris"),
        ("What is the capital of Australia?", "Canberra"),
        ("Who wrote the novel '1984'?", "George Orwell"),
        ("Which planet is known as the Red Planet?", "Mars"),
        ("What is the chemical symbol for gold?", "Au"),
        ("How many continents are there on Earth?", "Seven"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("What is the largest ocean on Earth?", "Pacific Ocean"),
        ("In which year did the Titanic sink?", "1912"),
        ("What is the smallest prime number?", "Two"),
    ]
    system = "You are a helpful AI assistant. Answer the user's question concisely. Avoid full sentences."
    for i, (q, a) in enumerate(items[: limit if limit else len(items)]):
        yield Example(
            sample_id=f"synthetic-{i:03d}",
            task="open_qa_synthetic",
            question=q,
            context="",
            gold_answers=[a],
            system_prompt=system,
            split="validation",
            meta={"dataset": "synthetic"},
        )


def iter_squad(cfg: dict, limit: int) -> Iterator[Example]:
    path = cfg.get("datasets", {}).get("squad", {}).get("raw_json")
    if not path or not Path(path).exists():
        raise ConfigError(
            "SQuAD raw json path missing/unmounted; set datasets.squad.raw_json in config. "
            "Status: blocked (adapter implemented, data unavailable on this machine)."
        )
    import json

    with open(path) as f:
        data = json.load(f)["data"]
    n = 0
    system = "You are a helpful AI assistant. Answer the user's question based on the provided context concisely."
    for article in data:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                if qa.get("is_impossible") or not qa.get("answers"):
                    continue
                yield Example(
                    sample_id=f"squad-{_short_hash(qa['id'])}",
                    task="rag_qa",
                    question=qa["question"],
                    context=para["context"],
                    gold_answers=[qa["answers"][0]["text"]],
                    system_prompt=system,
                    split="validation",
                    meta={"dataset": "squad"},
                )
                n += 1
                if limit and n >= limit:
                    return


def get_dataset_iterator(name: str, cfg: dict, limit: int) -> Iterator[Example]:
    if name == "triviaqa":
        return iter_triviaqa(cfg, limit)
    if name == "synthetic":
        return iter_synthetic(cfg, limit)
    if name == "squad":
        return iter_squad(cfg, limit)
    if name == "nq":
        raise ConfigError("NQ local file unavailable on this machine; status blocked. Use 'triviaqa' or 'synthetic'.")
    if name == "coqa":
        raise ConfigError("CoQA raw json not mounted; status blocked. Use 'triviaqa' or 'synthetic'.")
    raise ConfigError(f"unknown dataset '{name}'")
