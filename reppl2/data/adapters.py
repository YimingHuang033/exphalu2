from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterator

from ..types import Example
from ..config_loader import ConfigError

IMPLEMENTED_DATASETS = [
    "triviaqa", "synthetic", "gsm8k", "math500", "competition_math",
    "competition_math_level5", "gpqa_diamond", "mmlu_college_chemistry",
    "mmlu_college_computer_science", "mmlu_college_mathematics", "mmlu_pro",
    "hle_text",
]

DATASET_STATUS = {
    "triviaqa": "implemented",
    "synthetic": "implemented",
    "gsm8k": "implemented",
    "math500": "implemented",
    "competition_math": "implemented",
    "competition_math_level5": "implemented",
    "gpqa_diamond": "implemented",
    "mmlu_college_chemistry": "implemented",
    "mmlu_college_computer_science": "implemented",
    "mmlu_college_mathematics": "implemented",
    "mmlu_pro": "implemented",
    "hle_text": "implemented",
    "nq": "blocked:no-local-file",
    "squad": "blocked:raw-json-not-mounted",
    "coqa": "blocked:raw-json-not-mounted",
    "hle": "blocked:multimodal-items",
    "livebench": "blocked:not-factual-qa",
}

MATH_SYSTEM_DEFAULT = (
    "You are a helpful AI assistant. Solve the math problem step by step, keeping it brief. "
    "End your reply with the final answer on its own last line in the form: Final answer: <answer>"
)
MCQ_SYSTEM_DEFAULT = (
    "You are a helpful AI assistant. Answer the multiple-choice question. "
    "Reply with the correct option letter only."
)


def _short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:n]


def _ds_cfg(cfg: dict, name: str) -> dict:
    return (cfg.get("datasets") or {}).get(name) or {}


def _system_prompt(cfg: dict, name: str, default: str) -> str:
    return str(_ds_cfg(cfg, name).get("system_prompt") or default)


def gsm8k_gold(answer_text: str) -> str:
    """gsm8k stores reasoning + '#### <number>'; the ground truth is the tail."""
    for line in str(answer_text).strip().splitlines()[::-1]:
        if "####" in line:
            return line.split("####", 1)[1].strip().replace(",", "")
    return str(answer_text).strip().replace(",", "")


def mmlu_gold_letter(answer_idx) -> str:
    return "ABCDEFGHIJKL"[int(answer_idx)]


def _iter_math_jsonl(path: str, name: str, cfg: dict, limit: int,
                     problem_key: str, gold_fn) -> Iterator[Example]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"{name}: data file not found: {path}")
    system = _system_prompt(cfg, name, MATH_SYSTEM_DEFAULT)
    n = 0
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = str(row[problem_key]).strip()
            gold = gold_fn(row)
            if not q or not gold:
                continue
            yield Example(
                sample_id=f"{name}-{_short_hash(str(row.get('unique_id', q)))}",
                task="math",
                question=q,
                context="",
                gold_answers=[gold],
                system_prompt=system,
                split="test",
                meta={"dataset": name, "subject": row.get("subject"), "level": row.get("level")},
            )
            n += 1
            if limit and n >= limit:
                return


def iter_gsm8k(cfg: dict, limit: int) -> Iterator[Example]:
    path = _ds_cfg(cfg, "gsm8k").get("path")
    if not path:
        raise ConfigError("datasets.gsm8k.path missing in config")
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"gsm8k: data file not found: {path}")
    import pandas as pd

    df = pd.read_parquet(p).reset_index(drop=True)
    if limit and limit > 0:
        df = df.head(limit)
    system = _system_prompt(cfg, "gsm8k", MATH_SYSTEM_DEFAULT)
    for i, row in df.iterrows():
        q = str(row["question"]).strip()
        yield Example(
            sample_id=f"gsm8k-{_short_hash(q)}",
            task="math",
            question=q,
            context="",
            gold_answers=[gsm8k_gold(str(row["answer"]))],
            system_prompt=system,
            split="test",
            meta={"dataset": "gsm8k"},
        )


def iter_math500(cfg: dict, limit: int) -> Iterator[Example]:
    path = _ds_cfg(cfg, "math500").get("path")
    if not path:
        raise ConfigError("datasets.math500.path missing in config")
    yield from _iter_math_jsonl(path, "math500", cfg, limit, "problem",
                                lambda row: str(row.get("answer", "")).strip())


def iter_competition_math(cfg: dict, limit: int) -> Iterator[Example]:
    path = _ds_cfg(cfg, "competition_math").get("path")
    if not path:
        raise ConfigError("datasets.competition_math.path missing in config")
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"competition_math: data file not found: {path}")
    with open(p) as f:
        rows = json.load(f)
    system = _system_prompt(cfg, "competition_math", MATH_SYSTEM_DEFAULT)
    n = 0
    for row in rows:
        gold = row.get("numeric_answer")
        if gold is None:
            continue
        yield Example(
            sample_id=f"cmath-{_short_hash(str(row.get('id', row.get('problem', ''))))}",
            task="math",
            question=str(row["problem"]).strip(),
            context="",
            gold_answers=[str(gold)],
            system_prompt=system,
            split="test",
            meta={"dataset": "competition_math", "level": row.get("level"),
                  "type": row.get("type")},
        )
        n += 1
        if limit and n >= limit:
            return


def _render_mcq_question(question: str, choices: list[str]) -> str:
    """Letter the options; supports up to 12 options (MMLU-Pro uses up to 10)."""
    lines = [question.strip()]
    for letter, ch in zip("ABCDEFGHIJKL", choices):
        lines.append(f"{letter}. {str(ch).strip()}")
    return "\n".join(lines)


def _iter_mcq_json(path: str, name: str, cfg: dict, limit: int,
                   question_key: str, choices_key: str, answer_key: str) -> Iterator[Example]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"{name}: data file not found: {path}")
    with open(p) as f:
        rows = json.load(f)
    system = _system_prompt(cfg, name, MCQ_SYSTEM_DEFAULT)
    n = 0
    for i, row in enumerate(rows):
        gold = mmlu_gold_letter(row[answer_key])
        q = _render_mcq_question(str(row[question_key]), list(row[choices_key]))
        yield Example(
            sample_id=f"{name}-{_short_hash(str(row.get('id', i)))}",
            task="mcq",
            question=q,
            context="",
            gold_answers=[gold],
            system_prompt=system,
            split="test",
            meta={"dataset": name, "subject": row.get("subject")},
        )
        n += 1
        if limit and n >= limit:
            return


def iter_gpqa_diamond(cfg: dict, limit: int) -> Iterator[Example]:
    """GPQA-Diamond: the stored question already embeds the A-D choice block;
    the stored answer is the correct option letter.

    A minority of items use a shuffled remap block ("A. d\\nB. a\\nC. b\\nD. c")
    after an a)-d) option list; for those the block is stripped and the stored
    gold letter is mapped through it to the content option letter.
    """
    path = _ds_cfg(cfg, "gpqa_diamond").get("path")
    if not path:
        raise ConfigError("datasets.gpqa_diamond.path missing in config")
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"gpqa_diamond: data file not found: {path}")
    with open(p) as f:
        rows = json.load(f)
    system = _system_prompt(cfg, "gpqa_diamond", MCQ_SYSTEM_DEFAULT)
    n = 0
    for row in rows:
        gold = str(row.get("answer", "")).strip()
        q = str(row.get("question", "")).strip()
        if not gold or not q:
            continue
        q, gold = normalize_gpqa_item(q, gold)
        yield Example(
            sample_id=f"gpqa-{_short_hash(q)}",
            task="mcq",
            question=q,
            context="",
            gold_answers=[gold],
            system_prompt=system,
            split="test",
            meta={"dataset": "gpqa_diamond"},
        )
        n += 1
        if limit and n >= limit:
            return


_GPQA_REMAP_LINE_RE = re.compile(r"^([A-D])\.\s*([a-d])\s*$", re.M)


def normalize_gpqa_item(question: str, gold: str) -> tuple[str, str]:
    """Strip a trailing shuffled remap block and translate the gold letter through it.

    A remap block is 4 consecutive lines like 'A. d' / 'B. a' mapping presented
    letters to content option letters. Returns (clean_question, gold_letter).
    """
    lines = question.rstrip().splitlines()
    tail = lines[-4:] if len(lines) >= 4 else []
    matches = [_GPQA_REMAP_LINE_RE.match(ln.strip()) for ln in tail]
    if tail and all(matches):
        mapping = {m.group(1): m.group(2) for m in matches}
        content_gold = mapping.get(gold)
        if content_gold:
            question = "\n".join(lines[:-4]).rstrip()
            return question, content_gold.upper()
    return question, gold


def iter_mmlu_college_chemistry(cfg: dict, limit: int) -> Iterator[Example]:
    yield from _iter_mcq_json(_ds_cfg(cfg, "mmlu_college_chemistry").get("path", ""),
                              "mmlu-chem", cfg, limit, "question", "choices", "answer")


def iter_mmlu_college_computer_science(cfg: dict, limit: int) -> Iterator[Example]:
    yield from _iter_mcq_json(_ds_cfg(cfg, "mmlu_college_computer_science").get("path", ""),
                              "mmlu-cs", cfg, limit, "question", "choices", "answer")


def iter_mmlu_college_mathematics(cfg: dict, limit: int) -> Iterator[Example]:
    yield from _iter_mcq_json(_ds_cfg(cfg, "mmlu_college_mathematics").get("path", ""),
                              "mmlu-math", cfg, limit, "question", "choices", "answer")


def iter_mmlu_pro(cfg: dict, limit: int) -> Iterator[Example]:
    """MMLU-Pro: reasoning-heavy MCQ with up to 10 options (hard tier for hallucination)."""
    path = _ds_cfg(cfg, "mmlu_pro").get("path")
    if not path:
        raise ConfigError("datasets.mmlu_pro.path missing in config")
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"mmlu_pro: data file not found: {path}")
    with open(p) as f:
        rows = json.load(f)
    system = _system_prompt(cfg, "mmlu_pro", MCQ_SYSTEM_DEFAULT)
    n = 0
    for row in rows:
        choices = [str(c) for c in row.get("options", []) if str(c).strip()]
        if len(choices) < 2:
            continue
        idx = row.get("answer_index")
        gold = (mmlu_gold_letter(idx) if idx is not None
                else str(row.get("answer", "")).strip().upper())
        if not gold:
            continue
        q = _render_mcq_question(str(row["question"]), choices)
        yield Example(
            sample_id=f"mmlupro-{_short_hash(str(row.get('question_id', q)))}",
            task="mcq",
            question=q,
            context="",
            gold_answers=[gold],
            system_prompt=system,
            split="test",
            meta={"dataset": "mmlu_pro", "subject": row.get("category")},
        )
        n += 1
        if limit and n >= limit:
            return


def iter_competition_math_level5(cfg: dict, limit: int) -> Iterator[Example]:
    """Level-5 competition math: the hardest tier of the local MATH extract."""
    path = _ds_cfg(cfg, "competition_math_level5").get("path")
    if not path:
        raise ConfigError("datasets.competition_math_level5.path missing in config")
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"competition_math_level5: data file not found: {path}")
    with open(p) as f:
        rows = json.load(f)
    system = _system_prompt(cfg, "competition_math_level5", MATH_SYSTEM_DEFAULT)
    n = 0
    for row in rows:
        gold = row.get("numeric_answer")
        if gold is None:
            continue
        yield Example(
            sample_id=f"cmath5-{_short_hash(str(row.get('id', row.get('problem', ''))))}",
            task="math",
            question=str(row["problem"]).strip(),
            context="",
            gold_answers=[str(gold)],
            system_prompt=system,
            split="test",
            meta={"dataset": "competition_math_level5", "level": row.get("level"),
                  "type": row.get("type")},
        )
        n += 1
        if limit and n >= limit:
            return


def iter_hle_text(cfg: dict, limit: int) -> Iterator[Example]:
    """Humanity's Last Exam, text-only exactMatch subset (image items excluded):
    frontier-difficult questions where small open models are confidently wrong."""
    path = _ds_cfg(cfg, "hle_text").get("path")
    if not path:
        raise ConfigError("datasets.hle_text.path missing in config")
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"hle_text: data file not found: {path}")
    with open(p) as f:
        rows = json.load(f)
    system = _system_prompt(cfg, "hle_text",
                            "You are a helpful AI assistant. Answer the question concisely. "
                            "State only the short final answer.")
    n = 0
    for row in rows:
        if row.get("image"):
            continue
        if str(row.get("answer_type", "")) != "exactMatch":
            continue
        gold = str(row.get("answer", "")).strip()
        q = str(row.get("question", "")).strip()
        if not gold or not q:
            continue
        yield Example(
            sample_id=f"hle-{_short_hash(str(row.get('id', q)))}",
            task="open_qa_hard",
            question=q,
            context="",
            gold_answers=[gold],
            system_prompt=system,
            split="test",
            meta={"dataset": "hle_text", "category": row.get("category"),
                  "subject": row.get("raw_subject")},
        )
        n += 1
        if limit and n >= limit:
            return


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
    registry = {
        "triviaqa": iter_triviaqa,
        "synthetic": iter_synthetic,
        "gsm8k": iter_gsm8k,
        "math500": iter_math500,
        "competition_math": iter_competition_math,
        "competition_math_level5": iter_competition_math_level5,
        "gpqa_diamond": iter_gpqa_diamond,
        "mmlu_college_chemistry": iter_mmlu_college_chemistry,
        "mmlu_college_computer_science": iter_mmlu_college_computer_science,
        "mmlu_college_mathematics": iter_mmlu_college_mathematics,
        "mmlu_pro": iter_mmlu_pro,
        "hle_text": iter_hle_text,
    }
    if name in registry:
        return registry[name](cfg, limit)
    status = DATASET_STATUS.get(name)
    if status is not None:
        raise ConfigError(
            f"dataset '{name}' status={status} (blocked on this machine or unsuitable "
            f"for ground-truth hallucination labeling); implemented now: {IMPLEMENTED_DATASETS}")
    raise ConfigError(f"unknown dataset '{name}'")
