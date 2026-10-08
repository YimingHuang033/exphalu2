"""Evaluation stage + report aggregation."""
from __future__ import annotations

import string as _string
import re
from pathlib import Path

from .cache import RunStore
from .config_loader import ConfigError
from .evaluation.metrics import evaluate_method, judge_agreement
from .logging_utils import load_json, save_json
from .pipeline import load_generation
from .scoring import split_think


def _norm(x) -> str:
    x = str(x).lower()
    # Keep numeric signs, decimal points and fractions distinct.
    x = x.strip().rstrip(".!?")
    x = "".join(c for c in x if c not in _string.punctuation or c in "-+./")
    return " ".join(x.split())


def labels_from_gold(examples: dict, generations: list,
                     thinking_on: bool = False,
                     require_pattern: str = "") -> dict:
    """Conservative normalized exact match: 1 = wrong answer, label_source=em_gold.

    Accepts Example dataclass instances or plain dicts (as stored in dataset.json).
    With thinking enabled, truncated thinks (no </think> in the output) carry no
    judgeable answer and are excluded honestly rather than auto-labelled wrong.
    With a dataset require_pattern, outputs missing the instructed answer marker
    (truncated reasoning or instruction non-compliance) are format errors and are
    excluded the same way (DESIGN 9.5.3), never auto-labelled hallucination.
    """
    labels = {}
    for g in generations:
        ex = examples.get(g["sample_id"])
        if ex is None:
            continue
        golds_raw = (ex.get("gold_answers") or []) if isinstance(ex, dict) else (ex.gold_answers or [])
        golds = [v for a in golds_raw if a is not None and (v := _norm(a))]
        if not golds:
            continue
        greedy_text = load_generation(g).greedy.text
        if thinking_on and "</think>" not in greedy_text:
            continue
        answer = split_think(greedy_text)[1].strip()
        if require_pattern and not re.search(require_pattern, answer, re.I):
            continue
        explicit = re.findall(r"^\s*(?:final answer|answer)\s*:\s*(.+)$", answer, re.I | re.M)
        pred = _norm(explicit[-1] if explicit else answer.split("\n")[0])
        ok = bool(pred) and pred in golds
        labels[g["sample_id"]] = 0 if ok else 1
    return labels


def cmd_evaluate(cfg, logger, run_dir, label_source="auto") -> dict:
    import pandas as pd

    store = RunStore(run_dir, cfg["_config_hash_"])
    det = store.load_stage("detection")
    gen_payload = store.load_stage("generation")
    dataset_payload = load_json(run_dir / "dataset.json")
    ex_by_id = {e["sample_id"]: e for e in dataset_payload["examples"]}

    ds_name = str(dataset_payload.get("dataset") or "")
    ds_cfg = (cfg.get("datasets") or {}).get(ds_name) or {}
    em_labels = labels_from_gold(
        ex_by_id, gen_payload["generations"],
        thinking_on=bool((gen_payload.get("thinking") or {}).get("enabled", False)),
        require_pattern=str(ds_cfg.get("require_pattern") or ""))
    judge_labels, judge_scores = {}, {}
    if (run_dir / "judge.json").exists():
        jp = store.load_stage("judge")
        for r in jp["per_sample"]:
            if r.get("hard_verdict") is not None:
                judge_labels[r["sample_id"]] = int(r["hard_verdict"])
            judge_scores[r["sample_id"]] = r.get("continuous_score")

    if label_source == "judge" and not judge_labels:
        raise ConfigError("label_source=judge but judge.json has no hard verdicts; run judge stage")
    label_used = label_source
    if label_source == "auto":
        label_used = "judge" if judge_labels else "em_gold"
    src = judge_labels if label_used == "judge" else em_labels

    ids, labels = [], []
    for sid in sorted((g["sample_id"] for g in gen_payload["generations"])):
        lab = src.get(sid)
        if lab is None:
            continue
        ids.append(sid)
        labels.append(int(lab))
    if len(ids) == 0:
        raise ConfigError("no labeled samples; cannot evaluate")
    n_pos = sum(labels)
    logger.info(f"[eval] labels: n={len(ids)}, positives={n_pos} ({n_pos/len(ids):.1%}), "
                f"source={label_used}")

    method_scores = {m: {} for m in det.get("methods_requested", [])}
    for r in det["per_sample"]:
        for m, v in r["methods"].items():
            if not isinstance(v, dict):
                continue
            method_scores.setdefault(m, {})[r["sample_id"]] = v.get("risk")
            if m == "reppl-a":
                method_scores.setdefault("reppl-a-inner", {})[r["sample_id"]] = v.get("inner")
            if m == "reppl-b":
                method_scores.setdefault("reppl-b-inner", {})[r["sample_id"]] = v.get("inner")
    if judge_scores:
        method_scores["judge-continuous"] = {sid: judge_scores.get(sid) for sid in ids}

    rows, summary = [], {}
    for m, smap in sorted(method_scores.items()):
        scores = [smap.get(sid) for sid in ids]
        res = evaluate_method(m, scores, labels)
        rows.append(res)
        summary[m] = res
        logger.info(f"[eval] {m}: AUROC={res['auroc']:.4f} AUPRC={res['auprc']:.4f} "
                    f"invalid={res['n_invalid_scores']}"
                    if res["auroc"] == res["auroc"] else f"[eval] {m}: AUROC=nan")

    df = pd.DataFrame(rows)
    csv_path = run_dir / "eval.csv"
    df.to_csv(csv_path, index=False)
    payload = {
        "label_source": label_used,
        "n_samples": len(ids),
        "n_positive": n_pos,
        "per_method": summary,
        "judge_agreement_vs_em": (judge_agreement(
            [judge_labels[s] for s in ids if s in judge_labels and s in em_labels],
            [em_labels[s] for s in ids if s in judge_labels and s in em_labels])
            if judge_labels else None),
        "eval_csv": str(csv_path),
    }
    store.save_stage("evaluation", payload)
    logger.info(f"[eval] wrote {csv_path}")
    return payload
