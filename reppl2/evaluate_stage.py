"""Evaluation stage + report aggregation."""
from __future__ import annotations

import string as _string
from pathlib import Path

from .cache import RunStore
from .config_loader import ConfigError
from .evaluation.metrics import evaluate_method, judge_agreement
from .logging_utils import load_json, save_json
from .pipeline import load_generation


def _norm(x) -> str:
    x = str(x).lower()
    x = "".join(c for c in x if c not in _string.punctuation)
    return " ".join(x.split())


def labels_from_gold(examples: dict, generations: list) -> dict:
    """Exact/substring match label: 1 = wrong answer (hallucination-ish). label_source=em_gold.

    Accepts Example dataclass instances or plain dicts (as stored in dataset.json).
    """
    labels = {}
    for g in generations:
        ex = examples.get(g["sample_id"])
        golds_raw = ex.get("gold_answers", []) if isinstance(ex, dict) else (ex.gold_answers or [])
        golds = [_norm(a) for a in golds_raw if a]
        pred = _norm(load_generation(g).greedy.text.split("\n")[0])
        ok = any(pred and (pred == gl or pred in gl or gl in pred) for gl in golds)
        labels[g["sample_id"]] = 0 if ok else 1
    return labels


def cmd_evaluate(cfg, logger, run_dir, label_source="auto") -> dict:
    import pandas as pd

    store = RunStore(run_dir, cfg["_config_hash_"])
    det = store.load_stage("detection")
    gen_payload = store.load_stage("generation")
    dataset_payload = load_json(run_dir / "dataset.json")
    ex_by_id = {e["sample_id"]: e for e in dataset_payload["examples"]}

    em_labels = labels_from_gold(ex_by_id, gen_payload["generations"])
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

    method_scores = {}
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
            [judge_labels.get(s) for s in ids if judge_labels.get(s) is not None],
            [em_labels.get(s) for s in ids if judge_labels.get(s) is not None])
            if judge_labels else None),
        "eval_csv": str(csv_path),
    }
    store.save_stage("evaluation", payload)
    logger.info(f"[eval] wrote {csv_path}")
    return payload
