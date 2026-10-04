"""Detection metrics per DESIGN.md §9.2. Score direction fixed: higher = hallucination.

No post-hoc sign flipping: a direction flip on the test set would be label leakage.
"""
from __future__ import annotations

import numpy as np


def _clean(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    keep = np.isfinite(scores)
    return scores[keep], labels[keep], int((~keep).sum())


def auroc(scores, labels) -> float:
    from sklearn.metrics import roc_auc_score

    s, y, _ = _clean(scores, labels)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def auprc(scores, labels) -> float:
    from sklearn.metrics import average_precision_score

    s, y, _ = _clean(scores, labels)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def positive_rate(labels) -> float:
    y = np.asarray(labels, dtype=np.float64)
    if len(y) == 0:
        return float("nan")
    return float(y.mean())


def bootstrap_ci(scores, labels, metric=auroc, n_boot: int = 1000, seed: int = 42,
                 unit_ids=None) -> dict:
    """Resampling unit: sample_id by default. Returns 95% CI of the metric."""
    rng = np.random.default_rng(seed)
    s_all = np.asarray(scores, dtype=np.float64)
    y_all = np.asarray(labels, dtype=np.int64)
    n = len(y_all)
    if n < 4:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0, "note": "too few samples"}
    idx = np.arange(n)
    vals = []
    for _ in range(n_boot):
        take = rng.choice(idx, size=n, replace=True)
        y_b = y_all[take]
        if len(np.unique(y_b)) < 2:
            continue
        vals.append(metric(s_all[take], y_b))
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}
    return {
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
        "n_boot": int(len(vals)),
    }


def evaluate_method(method: str, scores, labels) -> dict:
    s, y, n_invalid = _clean(scores, labels)
    n_total = len(np.asarray(labels))
    res = {
        "method": method,
        "n_samples": int(n_total),
        "n_invalid_scores": int(n_invalid),
        "n_used": int(len(y)),
        "positive_rate": positive_rate(y),
        "auroc": auroc(s, y),
        "auprc": auprc(s, y),
    }
    if len(y) >= 4:
        res["auroc_ci"] = bootstrap_ci(s, y)
    else:
        res["auroc_ci"] = {"ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}
    return res


def judge_agreement(judge_labels, detection_labels) -> dict:
    j = np.asarray(judge_labels)
    d = np.asarray(detection_labels)
    if len(j) != len(d):
        return {"agreement": float("nan"), "note": "length mismatch"}
    return {"agreement": float((j == d).mean()), "n": int(len(j))}
