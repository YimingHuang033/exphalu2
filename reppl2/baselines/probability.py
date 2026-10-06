"""Probability/entropy-family baselines. Score direction: higher = more hallucination-prone."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..backends.base import BackendError


def outer_perplexity_risk(greedy_logprobs: list[float], eos_token_ids: Optional[set] = None,
                          token_ids: Optional[list[int]] = None) -> float:
    if token_ids is not None and len(token_ids) != len(greedy_logprobs):
        raise BackendError("greedy token/logprob length mismatch")
    n = len(greedy_logprobs)
    if eos_token_ids and token_ids:
        n = len(token_ids)
        while n > 0 and token_ids[n - 1] in eos_token_ids:
            n -= 1
    if n == 0:
        raise BackendError("no counted tokens for perplexity")
    lps = greedy_logprobs[:n]
    if any(lp is None or not math.isfinite(lp) for lp in lps):
        raise BackendError("non-finite logprob in greedy output")
    return float(-sum(lps) / n)


def lnpe_risk(samples_logprobs: list[list[float]], samples_token_ids: list[list[int]],
              eos_token_ids: Optional[set] = None) -> float:
    """Length-normalized predictive entropy over sampled outputs (risk direction)."""
    vals = []
    if len(samples_logprobs) != len(samples_token_ids):
        raise BackendError("sample/logprob count mismatch")
    for lps, ids in zip(samples_logprobs, samples_token_ids):
        if len(lps) != len(ids):
            raise BackendError("sample token/logprob length mismatch")
        n = len(ids) if not eos_token_ids else _valid_len(ids, eos_token_ids)
        if n == 0:
            continue
        seq = lps[:n]
        if any(lp is None or not math.isfinite(lp) for lp in seq):
            raise BackendError("non-finite logprob in sampled output")
        vals.append(-sum(seq) / n)
    if not vals:
        raise BackendError("no valid sampled outputs for LNPE")
    return float(np.mean(vals))


def _valid_len(ids: list[int], eos_token_ids: set) -> int:
    last = len(ids)
    while last > 0 and ids[last - 1] in eos_token_ids:
        last -= 1
    return last


def eigenscore_last_risk(states: list[np.ndarray], alpha: float = 1e-3) -> float:
    """EigenScore variant on LAST-layer mean-pooled output states (named variant; the
    original method specifies a middle layer). Risk direction: -mean(log10 eigenvalues)."""
    if len(states) < 2:
        raise BackendError("EigenScore needs >=2 sampled states")
    X = np.stack([np.asarray(s, dtype=np.float64).mean(axis=0) for s in states])
    if not np.isfinite(X).all():
        raise BackendError("non-finite pooled states in EigenScore")
    X = X - X.mean(axis=0, keepdims=True)
    d = X.shape[1]
    cov = (X.T @ X) / max(1, X.shape[0] - 1) + alpha * np.eye(d)
    try:
        s = np.linalg.svd(cov, compute_uv=False)
    except np.linalg.LinAlgError as e:
        raise BackendError(f"SVD failed in EigenScore: {e}")
    s = np.clip(s, 1e-30, None)
    return float(-np.mean(np.log10(s)))


def output_length_risk(sample_lens: list[int]) -> float:
    """Control covariate: mean valid sampled output length. Longer answers may correlate
    with hallucination in open QA; included as an explicit control, NOT a competitor."""
    if not sample_lens:
        raise BackendError("no lengths")
    return float(np.mean(sample_lens))


def semantic_entropy_risk(sample_texts: list[str], sample_logprobs: list[list[float]],
                          entailment_model=None, strict_entailment: bool = False,
                          eos_token_ids: Optional[set] = None,
                          sample_token_ids: Optional[list[list[int]]] = None) -> tuple[float, dict]:
    """Semantic entropy (Farquhar et al. style): entailment grouping + Rao predictive entropy.

    If entailment_model is None, uses deterministic normalized-text equality grouping and the
    result is named 'semantic-entropy-lexical' (a lexical clustering variant, NOT the original
    NLI-based method).
    """
    import re
    import string as _string

    def normalize(s: str) -> str:
        s = s.split("\n")[0]
        s = s.lower()
        s = "".join(ch for ch in s if ch not in _string.punctuation)
        return " ".join(s.split())

    def equivalent(a: str, b: str) -> bool:
        if entailment_model is None:
            return normalize(a) == normalize(b) and normalize(a) != ""
        p1 = entailment_model.check_implication(a, b)
        p2 = entailment_model.check_implication(b, a)
        if strict_entailment:
            return p1 == 2 and p2 == 2
        return (0 not in (p1, p2)) and (p1, p2) != (1, 1)

    n = len(sample_texts)
    if n < 2:
        raise BackendError("semantic entropy needs >=2 samples")
    ids = [-1] * n
    nxt = 0
    for i in range(n):
        if ids[i] == -1:
            ids[i] = nxt
            for j in range(i + 1, n):
                if equivalent(sample_texts[i], sample_texts[j]):
                    ids[j] = nxt
            nxt += 1

    def mean_ll(k: int) -> float:
        lps = sample_logprobs[k]
        ids_k = sample_token_ids[k] if sample_token_ids else list(range(len(lps)))
        m = len(ids_k) if not eos_token_ids else _valid_len(ids_k, eos_token_ids)
        m = max(1, min(m, len(lps)))
        seq = [lp for lp in lps[:m] if lp == lp]
        if not seq:
            return float("-inf")
        return sum(seq) / len(seq)

    log_liks = [mean_ll(k) for k in range(n)]
    total = None
    per_id = {}
    for uid in sorted(set(ids)):
        members = [log_liks[i] for i in range(n) if ids[i] == uid]
        m = max(members)
        lse = m + math.log(sum(math.exp(x - m) for x in members))
        per_id[uid] = lse
    lls = np.array(list(per_id.values()), dtype=np.float64)
    mx = lls.max()
    log_probs = lls - (mx + np.log(np.exp(lls - mx).sum()))
    probs = np.exp(log_probs)
    pe = float(-np.sum(probs * log_probs))
    variant = "semantic-entropy" if entailment_model is not None else "semantic-entropy-lexical"
    return pe, {"semantic_ids": ids, "variant": variant, "n_clusters": len(per_id)}
