"""Semantic Energy baseline (2025, arXiv:2508.14496; official temp repo
MaHuanAAA/SemanticEnergy, notebook `semantic_energy.ipynb`).

Official math (cells 0-3), ported 1:1 where the notebook is unambiguous:
- per sampled answer: a per-token scalar signal ("logits" in the notebook) and
  token probabilities; sequence prob = product of token probs;
- Boltzmann transform: boltzmann_i = -mean(signal_i);
- semantic clusters over the sampled answers; per cluster:
  prob_se = sum of sum-normalized sequence probs, logit_se = -sum(boltzmann_i);
- Semantic Energy score = value of the best (argmax logit_se) cluster.

Recorded deviations (honest, per DESIGN section 8.2):
- The notebook reads per-token scalars from cache files it does not regenerate
  (`item['logits']`); the generating code is not in the temp repo, so the exact
  signal source is ambiguous. We use the greedy-consistent per-token log-prob of
  each sampled answer as the signal and record `logit_signal: "token_logprob"`.
- The notebook's probs variant is plain semantic entropy (already implemented
  separately); this module implements the logits ("Semantic Energy") variant.
- Clusters come from the same entailment/lexical grouping used by our
  semantic-entropy baseline (the notebook consumed precomputed judgement files).
- A true-energy variant (raw vocab logits, E = -log Z) is NOT implemented: the
  native engine exposes normalized logprobs only; registered as planned.

Direction: the best cluster's summed mean logprob is high when one semantic mode
dominates confidently; risk = -best_value, so higher = more hallucination.
"""
from __future__ import annotations

import numpy as np

from ..backends.base import BackendError

VERSION = "semantic-energy-v1-official-port"


def boltzmann(tokens_signal: list[float]) -> float:
    """Official cal_boltzmann_logits: negated mean of the per-token signal."""
    if not tokens_signal:
        raise BackendError("semantic-energy: empty token signal")
    return -float(np.mean(np.asarray(tokens_signal, dtype=np.float64)))


def cal_cluster_ce(probs: list[float], logits: list[float],
                   clusters: list[list[int]]) -> tuple[list[float], list[float]]:
    """Official cal_cluster_ce: per-cluster sum-normalized prob and -sum(boltzmann)."""
    total = sum(probs)
    normalized = [p / total if total != 0 else 0.0 for p in probs]
    probs_se, logits_se = [], []
    for cluster in clusters:
        probs_se.append(float(sum(normalized[i] for i in cluster)))
        logits_se.append(float(-sum(logits[i] for i in cluster)))
    return probs_se, logits_se


def semantic_energy_risk(sample_token_logprobs: list[list[float]],
                         clusters: list[list[int]]) -> tuple[float, dict]:
    """sample_token_logprobs: K lists of per-token logprobs (one per sampled answer);
    clusters: semantic grouping of the K answers (from semantic-entropy grouping).
    Returns (risk, info) with risk = -max_cluster(sum_i mean-token-logprob_i)."""
    K = len(sample_token_logprobs)
    if K < 2:
        raise BackendError(f"semantic-energy needs K>=2 sampled answers, got {K}")
    if any(not lp for lp in sample_token_logprobs):
        raise BackendError("semantic-energy: empty token logprobs in the sample pool")
    if not clusters:
        raise BackendError("semantic-energy: no semantic clusters provided")
    seen = sorted(i for c in clusters for i in c)
    if seen != list(range(K)):
        raise BackendError(
            f"semantic-energy: clusters must partition 0..{K - 1}, got {clusters}")

    # official cal_probs: product of per-token probabilities = exp(sum of logprobs)
    probs = [float(np.exp(np.sum(np.asarray(lp, dtype=np.float64))))
             for lp in sample_token_logprobs]
    bolts = [boltzmann(lp) for lp in sample_token_logprobs]
    _, logits_se = cal_cluster_ce(probs, bolts, clusters)
    best = int(np.argmax(logits_se))
    risk = -float(logits_se[best])
    info = {"version": VERSION, "logit_signal": "token_logprob",
            "n_clusters": len(clusters), "best_cluster": clusters[best],
            "best_value": float(logits_se[best])}
    return risk, info
