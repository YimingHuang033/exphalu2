"""RAUQ baseline (Vazhentsev et al., ICML 2026; official repo
mbzuai-nlp/rauq-hallucination-detection, file `rauq.py`).

Recurrent Attention-based Uncertainty Quantification: for each selected layer,
pick the head with maximal mean attention-to-previous-token, then combine token
confidence with previous-token attention in a recurrent scheme:

    conf_0 = exp(lp_0)
    conf_j = alpha * p_j + (1 - alpha) * attn[j -> j-1] * p_{j-1}     (default formula)
    UQ_layer = 1 - (mean(conf) + min(conf)) / 2        # token_aggregation="meanmin"
    UQ      = mean over layers                          # aggregation="mean"

Official defaults: alpha=0.2, head="max" (dynamic per-sequence head selection),
token_aggregation="meanmin", aggregation="mean", layers = middle third
(range(n//3, ceil(2n/3)+1)). Ported 1:1; no lm-polygraph dependency.

Signal source contract (DESIGN section 8.5): RAUQ needs attention weights, which
the native vLLM path does not expose. It therefore runs on the Transformers
reference environment only (legacy port); the vLLM backend reports a capability
error instead of silently substituting. Direction: higher UQ = more uncertainty
= higher hallucination risk.
"""
from __future__ import annotations

import math

import numpy as np

from ..backends.base import BackendError

VERSION = "rauq-v1-official-port"
DEFAULT_ALPHA = 0.2


def middle_third_layers(n_layers: int) -> list[int]:
    """Official default layer subset: range(n//3, ceil(2n/3)+1)."""
    return list(range(n_layers // 3, int(math.ceil(n_layers / 3 * 2)) + 1))


def prev_token_attention(attentions: np.ndarray) -> np.ndarray:
    """attentions: (L, H, T_total, T_total) attention of one forward pass over
    [context, answer]. Returns (L, H, T_total-1): entry [..., t-1] is the
    attention of absolute position t to its predecessor (position 0 has none,
    matching the official np.diagonal(offset=-1) extraction)."""
    A = np.asarray(attentions, dtype=np.float64)
    if A.ndim != 4:
        raise BackendError(f"rauq: attention must be (L,H,T,T), got {A.shape}")
    return np.diagonal(A, offset=-1, axis1=2, axis2=3)  # (L, H, T_total-1)


def rauq_risk(attentions: np.ndarray, token_logprobs: list[float], ctx_len: int,
              alpha: float = DEFAULT_ALPHA, layers=None, head: str = "max",
              token_aggregation: str = "meanmin", aggregation: str = "mean") -> dict:
    """attentions: (L, H, T_total, T_total) from a single forward over
    [context, greedy answer]; token_logprobs: greedy answer token logprobs
    (len T_answer); ctx_len: number of context tokens. The prev-token attention
    of answer token j (0-based) is read at absolute position ctx_len+j - matching
    the official recurrence, which starts the attention term at j=1.
    Returns {"risk": float, "heads": [...], ...}."""
    A_full = prev_token_attention(attentions)
    L, H, Tm1 = A_full.shape
    lp = np.asarray(token_logprobs, dtype=np.float64)
    T = lp.shape[0]
    if ctx_len < 1 or ctx_len - 1 + T > Tm1:
        raise BackendError(
            f"rauq: ctx_len {ctx_len} + answer {T} exceeds attention length {Tm1 + 1}")
    if T < 1 or not np.isfinite(lp).all():
        raise BackendError("rauq: empty or non-finite token logprobs")
    if not (0.0 < alpha < 1.0):
        raise BackendError(f"rauq: alpha must be in (0,1), got {alpha}")
    if not np.isfinite(A_full).all():
        raise BackendError("rauq: non-finite attention weights")
    # answer rows only: A[l, h, j] = attention of answer token j to its predecessor
    A = A_full[:, :, ctx_len - 1:ctx_len - 1 + T]
    sel_layers = middle_third_layers(L) if layers is None else list(layers)
    if not sel_layers:
        raise BackendError("rauq: empty layer selection")

    uq_layers, heads_used = [], []
    for layer in sel_layers:
        # dynamic head: maximal mean prev-token attention over the answer tokens
        if head == "max":
            h = int(A[layer].mean(axis=-1).argmax())
            attn_h = A[layer, h]
        elif head == "mean":
            h = -1
            attn_h = A[layer].mean(axis=0)
        else:
            raise BackendError(f"rauq: head mode {head!r} not ported (max|mean)")
        heads_used.append(h)

        # official default recurrence (ablation=None): conf feeds back recurrently
        # (p_i[-1] in the official code)
        conf = np.empty(T, dtype=np.float64)
        conf[0] = np.exp(lp[0])
        for j in range(1, T):
            p_j = np.exp(lp[j])
            conf[j] = alpha * p_j + (1 - alpha) * attn_h[j] * conf[j - 1]

        if token_aggregation == "meanmin":
            uq = 1 - (np.mean(conf) + np.min(conf)) / 2
        elif token_aggregation == "mean":
            uq = 1 - np.mean(conf)
        elif token_aggregation == "min":
            uq = 1 - np.min(conf)
        elif token_aggregation == "max":
            uq = 1 - np.max(conf)
        else:
            raise BackendError(
                f"rauq: token_aggregation {token_aggregation!r} not ported")
        uq_layers.append(float(uq))

    if aggregation == "mean":
        risk = float(np.mean(uq_layers))
    elif aggregation == "median":
        risk = float(np.median(uq_layers))
    elif aggregation == "max":
        risk = float(np.max(uq_layers))
    else:
        raise BackendError(f"rauq: aggregation {aggregation!r} not ported")
    return {"risk": risk, "heads": heads_used, "layers": sel_layers,
            "uq_layers": uq_layers, "alpha": alpha, "version": VERSION}
