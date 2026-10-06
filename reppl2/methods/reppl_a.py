from __future__ import annotations

from typing import Optional

import numpy as np

from ..backends.base import BackendError
from ..types import Generation
from .common import AggregatorConfig, SharedAggregate, aggregate_A, cosine_matrix, row_softmax, l2_normalize


def reppl_a_inner(gen: Generation, replay_states: dict[str, np.ndarray],
                  unit_token_spans: list[dict], association_temperature: float,
                  agg_cfg: AggregatorConfig, outer: Optional[float],
                  require_valid_units: bool = True) -> tuple[SharedAggregate, dict]:
    """Method A per DESIGN.md §4.

    replay_states: key 'context' -> (L_ctx, H) PROMPT-side last-hidden states from replay of
                   [ctx, greedy_out] (len == len(prompt_token_ids)); key 'samples' -> list over
                   k of (L_out_k, H) output-token states for [ctx, sampled_out_k].
    Context-side unit representation u[j] is mean-pooled from the GREEDY replay's prompt states;
    per-sample output token states v[k,t] come from each sample's own replay.
    """
    ctx_states = replay_states["context"]
    L_ctx = ctx_states.shape[0]
    good = [u for u in unit_token_spans
            if u.get("is_factual") and 0 <= u.get("token_start", -1) < u.get("token_end", -1) <= L_ctx]
    if require_valid_units and not good:
        raise BackendError("no input unit could be aligned to token spans")
    mus = []
    for u in good:
        mus.append(ctx_states[u["token_start"]:u["token_end"]].mean(axis=0))
    U = np.stack(mus).astype(np.float64)

    K = len(replay_states["samples"])
    A = np.zeros((K, len(good)), dtype=np.float64)
    raw_sims = np.zeros((K, len(good)), dtype=np.float64)
    for k_idx, states in enumerate(replay_states["samples"]):
        sims = cosine_matrix(states.astype(np.float64), U)
        w = row_softmax(sims, axis=-1, temperature=association_temperature)
        A[k_idx] = w.mean(axis=0)
        raw_sims[k_idx] = sims.max(axis=0)

    agg = aggregate_A(A, outer_nll_sum=None, mean_sample_len=None, cfg=agg_cfg)
    info = {
        "units": good,
        "A": A,
        "raw_max_similarity": raw_sims,
        "association_temperature": association_temperature,
    }
    if outer is not None:
        eps = agg_cfg.epsilon
        if agg.validity.status == "ok":
            agg.risk = float((agg.inner + eps) * outer)
            agg.outer = float(outer)
    return agg, info
