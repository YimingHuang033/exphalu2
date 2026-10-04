from __future__ import annotations

import numpy as np

from ..backends.base import BackendError
from .common import AggregatorConfig, aggregate_A, l2_normalize


def reppl_b_inner(z: np.ndarray, z_edit: dict[int, np.ndarray],
                  agg_cfg: AggregatorConfig, outer: float | None) -> tuple[dict, dict]:
    """Method B per DESIGN.md §5.

    z: (K, H) pooled output states under original input, one row per sampled answer k.
    z_edit: unit_id -> (K, H) pooled output states under edited input x^-j (same fixed output ids).
    Returns (A_normalized, raw_q) with strict pairing checks; raises on invalid shapes.
    """
    K, H = z.shape
    if K < 2:
        raise BackendError(f"Method B needs K>=2 pooled states, got {K}")
    j_ids = sorted(z_edit.keys())
    if not j_ids:
        raise BackendError("no edited replays available for Method B")
    for j in j_ids:
        if z_edit[j].shape != (K, H):
            raise BackendError(
                f"shape mismatch for unit {j}: {z_edit[j].shape} vs z {(K, H)}"
            )

    zn = np.stack([l2_normalize(z[k]) for k in range(K)])
    q = np.zeros((K, len(j_ids)), dtype=np.float64)
    for col, j in enumerate(j_ids):
        ze = np.stack([l2_normalize(z_edit[j][k]) for k in range(K)])
        q[:, col] = np.linalg.norm(zn - ze, axis=1)
    denom = q.sum(axis=1, keepdims=True) + 1e-12
    A_norm = q / denom
    agg = aggregate_A(A_norm, outer_nll_sum=None, mean_sample_len=None, cfg=agg_cfg)
    if outer is not None and agg.validity.status == "ok":
        agg.risk = float((agg.inner + agg_cfg.epsilon) * outer)
        agg.outer = float(outer)
    return {"A": A_norm, "agg": agg, "j_ids": j_ids}, {"q_raw": q, "z": z,
            "z_edit": {j: z_edit[j] for j in j_ids}}
