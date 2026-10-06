"""D-Score baseline (Raimondi et al., arXiv:2607.24586v1).

D_tau(x) is the relative numerical rank of the hidden activation matrix
H = [h_1..h_T] in R^{d x T} (uncentered, columns = token states):

    D_tau(x) = #{ i : sigma_1(H) / sigma_i(H) <= tau }

computed exactly via the Gram matrix G = H^T H (eigenvalues lambda_i = sigma_i^2,
count lambda_i >= lambda_1 / tau^2). Direction contract: larger D = more spectral
spread = higher hallucination risk (paper Hypothesis 1).

Variant note (DESIGN.md section 8.1): the original paper selects the
best-performing layer, which is generally NOT the last layer. Our engine-native
path only exposes last-layer states (vLLM token_embed), so this implementation
is explicitly named `d-score-last` and must be reported as an adaptation, not as
the original best-layer configuration.
"""
from __future__ import annotations

import numpy as np

from ..backends.base import BackendError

VERSION = "d-score-last-v1"


def dscore_last_risk(states: np.ndarray, tau: float = 10.0) -> float:
    """states: (T, d) last-layer states of the answer tokens from a single replay.

    Returns float(D_tau). Raises BackendError on degenerate inputs instead of
    silently returning 0.
    """
    X = np.asarray(states, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1:
        raise BackendError(f"d-score-last: states must be (T, d), got {X.shape}")
    if not np.isfinite(X).all():
        raise BackendError("d-score-last: non-finite states")
    if tau <= 1.0:
        raise BackendError("d-score-last: tau must be > 1 (relative tolerance)")

    # Gram matrix of the uncentered activation matrix H = X^T  ->  G = X X^T (T x T)
    G = X @ X.T
    lam = np.linalg.eigvalsh(G)[::-1]  # descending
    lam = np.clip(lam, 0.0, None)
    lam_max = float(lam[0])
    if lam_max <= 0.0:
        raise BackendError("d-score-last: all-zero activation matrix")

    floor = lam_max / (tau * tau)
    count = int((lam >= floor).sum())
    return float(count)
