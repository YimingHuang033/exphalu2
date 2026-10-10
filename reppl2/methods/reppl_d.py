"""D: vector dispersion of input influence under meaning-preserving views."""
from __future__ import annotations

import numpy as np

from .reppl_c import response_array


def reppl_d_inner(responses, tau, min_signal=0.0):
    d = response_array(responses)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive")
    if not np.isfinite(min_signal) or min_signal < 0:
        raise ValueError("invalid min_signal")
    energy = np.square(d).sum(axis=-1).mean(axis=0)
    variance = np.square(d - d.mean(axis=0)).sum(axis=-1).mean(axis=0)
    valid = energy > min_signal
    u = variance / (energy + tau)
    if np.any(u < -1e-10) or np.any(u > 1 + 1e-10):
        raise ValueError("vector dispersion outside [0,1]")
    # Norm-only control: opposite directions of equal magnitude disappear here.
    norm_variance = np.linalg.norm(d, axis=-1).var(axis=0)
    return {"inner": float(u[valid].mean()) if valid.any() else None,
            "validity": "ok" if valid.any() else "low_signal",
            "energy": energy.tolist(), "variance": variance.tolist(), "u": u.tolist(),
            "mean_response": d.mean(axis=0).tolist(), "valid_units": valid.tolist(),
            "coverage": float(valid.mean()), "components": u[valid].tolist(),
            "norm_only_inner": float((norm_variance / (energy + tau))[valid].mean())
            if valid.any() else None}
