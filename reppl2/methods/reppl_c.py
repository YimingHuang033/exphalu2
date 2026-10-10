"""C: signed input influence along a fixed target/competitor semantic axis."""
from __future__ import annotations

import numpy as np


def unit_vector(x, floor=1e-12):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("expected a finite vector")
    norm = float(np.linalg.norm(x))
    if norm <= floor:
        raise ValueError("zero/indistinguishable semantic direction")
    return x / norm


def response_array(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] < 2 or min(x.shape[1:]) < 1:
        raise ValueError("responses must have shape (K>=2,J>=1,H>=1)")
    if not np.isfinite(x).all():
        raise ValueError("non-finite propagation responses")
    return x


def reppl_c_inner(responses, direction, tau, min_signal=0.0):
    d = response_array(responses)
    e = unit_vector(direction)
    if e.size != d.shape[-1] or not np.isfinite(tau) or tau <= 0:
        raise ValueError("direction dimension or tau invalid")
    if not np.isfinite(min_signal) or min_signal < 0:
        raise ValueError("invalid min_signal")
    a = d @ e
    scale = np.abs(a).mean(axis=0)
    std = a.std(axis=0, ddof=0)
    valid = scale > min_signal
    u = std / (scale + tau)
    # Never turn an absent measurement into zero risk. Partial coverage is explicit.
    inner = float(np.log1p(u[valid] ** 2).mean()) if valid.any() else None
    return {"inner": inner, "validity": "ok" if valid.any() else "low_signal",
            "a": a.tolist(), "mu": a.mean(axis=0).tolist(),
            "scale": scale.tolist(), "std": std.tolist(), "u": u.tolist(),
            "valid_units": valid.tolist(), "coverage": float(valid.mean()),
            "components": np.log1p(u[valid] ** 2).tolist()}
