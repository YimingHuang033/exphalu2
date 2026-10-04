"""Shared aggregation math per DESIGN.md §3.2. All fixed-array validated in tests/test_math.py."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class MathError(ValueError):
    pass


@dataclass
class AggregatorConfig:
    tau: float = 0.01
    alpha: float = 2.0
    epsilon: float = 1e-3
    version: str = "agg-v1-ddof0"

    def __post_init__(self):
        if self.tau <= 0 or self.alpha <= 0 or self.epsilon <= 0:
            raise MathError("tau/alpha/epsilon must be positive")


@dataclass
class Validity:
    status: str
    reason: str = ""


@dataclass
class SharedAggregate:
    inner: float
    outer: Optional[float]
    risk: Optional[float]
    mu: np.ndarray
    sigma: np.ndarray
    r: np.ndarray
    p_hat: np.ndarray
    n_units: int
    n_samples: int
    unique_outputs: int
    validity: Validity
    notes: list = field(default_factory=list)


def compute_cv(x: np.ndarray, axis: int = 0, ddof: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[axis] < 2:
        raise MathError(f"CV needs >=2 samples along axis {axis}, got {x.shape[axis]}")
    mean = x.mean(axis=axis)
    std = x.std(axis=axis, ddof=ddof)
    eps = 1e-12
    return std / (mean + eps)


def inv_func(x: np.ndarray, a: float = 2.0) -> np.ndarray:
    return 1.0 / (1.0 + np.asarray(x, dtype=np.float64) ** a)


def aggregate_A(A: np.ndarray, outer_nll_sum: Optional[float], mean_sample_len: Optional[float],
                cfg: AggregatorConfig) -> SharedAggregate:
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise MathError(f"A must be 2-D (K,J), got shape {A.shape}")
    K, J = A.shape
    if K < 2:
        return SharedAggregate(
            inner=float("nan"), outer=None, risk=None,
            mu=A.mean(axis=0) if J else np.zeros(0),
            sigma=np.zeros(J), r=np.zeros(J), p_hat=np.zeros(J),
            n_units=J, n_samples=K, unique_outputs=K,
            validity=Validity("invalid", f"K={K} < 2; cross-sampling variation undefined"),
        )
    if not np.isfinite(A).all():
        bad = int((~np.isfinite(A)).sum())
        return SharedAggregate(
            inner=float("nan"), outer=None, risk=None,
            mu=np.zeros(J), sigma=np.zeros(J), r=np.zeros(J), p_hat=np.zeros(J),
            n_units=J, n_samples=K, unique_outputs=K,
            validity=Validity("invalid", f"A has {bad} non-finite entries"),
        )
    if np.abs(A).sum() == 0.0:
        return SharedAggregate(
            inner=float("nan"), outer=None, risk=None,
            mu=np.zeros(J), sigma=np.zeros(J), r=np.zeros(J), p_hat=np.zeros(J),
            n_units=J, n_samples=K, unique_outputs=K,
            validity=Validity("invalid", "all-zero action matrix"),
        )
    mu = A.mean(axis=0)
    sigma = A.std(axis=0, ddof=0)
    r = sigma / (mu + cfg.tau)
    p_hat = 1.0 / (1.0 + r ** cfg.alpha)
    inner = float(np.mean(np.log1p(r ** cfg.alpha)))

    outer = None
    risk = None
    if outer_nll_sum is not None and mean_sample_len is not None and mean_sample_len > 0:
        outer = float(-outer_nll_sum / mean_sample_len)
        risk = float((inner + cfg.epsilon) * outer)
    notes = []
    if K >= 2:
        notes.append(f"unique_outputs={K}")
    return SharedAggregate(
        inner=inner, outer=outer, risk=risk,
        mu=mu, sigma=sigma, r=r, p_hat=p_hat,
        n_units=J, n_samples=K, unique_outputs=K,
        validity=Validity("ok"), notes=notes,
    )


def row_softmax(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if temperature <= 0:
        raise MathError("temperature must be positive")
    z = x / temperature
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def cosine_matrix(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    un = np.linalg.norm(U, axis=-1, keepdims=True)
    vn = np.linalg.norm(V, axis=-1, keepdims=True)
    if np.any(un == 0) or np.any(vn == 0):
        raise MathError("zero-norm vector in cosine similarity")
    return (U / un) @ (V / vn).T


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x)
    if n == 0:
        raise MathError("zero vector cannot be l2-normalized")
    return x / n
