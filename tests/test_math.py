"""Fixed-array validation of the shared math (DESIGN.md M1 acceptance)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from reppl2.methods.common import (aggregate_A, compute_cv, inv_func, row_softmax,
                                   cosine_matrix, l2_normalize, AggregatorConfig, MathError)


def test_cv_matches_definition():
    x = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
    cv = compute_cv(x.T)
    assert np.allclose(cv, np.array([np.sqrt(2.0 / 3.0) / 2.0, 0.0]), atol=1e-12)


def test_cv_ddof0_matches_std():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(7, 4)) * 3 + 5
    cv = compute_cv(x)
    manual = x.std(axis=0, ddof=0) / (x.mean(axis=0) + 1e-12)
    assert np.allclose(cv, manual)


def test_cv_needs_two_samples():
    with pytest.raises(MathError):
        compute_cv(np.ones((1, 3)))


def test_inv_func_values():
    assert np.allclose(inv_func(np.array([0.0]), a=2.0), [1.0])
    assert np.allclose(inv_func(np.array([1.0]), a=1.0), [0.5])


def test_row_softmax_axis_is_input_units():
    x = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    w = row_softmax(x, axis=-1, temperature=1.0)
    assert w.shape == (2, 3)
    assert np.allclose(w.sum(axis=-1), 1.0)
    assert np.allclose(w[0], w[1])


def test_row_softmax_temperature():
    x = np.array([[1.0, 3.0]])
    w_hot = row_softmax(x, axis=-1, temperature=10.0)
    w_cold = row_softmax(x, axis=-1, temperature=0.1)
    assert w_hot.max() < w_cold.max()


def test_row_softmax_invariant_to_output_token_permutation():
    rng = np.random.default_rng(1)
    sims = rng.normal(size=(5, 6))
    w1 = row_softmax(sims, axis=-1).mean(axis=0)
    perm = rng.permutation(5)
    w2 = row_softmax(sims[perm], axis=-1).mean(axis=0)
    assert np.allclose(w1, w2)


def test_cosine_matrix_known_values():
    U = np.array([[1.0, 0.0], [0.0, 2.0]])
    V = np.array([[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    C = cosine_matrix(V, U)
    assert np.allclose(C, np.array([[1.0, 0.0], [0.0, 1.0], [np.sqrt(0.5), np.sqrt(0.5)]]))


def test_cosine_rejects_zero_vector():
    with pytest.raises(MathError):
        cosine_matrix(np.zeros((1, 3)), np.ones((1, 3)))


def test_l2_normalize():
    v = l2_normalize(np.array([3.0, 4.0]))
    assert np.allclose(v, [0.6, 0.8])
    with pytest.raises(MathError):
        l2_normalize(np.zeros(2))


def test_aggregate_known_small_matrix():
    A = np.array([
        [0.5, 0.5],
        [0.5, 0.1],
    ])
    cfg = AggregatorConfig(tau=0.5, alpha=2.0, epsilon=0.001)
    agg = aggregate_A(A, outer_nll_sum=2.0, mean_sample_len=2.0, cfg=cfg)
    mu = A.mean(axis=0)
    sigma = A.std(axis=0, ddof=0)
    r = sigma / (mu + 0.5)
    inner = np.mean(np.log1p(r ** 2))
    assert np.allclose(agg.mu, mu)
    assert np.allclose(agg.sigma, sigma)
    assert np.allclose(agg.r, r)
    assert np.isclose(agg.inner, inner)
    assert np.isclose(agg.outer, -2.0 / 2.0)
    assert np.isclose(agg.risk, (inner + 0.001) * (-1.0))


def test_aggregate_k2_invalid():
    A = np.array([[0.1, 0.2]])
    agg = aggregate_A(A, 1.0, 1.0, AggregatorConfig())
    assert agg.validity.status == "invalid"
    assert agg.risk is None


def test_aggregate_allzero_invalid():
    A = np.zeros((3, 4))
    agg = aggregate_A(A, 1.0, 1.0, AggregatorConfig())
    assert agg.validity.status == "invalid"


def test_aggregate_nonfinite_invalid():
    A = np.array([[0.1, np.nan], [0.2, 0.3], [0.3, 0.1]])
    agg = aggregate_A(A, 1.0, 1.0, AggregatorConfig())
    assert agg.validity.status == "invalid"


def test_aggregate_identical_samples_low_variance_is_kept():
    A = np.ones((4, 3)) * 0.25
    agg = aggregate_A(A, 1.0, 1.0, AggregatorConfig())
    assert agg.validity.status == "ok"
    assert np.allclose(agg.r, 0.0)
    assert np.isclose(agg.inner, 0.0)


def test_aggregate_unit_order_invariance():
    rng = np.random.default_rng(2)
    A = rng.uniform(0, 1, size=(4, 6))
    cfg = AggregatorConfig()
    a1 = aggregate_A(A, 2.0, 2.0, cfg)
    perm = rng.permutation(6)
    a2 = aggregate_A(A[:, perm], 2.0, 2.0, cfg)
    assert np.isclose(a1.inner, a2.inner)
    assert np.allclose(a1.mu[perm], a2.mu)


def test_score_direction_contract():
    high_var = np.array([
        [0.9, 0.1, 0.5], [0.1, 0.9, 0.5], [0.5, 0.1, 0.9], [0.1, 0.5, 0.1],
    ])
    low_var = np.ones((4, 3)) * 0.333333
    cfg = AggregatorConfig()
    agg_hi = aggregate_A(high_var, 1.0, 1.0, cfg)
    agg_lo = aggregate_A(low_var, 1.0, 1.0, cfg)
    assert agg_hi.inner > agg_lo.inner


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
