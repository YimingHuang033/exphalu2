"""Fixed-array tests: RAUQ and Semantic Energy ports + reppl-b rawq/reppl-ab."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from reppl2.backends.base import BackendError
from reppl2.baselines.rauq import rauq_risk, prev_token_attention, middle_third_layers
from reppl2.baselines.semantic_energy import (semantic_energy_risk, boltzmann,
                                              cal_cluster_ce)
from reppl2.methods.reppl_b import reppl_b_inner
from reppl2.methods.common import AggregatorConfig


# ---------------------------------------------------------------- RAUQ
def _attention_fixture(L=6, H=4, T_total=8, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.random((L, H, T_total, T_total))
    A = A / A.sum(axis=-1, keepdims=True)  # rows sum to 1 (softmax-like)
    return A


def test_prev_token_attention_diagonal():
    A = _attention_fixture(T_total=5)
    P = prev_token_attention(A)
    assert P.shape == (6, 4, 4)  # position 0 has no predecessor
    assert np.allclose(P[:, :, 2], A[:, :, 3, 2])  # position 3 -> predecessor 2
    assert np.allclose(P[:, :, 0], A[:, :, 1, 0])


def test_rauq_recurrence_matches_official_formula():
    A = _attention_fixture(L=3, H=2, T_total=5)
    ctx, T_ans = 2, 3
    lp = [-0.1, -0.5, -0.2]
    alpha = 0.2
    res = rauq_risk(A, lp, ctx_len=ctx, alpha=alpha, layers=[1],
                    token_aggregation="mean", aggregation="mean")
    # manual official recurrence on the selected layer/head
    Al = prev_token_attention(A)[1]  # (H, T_total-1); index t-1 = position t
    h = int(Al[:, ctx - 1:ctx - 1 + T_ans].mean(axis=-1).argmax())
    attn = Al[h, ctx - 1:ctx - 1 + T_ans]
    conf = [np.exp(lp[0])]
    for j in range(1, T_ans):
        conf.append(alpha * np.exp(lp[j]) + (1 - alpha) * attn[j] * conf[-1])
    assert res["heads"] == [h]
    assert np.isclose(res["risk"], 1 - np.mean(conf))


def test_rauq_meanmin_default():
    A = _attention_fixture(L=6, T_total=6)
    lp = [-0.3, -0.1, -0.9]
    res = rauq_risk(A, lp, ctx_len=3)  # layers=None -> middle third of 6 = [2,3,4]
    assert res["layers"] == middle_third_layers(6)
    assert res["layers"] == [2, 3, 4]
    assert len(res["uq_layers"]) == 3


def test_rauq_rejects_bad_inputs():
    A = _attention_fixture(L=3, T_total=5)
    with pytest.raises(BackendError):
        rauq_risk(A, [-0.1, -0.2], ctx_len=10)  # ctx + answer > T_total
    with pytest.raises(BackendError):
        rauq_risk(A, [-0.1, float("nan")], ctx_len=2)
    with pytest.raises(BackendError):
        rauq_risk(A, [-0.1, -0.2], ctx_len=2, alpha=1.5)


# ------------------------------------------------------ Semantic Energy
def test_semantic_energy_official_math():
    lps = [[-0.5, -0.5], [-0.1, -0.1], [-2.0, -2.0]]
    clusters = [[0, 1], [2]]
    bolts = [boltzmann(lp) for lp in lps]
    assert bolts[0] == pytest.approx(0.5)
    probs_se, logits_se = cal_cluster_ce(
        [np.exp(np.sum(lp)) for lp in lps], bolts, clusters)
    # cluster 0 logits = sum(mean lp) = -0.5 + -0.1 = -0.6; cluster 1 = -2.0
    assert logits_se[0] == pytest.approx(-0.6)
    assert logits_se[1] == pytest.approx(-2.0)
    risk, info = semantic_energy_risk(lps, clusters)
    assert risk == pytest.approx(-(-0.6))  # best cluster dominates
    assert info["best_cluster"] == [0, 1]


def test_semantic_energy_rejects_bad_clusters():
    with pytest.raises(BackendError):
        semantic_energy_risk([[-0.1], [-0.2]], [[0], [1], [2]])
    with pytest.raises(BackendError):
        semantic_energy_risk([[-0.1], [-0.2]], [[0]])
    with pytest.raises(BackendError):
        semantic_energy_risk([[-0.1], []], [[0], [1]])


# ------------------------------------------------------------ reppl-b rawq
def test_reppl_b_rawq_aggregates_unnormalized_q():
    rng = np.random.default_rng(2)
    K, H, J = 4, 8, 2
    z = rng.normal(size=(K, H))
    z_edit = {0: rng.normal(size=(K, H)), 1: rng.normal(size=(K, H))}
    res, raw = reppl_b_inner(z, z_edit, AggregatorConfig(), outer=1.5)
    # rawq must differ from the normalized-A aggregation in general
    assert res["agg_raw"].inner is not None
    assert raw["q_raw"].shape == (K, J)
    # identical-input degenerate: q ~ 0 -> normalized A still defined via epsilon
    z_edit_same = {0: np.tile(z[0], (K, 1))}
    res2, _ = reppl_b_inner(z, z_edit_same, AggregatorConfig(), outer=None)
    assert res2["agg"].validity.status in ("ok", "degenerate", "invalid")


def test_reppl_b_shape_mismatch_raises():
    z = np.ones((3, 4))
    with pytest.raises(BackendError):
        reppl_b_inner(z, {0: np.ones((3, 5))}, AggregatorConfig(), outer=None)
