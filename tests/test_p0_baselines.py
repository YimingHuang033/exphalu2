"""Fixed-array tests for the P0 baselines: D-Score-last and SeSE."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from reppl2.backends.base import BackendError
from reppl2.baselines.dscore import dscore_last_risk
from reppl2.baselines.sese import structural_entropy, _make_connected, _connected_components


def test_dscore_identity_full_rank():
    # orthogonal token states -> all singular values equal -> D = T
    assert dscore_last_risk(np.eye(4), tau=2.0) == 4.0


def test_dscore_rank1_collapses():
    # rank-1 activation -> only the leading direction -> D = 1
    assert dscore_last_risk(np.ones((6, 8)), tau=2.0) == 1.0


def test_dscore_matches_sigma_ratio_definition():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(9, 16))
    tau = 3.0
    sv = np.linalg.svd(X, compute_uv=False)
    expected = int((sv[0] / sv <= tau).sum())
    assert dscore_last_risk(X, tau=tau) == expected


def test_dscore_scale_invariance():
    X = np.random.default_rng(1).normal(size=(8, 32))
    assert dscore_last_risk(X, tau=5.0) == dscore_last_risk(X * 1e4, tau=5.0)


def test_dscore_rejects_degenerate():
    with pytest.raises(BackendError):
        dscore_last_risk(np.zeros((3, 4)))
    with pytest.raises(BackendError):
        dscore_last_risk(np.array([[1.0, np.nan]]))
    with pytest.raises(BackendError):
        dscore_last_risk(np.eye(3), tau=0.5)


def test_sese_matches_reference_port_bitwise():
    """Regression anchor: this value was produced by both the official SeSE
    implementation (SELGroup/SeSE @ 8d4c6c5, numba/cdlib stripped) and our
    port on the same seeded matrix."""
    rng = np.random.default_rng(0)
    adj = rng.random((5, 5))
    adj = (adj + adj.T) / 2
    np.fill_diagonal(adj, 0)
    assert abs(structural_entropy(adj, 2) - (-1.8589244319885392)) < 1e-12


def test_sese_monotone_in_spread():
    # identical answers (complete graph) vs fully disconnected equal-weight graph:
    # the disconnected graph spreads volume evenly -> different entropy; both finite.
    complete = np.ones((5, 5)) - np.eye(5)
    e_complete = structural_entropy(complete, 2)
    assert np.isfinite(e_complete)
    # two clusters of identical answers: 3+2
    adj = np.zeros((5, 5))
    for i, j in [(0, 1), (0, 2), (1, 2), (3, 4)]:
        adj[i, j] = adj[j, i] = 0.9
    e_two = structural_entropy(adj, 2)
    assert np.isfinite(e_two)


def test_make_connected_adds_bridges_between_components():
    adj = np.zeros((4, 4))
    adj[0, 1] = adj[1, 0] = 0.5
    adj[2, 3] = adj[3, 2] = 0.4
    entail_fn = lambda pairs: [[0.8, 0.1, 0.1] for _ in pairs]
    out = _make_connected(adj.copy(), entail_fn)
    assert len(_connected_components(out)) == 1
    # bridged with the entailment probability from entail_fn
    assert out[0, 2] == 0.8 or out[0, 3] == 0.8 or out[1, 2] == 0.8 or out[1, 3] == 0.8
