import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from reppl2.baselines.probability import (outer_perplexity_risk, lnpe_risk,
                                          eigenscore_last_risk, output_length_risk,
                                          semantic_entropy_risk)
from reppl2.backends.base import BackendError


def test_outer_perplexity_basic():
    lps = [-0.1, -0.2, -0.3]
    assert np.isclose(outer_perplexity_risk(lps), 0.2)


def test_outer_perplexity_eos_excluded():
    lps = [-0.1, -0.2, -5.0]
    ids = [7, 8, 151645]
    assert np.isclose(outer_perplexity_risk(lps, eos_token_ids={151645}, token_ids=ids), 0.15)


def test_outer_perplexity_rejects_nan():
    with pytest.raises(BackendError):
        outer_perplexity_risk([-0.1, float("nan")])


def test_lnpe_mean_over_samples():
    a = [-0.1, -0.3]
    b = [-0.5, -0.5, -0.5]
    v = lnpe_risk([a, b], [[1, 2], [3, 4, 5]])
    assert np.isclose(v, (0.2 + 0.5) / 2)


def test_eigenscore_direction_collapse_is_risky():
    rng = np.random.default_rng(0)
    diverse = [rng.normal(size=(4, 8)) for _ in range(5)]
    collapsed = [np.tile(np.arange(8.0), (4, 1)) + 1e-6 * rng.normal(size=(4, 8)) for _ in range(5)]
    r_div = eigenscore_last_risk(diverse)
    r_col = eigenscore_last_risk(collapsed)
    assert r_col > r_div


def test_eigenscore_needs_two():
    with pytest.raises(BackendError):
        eigenscore_last_risk([np.ones((3, 4))])


def test_length_risk():
    assert np.isclose(output_length_risk([2, 4, 6]), 4.0)


def test_semantic_entropy_lexical_identical_cluster_low():
    texts = ["Paris", "Paris", "Paris"]
    lps = [[-0.1], [-0.1], [-0.1]]
    pe, info = semantic_entropy_risk(texts, lps)
    assert info["variant"] == "semantic-entropy-lexical"
    assert pe < 0.01


def test_semantic_entropy_lexical_diverse_high():
    texts = ["Paris", "London", "Berlin"]
    lps = [[-0.1], [-0.1], [-0.1]]
    pe, info = semantic_entropy_risk(texts, lps)
    assert info["n_clusters"] == 3
    assert pe > 1.0


def test_semantic_entropy_needs_two_samples():
    with pytest.raises(BackendError):
        semantic_entropy_risk(["a"], [[-0.1]])
