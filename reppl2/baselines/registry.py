from __future__ import annotations

from .probability import (outer_perplexity_risk, lnpe_risk, eigenscore_last_risk,
                          output_length_risk, semantic_entropy_risk)

METHOD_REGISTRY = {
    "reppl-a": {"family": "reppl", "direction": "higher=hallucination", "status": "implemented"},
    "reppl-b": {"family": "reppl", "direction": "higher=hallucination", "status": "implemented"},
    "outer-perplexity": {"family": "probability", "direction": "higher=hallucination", "status": "implemented"},
    "lnpe": {"family": "probability", "direction": "higher=hallucination", "status": "implemented"},
    "eigenscore-last": {"family": "probability", "direction": "higher=hallucination",
                        "status": "implemented", "note": "last-layer variant; original specifies middle layer"},
    "semantic-entropy": {"family": "semantic", "direction": "higher=hallucination", "status": "implemented"},
    "semantic-entropy-lexical": {"family": "semantic", "direction": "higher=hallucination",
                                 "status": "implemented", "note": "lexical fallback grouping, not NLI-based"},
    "length": {"family": "control", "direction": "covariate", "status": "implemented"},
    "sese": {"family": "semantic", "direction": "higher=hallucination", "status": "planned"},
    "had": {"family": "supervised", "direction": "higher=hallucination", "status": "planned"},
    "d-score": {"family": "internal-state", "direction": "higher=hallucination", "status": "planned"},
    "rauq": {"family": "internal-state", "direction": "higher=hallucination", "status": "planned",
             "note": "requires attention; blocked under native-only constraint"},
    "lafact": {"family": "internal-state", "direction": "higher=hallucination", "status": "planned"},
    "laab": {"family": "fusion", "direction": "higher=hallucination", "status": "planned"},
    "semantic-energy": {"family": "semantic", "direction": "higher=hallucination", "status": "planned",
                        "note": "needs unnormalized logits; not recoverable from top-k logprobs"},
}
