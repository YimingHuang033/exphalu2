from __future__ import annotations

from .probability import (outer_perplexity_risk, lnpe_risk, eigenscore_last_risk,
                          output_length_risk, semantic_entropy_risk)
from .dscore import dscore_last_risk
from .sese import sese_risk

METHOD_REGISTRY = {
    "reppl-a": {"family": "reppl", "direction": "higher=hallucination", "status": "implemented"},
    "reppl-b": {"family": "reppl", "direction": "higher=hallucination", "status": "implemented"},
    "reppl-b-rawq": {"family": "reppl", "direction": "higher=hallucination", "status": "implemented",
                     "note": "DESIGN 5.1 ablation: shared aggregator on unnormalized q"},
    "reppl-ab": {"family": "reppl", "direction": "higher=hallucination", "status": "implemented",
                 "note": "DESIGN 5.3 cost-controlled variant: A nominates top-J units for B editing; "
                         "reported separately from full RepplB"},
    "outer-perplexity": {"family": "probability", "direction": "higher=hallucination", "status": "implemented"},
    "lnpe": {"family": "probability", "direction": "higher=hallucination", "status": "implemented"},
    "eigenscore-last": {"family": "probability", "direction": "higher=hallucination",
                        "status": "implemented", "note": "last-layer variant; original specifies middle layer"},
    "semantic-entropy": {"family": "semantic", "direction": "higher=hallucination", "status": "implemented"},
    "semantic-entropy-lexical": {"family": "semantic", "direction": "higher=hallucination",
                                 "status": "implemented", "note": "lexical fallback grouping, not NLI-based"},
    "length": {"family": "control", "direction": "covariate", "status": "implemented"},
    "d-score-last": {"family": "internal-state", "direction": "higher=hallucination",
                     "status": "implemented", "runner": dscore_last_risk,
                     "note": "adapted variant; original D-Score selects the best layer, "
                             "engine-native path exposes last-layer states only"},
    "d-score": {"family": "internal-state", "direction": "higher=hallucination", "status": "planned",
                "note": "original best-layer configuration; needs middle-layer state access"},
    "sese": {"family": "semantic", "direction": "higher=hallucination", "status": "implemented",
             "runner": sese_risk,
             "note": "official port (SELGroup/SeSE @8d4c6c5); GPT-4o answer enhancement "
                     "skipped (no key) and recorded, not substituted"},
    "rauq": {"family": "internal-state", "direction": "higher=hallucination", "status": "implemented",
             "note": "official math port (mbzuai-nlp/rauq-hallucination-detection); needs attention "
                     "-> runs on the transformers reference env only, vLLM path reports capability error"},
    "semantic-energy": {"family": "semantic", "direction": "higher=hallucination", "status": "implemented",
                        "note": "official notebook math (MaHAAA/SemanticEnergy); per-token scalar logit "
                                "signal implemented as token logprob (source ambiguous in temp repo, recorded)"},
    "had": {"family": "supervised", "direction": "higher=hallucination",
            "status": "blocked:weights-not-released",
            "note": "official repo (pku0xff/HAD) releases data+prompts only; no model weights on "
                    "GitHub or HF mirror as of 2026-10-05"},
    "lafact": {"family": "internal-state", "direction": "higher=hallucination", "status": "planned"},
    "laab": {"family": "fusion", "direction": "higher=hallucination", "status": "planned"},
}

# These methods use the frozen-generation CDE runner, not the legacy A/B CLI.
for _name in ("reppl-c", "reppl-d", "reppl-e", "reppl-c-inner", "reppl-d-inner",
              "reppl-e-inner", "reppl-c-gentle", "reppl-d-gentle", "reppl-e-gentle"):
    METHOD_REGISTRY[_name] = {
        "family": "reppl", "direction": "higher=hallucination", "status": "implemented",
        "entrypoint": "reppl2.cde",
        "note": "native propagation experiment; GPU/performance validation pending; E is supervised OOF",
    }
for _name in ("c-random-direction-inner", "d-norm-only-inner", "d-output-variance-inner"):
    METHOD_REGISTRY[_name] = {
        "family": "control", "direction": "higher=hallucination", "status": "implemented",
        "entrypoint": "reppl2.cde",
    }
