"""Fixed-array validation of input-unit -> prompt token span alignment and the
RePPL-A context-state contract (regression for the misalignment bug found on
Qwen3.5-4B: source-relative char spans were mapped directly onto the rendered
prompt, and prompt-side states were missing from backend replays)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from reppl2.backends.base import BackendError
from reppl2.methods.reppl_a import reppl_a_inner
from reppl2.methods.common import AggregatorConfig
from reppl2.segmentation import unit_token_spans, _prompt_span


class MockTokenizer:
    """Whitespace tokenizer: first token is a template symbol with (0,0) offsets,
    mirroring real chat-template special tokens; the rest carry true offsets."""

    def __init__(self, prompt_text: str):
        self.prompt_text = prompt_text
        prefix = "<|t|>"
        assert prompt_text.startswith(prefix)
        body = prompt_text[len(prefix):]
        self.offsets = [(0, 0)]
        self.words = [prefix]
        pos = len(prefix)
        for w in body.split(" "):
            if w == "":
                pos += 1
                continue
            self.offsets.append((pos, pos + len(w)))
            self.words.append(w)
            pos += len(w) + 1

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        assert text == self.prompt_text
        out = {"input_ids": list(range(len(self.words)))}
        if return_offsets_mapping:
            out["offset_mapping"] = self.offsets
        return out

    def convert_ids_to_tokens(self, ids):
        return [self.words[i] for i in ids]


PROMPT = "<|t|>Question: Who wrote Hamlet? When was it first performed? Answer briefly."
QUESTIONS = ["Who wrote Hamlet?", "When was it first performed?"]
Q_STARTS = [PROMPT.index(q) for q in QUESTIONS]


def _units():
    return [
        {"unit_id": 0, "text": "You are terse.", "char_start": 0, "char_end": 14,
         "source": "system", "is_factual": False},
        {"unit_id": 1, "text": QUESTIONS[0], "char_start": 0, "char_end": len(QUESTIONS[0]),
         "source": "question_sentence", "is_factual": True},
        {"unit_id": 2, "text": QUESTIONS[1], "char_start": len(QUESTIONS[0]) + 1,
         "char_end": len(QUESTIONS[0]) + 1 + len(QUESTIONS[1]),
         "source": "question_sentence", "is_factual": True},
    ]


def test_prompt_span_prefers_text_search():
    u = _units()[1]
    cs, ce, src = _prompt_span(u, PROMPT)
    assert (cs, ce) == (Q_STARTS[0], Q_STARTS[0] + len(QUESTIONS[0]))
    assert src == "text_search"


def test_prompt_span_fallback_is_labelled():
    u = {"text": "not in prompt", "char_start": 3, "char_end": 16}
    cs, ce, src = _prompt_span(u, PROMPT)
    assert (cs, ce) == (3, 16)
    assert src == "source_relative_unverified"


def test_token_spans_land_on_question_not_template():
    tok = MockTokenizer(PROMPT)
    spans = unit_token_spans(_units(), PROMPT, list(range(len(tok.words))), tok)
    # factual units, in order
    for u in spans[1:]:
        assert u["token_start"] >= 1  # never the (0,0) template token
        covered = " ".join(tok.words[u["token_start"]:u["token_end"]])
        assert u["text"].split()[0] in covered
    for u, q_start in zip(spans[1:], Q_STARTS):
        assert u["char_start_in_prompt"] == q_start
    # source-relative fields stay untouched (reppl-b editing contract)
    assert spans[1]["char_start"] == 0
    assert spans[1]["char_end"] == len(QUESTIONS[0])


def test_reppl_a_pools_context_side_states():
    """u[j] must come from prompt-side states: with output-only states (the old
    buggy contract) L_ctx < token_end and every unit is rejected."""
    tok = MockTokenizer(PROMPT)
    n_prompt = len(tok.words)
    spans = unit_token_spans(_units(), PROMPT, list(range(n_prompt)), tok)
    good = [u for u in spans if u["is_factual"]]
    assert len(good) == 2
    assert all(0 <= u["token_start"] < u["token_end"] <= n_prompt for u in good)

    rng = np.random.default_rng(3)
    H = 8
    ctx = rng.normal(size=(n_prompt, H)) * 0.01
    # orthogonal anchors inside the two question spans
    ts0, te0 = good[0]["token_start"], good[0]["token_end"]
    ts1, te1 = good[1]["token_start"], good[1]["token_end"]
    ctx[ts0] = np.r_[1.0, np.zeros(H - 1)]
    ctx[ts1] = np.r_[0.0, 1.0, np.zeros(H - 2)]
    # per-sample output states near different anchors -> cross-sample variance > 0
    samples = [
        np.tile(np.r_[0.9, 0.1, np.zeros(H - 2)], (3, 1)),
        np.tile(np.r_[0.1, 0.9, np.zeros(H - 2)], (3, 1)),
    ]
    agg, info = reppl_a_inner(
        None, {"context": ctx, "samples": samples}, spans,
        association_temperature=1.0, agg_cfg=AggregatorConfig(), outer=2.0,
        require_valid_units=False)
    assert agg.inner > 0.0
    assert info["A"].shape == (2, 2)
    assert not np.allclose(info["A"][0], info["A"][1])


def test_reppl_a_rejects_when_no_unit_aligns():
    spans = [{"unit_id": 0, "text": "?", "char_start": 0, "char_end": 1,
              "source": "question_sentence", "is_factual": True,
              "token_start": -1, "token_end": -1}]
    ctx = np.ones((5, 4))
    with pytest.raises(BackendError):
        reppl_a_inner(None, {"context": ctx, "samples": [np.ones((2, 4))] * 2},
                      spans, 1.0, AggregatorConfig(), outer=1.0)
