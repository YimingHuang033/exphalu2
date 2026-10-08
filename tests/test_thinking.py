"""Reasoning-model (thinking) support: think/answer split, chat-template kwargs
resolution, post-think answer views for the semantic methods (DESIGN 9.5.4)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from reppl2.scoring import (THINK_CLOSE, answer_token_start, default_chat_template_kwargs,
                            resolve_chat_template_kwargs, split_think, think_boundary_id)
from reppl2.pipeline import _semantic_view_invalid, sample_answer_views
from reppl2.types import Generation, SampledOutput


# ---------- split_think ----------

def test_split_think_qwen3_style_both_markers():
    text = "<think>\nreasoning here\n</think>\nFinal answer: B"
    think, ans = split_think(text)
    assert "reasoning here" in think and "B" not in think
    assert ans == "Final answer: B"


def test_split_think_qwen35_style_close_marker_only():
    # Qwen3.5 renders the opening <think> inside the prompt, so the completion
    # only contains the closing marker
    think, ans = split_think("chain of thought ...\n</think>\n\nFinal answer: 42")
    assert "chain of thought" in think and ans == "Final answer: 42"


def test_split_think_no_marker_returns_full_text_as_answer():
    think, ans = split_think("Paris")
    assert think == "" and ans == "Paris"


def test_split_think_truncated_think_has_empty_answer():
    think, ans = split_think("partial reasoning, budget exhausted ...")
    assert think == "" and ans == "partial reasoning, budget exhausted ..."


def test_split_think_uses_last_marker():
    think, ans = split_think(f"a {THINK_CLOSE} b {THINK_CLOSE} c")
    assert "b" in think and "a" in think
    assert ans.strip() == "c"


# ---------- chat-template kwargs resolution ----------

def test_default_disables_thinking_for_qwen3_names_only():
    assert default_chat_template_kwargs("/mnt/data/Qwen3-1.7B") == {"enable_thinking": False}
    assert default_chat_template_kwargs("Qwen3.5-4B") == {"enable_thinking": False}
    assert default_chat_template_kwargs("/mnt/data/Qwen2.5-1.5B-Instruct") == {}
    assert default_chat_template_kwargs("") == {}


def test_resolve_kwargs_run_level_wins():
    cfg = {"model": "qwen3_5_4b",
           "chat_template_kwargs": {"enable_thinking": True},
           "models": {"qwen3_5_4b": {"chat_template_kwargs": {"enable_thinking": False}}}}
    assert resolve_chat_template_kwargs(cfg, None) == {"enable_thinking": True}


def test_resolve_kwargs_model_entry_fallback():
    cfg = {"model": "qwen3_5_4b",
           "models": {"qwen3_5_4b": {"chat_template_kwargs": {"enable_thinking": False}}}}
    assert resolve_chat_template_kwargs(cfg, None) == {"enable_thinking": False}
    # no model entry and no run-level key -> None (render_prompt applies the
    # name-based legacy default itself)
    assert resolve_chat_template_kwargs({"model": "qwen2_5_0_5b", "models": {}}, None) is None


class _CaptureTokenizer:
    name_or_path = "/mnt/data/Qwen3-1.7B"
    chat_template = "T"

    def apply_chat_template(self, messages, add_generation_prompt, tokenize=False, **kw):
        self.captured_kwargs = dict(kw)
        return "rendered"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1, 2]}


def test_render_prompt_explicit_kwargs_override_name_default():
    tok = _CaptureTokenizer()
    from reppl2.scoring import render_prompt
    from reppl2.types import Example
    ex = Example(sample_id="s", task="t", question="q")
    render_prompt(ex, tok, {"enable_thinking": True})
    assert tok.captured_kwargs == {"enable_thinking": True}
    # without explicit kwargs the qwen3 name default kicks in
    render_prompt(ex, tok)
    assert tok.captured_kwargs == {"enable_thinking": False}


def test_sample_and_score_forwards_chat_template_kwargs():
    # regression: the run-level enable_thinking must reach render_prompt, or a
    # "thinking ON" run silently renders the non-thinking prompt
    from reppl2.scoring import sample_and_score
    from reppl2.types import Example

    class _FakeBackend:
        name = "fake"

        def sample(self, prompt_ids, params, n=1, seed=None):
            return {"token_ids": [[5]] * n, "texts": ["x"] * n,
                    "logprobs": [[-0.5]] * n, "finish_reasons": ["stop"] * n}

    tok = _CaptureTokenizer()
    tok.name_or_path = "plain-model"   # no name-based default -> would render {}
    ex = Example(sample_id="s", task="t", question="q")
    sample_and_score(_FakeBackend(), ex, tok, {"max_new_tokens": 8}, k=2,
                     chat_template_kwargs={"enable_thinking": True})
    assert tok.captured_kwargs == {"enable_thinking": True}


# ---------- post-think answer views (semantic methods) ----------

class _Tok:
    def convert_tokens_to_ids(self, token):
        return 151668 if token == THINK_CLOSE else None


def _gen(texts_and_boundaries):
    samples = []
    for i, (text, ids) in enumerate(texts_and_boundaries):
        samples.append(SampledOutput(sample_id="s", k=i + 1, token_ids=ids,
                                     text=text, token_logprobs=[-0.1] * len(ids)))
    return Generation(sample_id="s", prompt_text="p", prompt_token_ids=[1],
                      greedy=samples[0], samples=samples, sampling_config={})


B = 151668  # </think> token id


def test_answer_views_without_markers_unchanged():
    gen = _gen([("Paris", [5, 6]), ("Rome", [7, 8])])
    v = sample_answer_views(gen, _Tok(), set())
    assert v["texts"] == ["Paris", "Rome"]
    assert v["token_ids"] == [[5, 6], [7, 8]]
    assert v["n_missing_answer"] == 0
    assert _semantic_view_invalid(v, "post-think") is None


def test_answer_views_slice_after_think_marker():
    # Qwen3.5: <think> opened in the prompt; completion holds reasoning then </think>
    gen = _gen([("reason a" + THINK_CLOSE + " Final answer: B", [11, 12, B, 13, 14]),
                ("reason b" + THINK_CLOSE + " Final answer: A", [21, B, 22])])
    v = sample_answer_views(gen, _Tok(), set())
    assert v["texts"] == [" Final answer: B", " Final answer: A"]
    assert "reason a" not in v["texts"][0]
    assert v["token_ids"] == [[13, 14], [22]]
    assert v["token_logprobs"] == [[-0.1, -0.1], [-0.1]]


def test_answer_views_truncated_think_flagged():
    gen = _gen([("reason a" + THINK_CLOSE + " Paris", [1, B, 2]),
                ("still reasoning without close", [3, 4])])
    # with thinking known ON, a marker-less sample is a truncated think
    v = sample_answer_views(gen, _Tok(), set(), thinking_on=True)
    assert v["n_missing_answer"] == 1
    msg = _semantic_view_invalid(v, "post-think")
    assert msg and "truncated" in msg
    # with thinking OFF the same pool scores unchanged (legacy behaviour)
    v_off = sample_answer_views(gen, _Tok(), set())
    assert v_off["n_missing_answer"] == 0
    assert _semantic_view_invalid(v_off, "post-think") is None


def test_answer_views_text_marker_without_token_boundary_invalid():
    # marker in text but </think> not resolvable as a single token: token-sliced
    # methods must not silently mix think tokens into the answer span
    tok = _Tok()

    def _no_ids(token):
        return None

    tok.convert_tokens_to_ids = _no_ids
    gen = _gen([("r" + THINK_CLOSE + " Paris", [1, 2, 3]),
                ("r2" + THINK_CLOSE + " Rome", [4, 5])])
    v = sample_answer_views(gen, tok, set())
    assert v["token_boundary_ok"] is False
    assert _semantic_view_invalid(v, "post-think") is not None


def test_answer_token_start_uses_last_occurrence():
    assert answer_token_start([1, B, 2, B, 3], B) == 4
    assert answer_token_start([1, 2, 3], B) is None
    assert answer_token_start([1, 2, 3], None) is None


def test_think_boundary_id_none_for_missing_token():
    class _T:
        def convert_tokens_to_ids(self, token):
            return None

    assert think_boundary_id(_T()) is None


# ---------- think-mode sampling budget sanity (config contract) ----------

def test_reasoning_config_pins_thinking_and_budget():
    import yaml
    cfg = yaml.safe_load(Path(__file__).resolve().parent.parent
                         .joinpath("config/generation_eval_reasoning.yaml").read_text())
    assert cfg["chat_template_kwargs"] == {"enable_thinking": True}
    assert cfg["sampling"]["max_new_tokens"] >= 1024
    assert cfg["datasets"]["supergpqa"]["sampling"]["max_new_tokens"] >= 1536
    assert cfg["datasets"]["popqa"]["sampling"]["max_new_tokens"] >= 1024
    assert cfg["judge"]["provider"] == "gpt_oss_20b"
