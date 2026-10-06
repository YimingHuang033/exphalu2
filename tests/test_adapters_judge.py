import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from reppl2.config_loader import ConfigError
from reppl2.data.adapters import (get_dataset_iterator, gsm8k_gold, mmlu_gold_letter,
                                  normalize_gpqa_item, _render_mcq_question)
from reppl2.judges.local import (JUDGE_PROMPT_VERSION, build_judge_messages,
                                 judge_chat_template_kwargs, judge_prompt_final_prefix)
from reppl2.judges.systemone import (SYSTEMONE_PROMPT_VERSION, SystemOneJudge,
                                     build_factuality_request)


# ---------- judge prompt v2 (reference-aware) ----------

def test_prompt_v2_includes_reference_block():
    msgs = build_judge_messages("What is 7*8?", "Final answer: 54", "",
                                gold_answers=["56"])
    user = msgs[1]["content"]
    assert "Reference (ground truth): 56" in user
    assert "Answer: Final answer: 54" in user


def test_prompt_v2_no_reference_block_without_gold():
    msgs = build_judge_messages("Q?", "A", "")
    assert "Reference (ground truth)" not in msgs[1]["content"]


def test_prompt_v2_multiple_gold_joined():
    msgs = build_judge_messages("Q?", "A", "", gold_answers=["Paris", "City of Light"])
    assert "Paris | City of Light" in msgs[1]["content"]


def test_prompt_v2_truncation_guard():
    msgs = build_judge_messages("Q?", "x" * 5000, "c" * 9000,
                                gold_answers=["g" * 2000],
                                max_chars={"context": 100, "answer": 50, "gold": 40})
    user = msgs[1]["content"]
    assert "[truncated]" in user
    assert user.count("[truncated]") == 3


def test_prompt_version_is_v2():
    assert JUDGE_PROMPT_VERSION == "judge-yesno-v2"


def test_gpt_oss_final_prefix_only_for_gpt_oss():
    assert judge_prompt_final_prefix("/mnt/data/gpt-oss-20b") == "<|channel|>final<|message|>"
    assert judge_prompt_final_prefix("gpt_oss-20b-Fork") == "<|channel|>final<|message|>"
    assert judge_prompt_final_prefix("/mnt/data/Qwen3-8B") == ""
    assert judge_prompt_final_prefix("") == ""


def test_chat_template_kwargs_by_family():
    assert judge_chat_template_kwargs("/mnt/data/Qwen3-8B") == {"enable_thinking": False}
    assert judge_chat_template_kwargs("/mnt/data/Qwen2.5-1.5B-Instruct") == {}
    assert judge_chat_template_kwargs("/mnt/data/gpt-oss-20b", "low") == {
        "reasoning_effort": "low"}
    assert judge_chat_template_kwargs("/mnt/data/gpt-oss-20b", None) == {}


# ---------- adapter gold extraction ----------

def test_gsm8k_gold_after_hash():
    assert gsm8k_gold("reasoning line 1\nline 2\n#### 72") == "72"


def test_gsm8k_gold_strips_commas():
    assert gsm8k_gold("work\n#### 1,234") == "1234"


def test_gsm8k_gold_fallback_plain():
    assert gsm8k_gold("  42 ") == "42"


def test_mmlu_gold_letter():
    assert mmlu_gold_letter(0) == "A"
    assert mmlu_gold_letter(3) == "D"


def test_render_mcq_question_options():
    q = _render_mcq_question("Pick one", ["alpha", "beta", "gamma", "delta"])
    assert q.splitlines()[1:] == ["A. alpha", "B. beta", "C. gamma", "D. delta"]


def test_gpqa_remap_block_stripped_and_gold_mapped():
    q = ("Which one?\n\na) x\nb) y\nc) z\nd) w\n\nA. d\nB. a\nC. b\nD. c")
    q2, gold = normalize_gpqa_item(q, "D")
    assert q2.endswith("d) w")
    assert "A. d" not in q2
    assert gold == "C"


def test_gpqa_plain_question_untouched():
    q = "Which?\n\nA. ~0.32\nB. ~0.39\nC. ~0.07\nD. ~0.11"
    q2, gold = normalize_gpqa_item(q, "A")
    assert q2 == q and gold == "A"


# ---------- adapter iterators on temp fixtures ----------

def _cfg_with(ds_name, entry):
    return {"datasets": {ds_name: entry}}


def test_math500_adapter(tmp_path):
    p = tmp_path / "test.jsonl"
    p.write_text(
        '{"problem": "What is 1+1?", "answer": "2", "subject": "Algebra", "level": 1, '
        '"unique_id": "t/1"}\n'
        '{"problem": "What is 2+2?", "answer": "4", "subject": "Algebra", "level": 1, '
        '"unique_id": "t/2"}\n')
    exs = list(get_dataset_iterator("math500", _cfg_with("math500", {"path": str(p)}), 5))
    assert len(exs) == 2
    assert exs[0].gold_answers == ["2"]
    assert exs[0].task == "math"
    assert exs[0].system_prompt  # non-empty default system prompt


def test_math500_adapter_limit(tmp_path):
    p = tmp_path / "test.jsonl"
    p.write_text('{"problem": "a", "answer": "1"}\n{"problem": "b", "answer": "2"}\n')
    exs = list(get_dataset_iterator("math500", _cfg_with("math500", {"path": str(p)}), 1))
    assert len(exs) == 1


def test_competition_math_adapter(tmp_path):
    p = tmp_path / "data.json"
    p.write_text('[{"problem": "1+1?", "numeric_answer": 2, "level": "Level 1", '
                 '"type": "Algebra", "id": 0}]')
    exs = list(get_dataset_iterator("competition_math",
                                    _cfg_with("competition_math", {"path": str(p)}), 5))
    assert len(exs) == 1
    assert exs[0].gold_answers == ["2"]


def test_gpqa_adapter(tmp_path):
    p = tmp_path / "gpqa.json"
    p.write_text('[{"question": "Which?\\nA. x\\nB. y", "answer": "B"}]')
    exs = list(get_dataset_iterator("gpqa_diamond", _cfg_with("gpqa_diamond", {"path": str(p)}), 5))
    assert len(exs) == 1
    assert exs[0].gold_answers == ["B"]
    assert exs[0].task == "mcq"


def test_mmlu_adapter(tmp_path):
    p = tmp_path / "mmlu.json"
    p.write_text('[{"question": "Pick", "choices": ["a", "b", "c", "d"], "answer": 2, '
                 '"id": 0, "subject": "college_mathematics"}]')
    exs = list(get_dataset_iterator("mmlu_college_mathematics",
                                    _cfg_with("mmlu_college_mathematics", {"path": str(p)}), 5))
    assert len(exs) == 1
    assert exs[0].gold_answers == ["C"]
    assert "C. c" in exs[0].question


def test_gsm8k_adapter(tmp_path):
    pd = pytest.importorskip("pandas")
    pyarrow = pytest.importorskip("pyarrow")
    p = tmp_path / "test.parquet"
    pd.DataFrame({"question": ["What is 2+3?"],
                  "answer": ["Add: 2+3=5.\n#### 5"]}).to_parquet(p)
    exs = list(get_dataset_iterator("gsm8k", _cfg_with("gsm8k", {"path": str(p)}), 5))
    assert len(exs) == 1
    assert exs[0].gold_answers == ["5"]


# ---------- blocked datasets stay blocked (no silent substitution) ----------

@pytest.mark.parametrize("name", ["hle", "livebench", "squad", "nq"])
def test_blocked_datasets_raise(name):
    with pytest.raises(ConfigError):
        list(get_dataset_iterator(name, {}, 4))


def test_unknown_dataset_raises():
    with pytest.raises(ConfigError):
        list(get_dataset_iterator("nope", {}, 4))


# ---------- SystemOne judge (StartLux-Decision /v1/systemone) ----------

def test_systemone_request_is_reference_aware_choice():
    req = build_factuality_request("What is 7*8?", "Final answer: 54",
                                   gold_answers=["56"])
    assert req["questions"]["factuality"]["type"] == "choice"
    st = req["state"]
    assert st["reference_ground_truth"] == "56"
    assert st["answer"] == "Final answer: 54"
    crit = req["questions"]["factuality"]["criteria"]
    assert set(crit) == {"correct", "hallucinated"}


def test_systemone_request_truncation_and_no_reference():
    long_q = "q" * 6000
    req = build_factuality_request(long_q, "a" * 3000, context="c" * 6000,
                                   max_chars={"question": 100, "context": 100, "answer": 50})
    assert "...[truncated]" in req["state"]["question"]
    assert "...[truncated]" in req["state"]["context"]
    assert "reference_ground_truth" not in req["state"]
    assert req["state"]["answer"].endswith("...[truncated]")


class _FakeService:
    """Mimics the startlux_decision /v1/systemone response."""

    def __init__(self, p_correct):
        self.p = p_correct

    def __call__(self, url, body, timeout):
        import io
        import urllib.response
        payload = {
            "answers": {"factuality": {"type": "choice", "choice":
                        "correct" if self.p >= 0.5 else "hallucinated",
                        "confidence": max(self.p, 1 - self.p),
                        "probabilities": {"correct": self.p, "hallucinated": 1 - self.p}}},
            "usage": {"input_tokens": 10, "output_tokens": 0},
            "model": "fake",
        }
        raw = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        return urllib.response.addinfourl(io.BytesIO(raw), headers, url, 200)


def test_systemone_judge_maps_choice_and_probability(monkeypatch):
    import json as _json

    j = SystemOneJudge(base_url="http://127.0.0.1:1")

    def fake_post(payload):
        return _json.loads(_json.dumps({
            "answers": {"factuality": {"type": "choice", "choice": "correct",
                                       "confidence": 0.9,
                                       "probabilities": {"correct": 0.9, "hallucinated": 0.1}}},
            "usage": {"input_tokens": 10, "output_tokens": 0},
            "model": "fake",
        }))
    monkeypatch.setattr(j, "_post", fake_post)
    r = j.judge("s1", "Q?", "good answer", gold_answers=["g"])
    assert r.hard_verdict == 0
    assert r.coverage == "ok"
    assert r.score_type == "choice_probability"
    assert abs(r.continuous_score - 0.1) < 1e-9

    def fake_post2(payload):
        return {"answers": {"factuality": {"type": "choice", "choice": "hallucinated",
                                           "confidence": 0.8,
                                           "probabilities": {"correct": 0.2, "hallucinated": 0.8}}},
                "usage": {}, "model": "fake"}
    monkeypatch.setattr(j, "_post", fake_post2)
    r2 = j.judge("s2", "Q?", "wrong answer", gold_answers=["g"])
    assert r2.hard_verdict == 1
    assert abs(r2.continuous_score - 0.8) < 1e-9


def test_systemone_judge_service_error_is_honest(monkeypatch):
    import urllib.error

    j = SystemOneJudge(base_url="http://127.0.0.1:1")

    def boom(payload):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(j, "_post", boom)
    r = j.judge("s1", "Q?", "A")
    assert r.hard_verdict is None and r.continuous_score is None
    assert r.coverage == "failed"
    assert "unreachable" in r.reason


def test_systemone_prompt_version():
    assert SYSTEMONE_PROMPT_VERSION == "systemone-factuality-v1"


# ---------- hard-tier adapters ----------

def test_mmlu_pro_adapter_ten_options(tmp_path):
    p = tmp_path / "mmlu_pro.json"
    opts = ["o0", "o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9"]
    p.write_text(json.dumps([{"question_id": "70", "question": "Pick",
                              "options": opts, "answer": "I", "answer_index": "8",
                              "cot_content": "", "category": "math", "src": "x"}]))
    exs = list(get_dataset_iterator("mmlu_pro", _cfg_with("mmlu_pro", {"path": str(p)}), 5))
    assert len(exs) == 1
    assert exs[0].gold_answers == ["I"]
    assert exs[0].question.splitlines()[-1].startswith("J. o9")
    assert exs[0].task == "mcq"


def test_hle_text_adapter_filters_images_and_mc(tmp_path):
    p = tmp_path / "hle.json"
    rows = [
        {"id": "a", "question": "q1", "answer": "ans1", "answer_type": "exactMatch",
         "image": "", "category": "Math", "raw_subject": "Trivia"},
        {"id": "b", "question": "q2", "answer": "ans2", "answer_type": "exactMatch",
         "image": "data:image/jpeg;base64,xxx", "category": "Other", "raw_subject": "X"},
        {"id": "c", "question": "q3", "answer": "(A)", "answer_type": "multipleChoice",
         "image": "", "category": "Physics", "raw_subject": "Y"},
    ]
    p.write_text(json.dumps(rows))
    exs = list(get_dataset_iterator("hle_text", _cfg_with("hle_text", {"path": str(p)}), 5))
    assert len(exs) == 1
    assert exs[0].sample_id.startswith("hle-")
    assert exs[0].gold_answers == ["ans1"]
    assert exs[0].task == "open_qa_hard"


def test_competition_math_level5_adapter(tmp_path):
    p = tmp_path / "l5.json"
    p.write_text(json.dumps([{"problem": "hard problem", "numeric_answer": "13",
                              "level": "Level 5", "type": "Algebra", "id": "0",
                              "solution": "..."}]))
    exs = list(get_dataset_iterator("competition_math_level5",
                                    _cfg_with("competition_math_level5", {"path": str(p)}), 5))
    assert len(exs) == 1
    assert exs[0].gold_answers == ["13"]
    assert exs[0].task == "math"
