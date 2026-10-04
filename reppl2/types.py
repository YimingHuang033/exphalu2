from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class InputUnit:
    unit_id: int
    text: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    source: str
    is_factual: bool = True


@dataclass
class Example:
    sample_id: str
    task: str
    question: str
    context: str = ""
    evidence: list[str] = field(default_factory=list)
    gold_answers: list[str] = field(default_factory=list)
    system_prompt: str = ""
    user_template: str = "{question}"
    split: str = "test"
    label: Optional[int] = None
    label_source: str = "none"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampledOutput:
    sample_id: str
    k: int
    token_ids: list[int]
    text: str
    token_logprobs: list[float]
    finish_reason: str = ""
    valid_len: int = -1

    def __post_init__(self):
        if self.valid_len < 0:
            self.valid_len = len(self.token_ids)


@dataclass
class Generation:
    sample_id: str
    prompt_text: str
    prompt_token_ids: list[int]
    greedy: SampledOutput
    samples: list[SampledOutput]
    sampling_config: dict[str, Any]
    template_hash: str = ""
    model_revision: str = ""
    backend: str = ""

    def to_dict(self):
        return {
            "sample_id": self.sample_id,
            "prompt_text": self.prompt_text,
            "prompt_token_ids": self.prompt_token_ids,
            "greedy": asdict(self.greedy),
            "samples": [asdict(s) for s in self.samples],
            "sampling_config": self.sampling_config,
            "template_hash": self.template_hash,
            "model_revision": self.model_revision,
            "backend": self.backend,
        }

    @staticmethod
    def from_dict(d):
        g = d["greedy"]
        return Generation(
            sample_id=d["sample_id"],
            prompt_text=d["prompt_text"],
            prompt_token_ids=d["prompt_token_ids"],
            greedy=SampledOutput(**g),
            samples=[SampledOutput(**s) for s in d["samples"]],
            sampling_config=d["sampling_config"],
            template_hash=d.get("template_hash", ""),
            model_revision=d.get("model_revision", ""),
            backend=d.get("backend", ""),
        )


@dataclass
class ReplayResult:
    sample_id: str
    context_token_ids: list[int]
    output_token_ids: list[int]
    hidden: Any
    valid_mask: list[int]
    timing_s: float
    backend: str
    dtype: str = ""
    note: str = ""


@dataclass
class Detection:
    sample_id: str
    method: str
    risk: Optional[float]
    inner: Optional[float]
    outer: Optional[float]
    validity: str
    reason: str = ""
    input_scores: Optional[dict] = None
    output_scores: Optional[dict] = None
    cost: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def to_row(self):
        return {
            "sample_id": self.sample_id,
            "method": self.method,
            "risk": self.risk,
            "inner": self.inner,
            "outer": self.outer,
            "validity": self.validity,
            "reason": self.reason,
            "gen_time_s": self.cost.get("gen_time_s"),
            "replay_time_s": self.cost.get("replay_time_s"),
        }


@dataclass
class JudgeResult:
    sample_id: str
    judge_name: str
    hard_verdict: Optional[int]
    continuous_score: Optional[float]
    score_type: str
    raw_output: str = ""
    coverage: str = "ok"
    reason: str = ""
    prompt_version: str = ""
    model_revision: str = ""
    timing_s: float = 0.0


def new_run_id(prefix: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{ts}-{uuid.uuid4().hex[:6]}"
