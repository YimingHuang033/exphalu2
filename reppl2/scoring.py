from __future__ import annotations

import hashlib
import time
from typing import Optional

from .backends.base import InferenceBackend, BackendError
from .types import Example, Generation, SampledOutput


def render_prompt(example: Example, tokenizer) -> tuple[str, list[int], str]:
    user_content = example.user_template.format(question=example.question)
    if example.context:
        user_content = f"{user_content}\n\nContext: {example.context}"
    messages = []
    if example.system_prompt:
        messages.append({"role": "system", "content": example.system_prompt})
    messages.append({"role": "user", "content": user_content})
    kwargs = {}
    name = str(getattr(tokenizer, "name_or_path", ""))
    if "qwen3" in name.lower():
        kwargs["enable_thinking"] = False
    try:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **kwargs
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    tmpl_hash = hashlib.sha256((getattr(tokenizer, "chat_template", "") or "").encode()).hexdigest()[:12]
    return text, ids, tmpl_hash


def sample_and_score(backend: InferenceBackend, example: Example, tokenizer,
                     sampling_cfg: dict, k: int, sample_seed: Optional[int] = None,
                     logger=None) -> tuple[Generation, dict]:
    prompt_text, prompt_ids, tmpl_hash = render_prompt(example, tokenizer)
    max_new = int(sampling_cfg.get("max_new_tokens", 32))
    stop_ids = sampling_cfg.get("stop_token_ids")

    t0 = time.time()
    greedy = backend.sample(
        prompt_ids,
        {"temperature": 0.0, "max_new_tokens": max_new, "stop_token_ids": stop_ids},
        n=1, seed=None,
    )
    t1 = time.time()
    sampled = backend.sample(
        prompt_ids,
        {"temperature": float(sampling_cfg.get("temperature", 1.0)),
         "top_p": float(sampling_cfg.get("top_p", 1.0)),
         "top_k": int(sampling_cfg.get("top_k", -1)),
         "max_new_tokens": max_new, "stop_token_ids": stop_ids},
        n=k, seed=sample_seed,
    )
    gen_time = (t1 - t0) + (time.time() - t1)

    greedy_out = SampledOutput(
        sample_id=example.sample_id, k=0,
        token_ids=greedy["token_ids"][0], text=greedy["texts"][0],
        token_logprobs=greedy["logprobs"][0], finish_reason=greedy["finish_reasons"][0],
    )
    if len(greedy_out.token_ids) == 0:
        raise BackendError(f"greedy generation empty for {example.sample_id}")
    samples = [
        SampledOutput(
            sample_id=example.sample_id, k=i + 1,
            token_ids=sampled["token_ids"][i], text=sampled["texts"][i],
            token_logprobs=sampled["logprobs"][i], finish_reason=sampled["finish_reasons"][i],
        )
        for i in range(len(sampled["token_ids"]))
    ]
    if len(samples) < 2:
        raise BackendError(f"K<2 sampled outputs for {example.sample_id}; raise K or check backend")
    gen = Generation(
        sample_id=example.sample_id,
        prompt_text=prompt_text,
        prompt_token_ids=prompt_ids,
        greedy=greedy_out,
        samples=samples,
        sampling_config=dict(sampling_cfg),
        template_hash=tmpl_hash,
        backend=backend.name,
    )
    return gen, {"gen_time_s": gen_time}


def valid_len(token_ids: list[int], eos_token_ids: Optional[set]) -> int:
    if not eos_token_ids:
        return len(token_ids)
    last = len(token_ids)
    while last > 0 and token_ids[last - 1] in eos_token_ids:
        last -= 1
    return last


def compute_outer(greedy: SampledOutput, sample_lens: list[int],
                  eos_token_ids: Optional[set] = None) -> float:
    """Outer = -sum_t log p(y0[t]|x,y0[:t]) / mean_k len(yk)  (DESIGN.md §3.2)."""
    ids, lps = greedy.token_ids, greedy.token_logprobs
    if len(ids) != len(lps):
        raise BackendError(f"greedy token/logprob length mismatch: {len(ids)} vs {len(lps)}")
    n = valid_len(ids, eos_token_ids)
    if n == 0:
        raise BackendError("greedy output has no non-special tokens")
    tail = lps[n:]
    if any(lp != lp for lp in tail):
        raise BackendError("non-finite logprob inside counted region")
    nll_sum = -sum(lps[:n])
    if not sample_lens:
        raise BackendError("no sampled output lengths for Outer normalization")
    mean_len = sum(sample_lens) / len(sample_lens)
    if mean_len <= 0:
        raise BackendError("mean sampled length is zero")
    return float(nll_sum / mean_len)
