"""Detection pipeline: per-sample orchestration of RePPL-A/B and probability baselines."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .backends.base import BackendError
from .baselines.probability import (outer_perplexity_risk, lnpe_risk, eigenscore_last_risk,
                                    output_length_risk, semantic_entropy_risk)
from .interventions import neutralize_unit, EditError
from .methods.common import AggregatorConfig
from .methods.reppl_a import reppl_a_inner
from .methods.reppl_b import reppl_b_inner
from .scoring import compute_outer, render_prompt, valid_len
from .segmentation import segment_input, unit_token_spans, SegmentationError
from .types import Generation, SampledOutput


def load_generation(g: dict) -> Generation:
    return Generation(
        sample_id=g["sample_id"], prompt_text=g["prompt_text"],
        prompt_token_ids=g["prompt_token_ids"],
        greedy=SampledOutput(**g["greedy"]),
        samples=[SampledOutput(**s) for s in g["samples"]],
        sampling_config=g.get("sampling_config", {}),
        template_hash=g.get("template_hash", ""),
        model_revision=g.get("model_revision", ""),
        backend=g.get("backend", ""),
    )


DEFAULT_METHODS = ["reppl-a", "outer-perplexity", "lnpe", "eigenscore-last",
                   "semantic-entropy-lexical", "length"]


def detect_one(backend, tokenizer, cfg, gen: Generation, ex, entailment_model=None,
               interp: bool = False, logger=None) -> dict:
    out = {"sample_id": gen.sample_id, "methods": {}, "interp": None}
    methods = cfg.get("methods", DEFAULT_METHODS)
    agg_cfg = AggregatorConfig(**(cfg.get("aggregator") or {}))
    eos = set(cfg.get("_eos_ids_", []))
    sample_lens = [valid_len(s.token_ids, eos) for s in gen.samples]

    outer = None
    if any(m in methods for m in ("reppl-a", "reppl-b", "outer-perplexity")):
        outer = compute_outer(gen.greedy, sample_lens, eos_token_ids=eos)

    need_states = any(m in methods for m in ("reppl-a", "eigenscore-last", "reppl-b"))
    replay_ctx = None
    replay_samples = []
    if need_states:
        rep = backend.replay_last_hidden(gen.prompt_token_ids, gen.greedy.token_ids)
        replay_ctx = rep["hidden"]
        for s in gen.samples:
            repk = backend.replay_last_hidden(gen.prompt_token_ids, s.token_ids)
            replay_samples.append(repk["hidden"])

    if "outer-perplexity" in methods:
        out["methods"]["outer-perplexity"] = {
            "risk": float(outer), "inner": None, "outer": None,
            "validity": "ok", "reason": ""}
    if "lnpe" in methods:
        try:
            v = lnpe_risk([s.token_logprobs for s in gen.samples],
                          [s.token_ids for s in gen.samples], eos_token_ids=eos)
            out["methods"]["lnpe"] = {"risk": v, "inner": None, "outer": None,
                                      "validity": "ok", "reason": ""}
        except BackendError as e:
            out["methods"]["lnpe"] = {"risk": None, "inner": None, "outer": None,
                                      "validity": "invalid", "reason": str(e)}
    if "length" in methods:
        out["methods"]["length"] = {"risk": output_length_risk(sample_lens), "inner": None,
                                    "outer": None, "validity": "ok", "reason": ""}
    if "eigenscore-last" in methods:
        try:
            v = eigenscore_last_risk(replay_samples)
            out["methods"]["eigenscore-last"] = {"risk": v, "inner": None, "outer": None,
                                                 "validity": "ok", "reason": ""}
        except BackendError as e:
            out["methods"]["eigenscore-last"] = {"risk": None, "inner": None, "outer": None,
                                                 "validity": "invalid", "reason": str(e)}
    if any(m in methods for m in ("semantic-entropy", "semantic-entropy-lexical")):
        for name in ("semantic-entropy", "semantic-entropy-lexical"):
            if name not in methods:
                continue
            model = entailment_model if name == "semantic-entropy" else None
            try:
                v, info = semantic_entropy_risk(
                    [s.text for s in gen.samples],
                    [s.token_logprobs for s in gen.samples],
                    entailment_model=model,
                    eos_token_ids=eos, sample_token_ids=[s.token_ids for s in gen.samples])
                out["methods"][name] = {"risk": v, "inner": None, "outer": None,
                                        "validity": "ok", "reason": str(info)}
            except BackendError as e:
                out["methods"][name] = {"risk": None, "inner": None, "outer": None,
                                        "validity": "invalid", "reason": str(e)}

    if "reppl-a" in methods:
        try:
            units = segment_input(ex)
            spans = unit_token_spans(units, gen.prompt_text, gen.prompt_token_ids, tokenizer)
            agg, info = reppl_a_inner(
                gen, {"context": replay_ctx, "samples": replay_samples},
                spans, float(cfg.get("association_temperature", 1.0)), agg_cfg, outer)
            out["methods"]["reppl-a"] = {
                "risk": agg.risk, "inner": agg.inner, "outer": agg.outer,
                "validity": agg.validity.status, "reason": agg.validity.reason}
            if interp:
                out["interp"] = {
                    "reppl-a": {
                        "units": info["units"],
                        "A": info["A"].tolist(),
                        "mu": agg.mu.tolist(), "sigma": agg.sigma.tolist(),
                        "r": agg.r.tolist(), "p_hat": agg.p_hat.tolist(),
                        "raw_max_similarity": info["raw_max_similarity"].tolist(),
                    },
                    "output_nll": {
                        "greedy_token_nll": [-lp for lp in gen.greedy.token_logprobs],
                    },
                }
        except (BackendError, SegmentationError) as e:
            out["methods"]["reppl-a"] = {"risk": None, "inner": None, "outer": None,
                                         "validity": "invalid", "reason": str(e)}

    if "reppl-b" in methods:
        try:
            out["methods"]["reppl-b"] = _detect_b(
                backend, tokenizer, cfg, gen, ex, agg_cfg, outer, eos, interp, out)
        except (BackendError, SegmentationError, EditError) as e:
            out["methods"]["reppl-b"] = {"risk": None, "inner": None, "outer": None,
                                         "validity": "invalid", "reason": str(e)}
    return out


def _detect_b(backend, tokenizer, cfg, gen: Generation, ex, agg_cfg, outer, eos,
              interp: bool, out: dict) -> dict:
    units = segment_input(ex)
    spans = unit_token_spans(units, gen.prompt_text, gen.prompt_token_ids, tokenizer)
    factual = [u for u in spans if u.get("is_factual")]
    max_j = int(cfg.get("reppl_b", {}).get("max_units", 4))
    factual = factual[:max_j]
    if not factual:
        return {"risk": None, "inner": None, "outer": None, "validity": "skipped",
                "reason": "no factual unit eligible for editing", "coverage": 0.0}

    K = len(gen.samples)
    z = np.stack([s.mean(axis=0) for s in
                  [backend.replay_last_hidden(gen.prompt_token_ids, s.token_ids)["hidden"]
                   for s in gen.samples]])

    z_edit = {}
    edit_records = []
    for u in factual:
        try:
            new_ctx, new_q, rec = neutralize_unit(ex.context, ex.question, u)
        except EditError as e:
            edit_records.append({"unit_id": u["unit_id"], "skipped": True, "reason": str(e)})
            continue
        ex_edit = ex.model_copy() if hasattr(ex, "model_copy") else _copy_ex(ex)
        ex_edit.context, ex_edit.question = new_ctx, new_q
        _, edit_prompt_ids, _ = render_prompt(ex_edit, tokenizer)
        if edit_prompt_ids == gen.prompt_token_ids:
            edit_records.append({"unit_id": u["unit_id"], "skipped": True,
                                 "reason": "edited prompt identical to original"})
            continue
        zj = []
        for s in gen.samples:
            repj = backend.replay_last_hidden(edit_prompt_ids, s.token_ids)
            zj.append(repj["hidden"].mean(axis=0))
        z_edit[u["unit_id"]] = np.stack(zj)
        edit_records.append({"unit_id": u["unit_id"], "skipped": False,
                             "original_text": rec.original_text,
                             "replacement": rec.replacement_text,
                             "edit_type": rec.edit_type})

    res, raw = reppl_b_inner(z, z_edit, agg_cfg, outer)
    agg = res["agg"]
    coverage = len(z_edit) / max(1, len(factual))
    if interp and out.get("interp") is not None:
        out["interp"]["reppl-b"] = {
            "j_ids": res["j_ids"], "q_raw": raw["q_raw"].tolist(),
            "A": res["A"].tolist(), "edits": edit_records, "coverage": coverage,
            "mu": agg.mu.tolist(), "r": agg.r.tolist()}
    elif interp:
        out["interp"] = {"reppl-b": {"j_ids": res["j_ids"], "q_raw": raw["q_raw"].tolist(),
                                     "A": res["A"].tolist(), "edits": edit_records,
                                     "coverage": coverage}}
    return {"risk": agg.risk, "inner": agg.inner, "outer": agg.outer,
            "validity": agg.validity.status, "reason": agg.validity.reason,
            "coverage": coverage, "n_units_edited": len(z_edit)}


def _copy_ex(ex):
    import copy

    e = copy.copy(ex)
    return e
