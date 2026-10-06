"""Detection pipeline: per-sample orchestration of RePPL-A/B and probability baselines."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .backends.base import BackendError
from .baselines.probability import (outer_perplexity_risk, lnpe_risk, eigenscore_last_risk,
                                    output_length_risk, semantic_entropy_risk)
from .baselines.sese import sese_risk
from .baselines.semantic_energy import semantic_energy_risk
from .baselines.rauq import rauq_risk
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


def _groups_to_clusters(semantic_ids: list[int]) -> list[list[int]]:
    """semantic-entropy grouping -> index clusters for semantic-energy."""
    groups: dict[int, list[int]] = {}
    for i, sid in enumerate(semantic_ids):
        groups.setdefault(int(sid), []).append(i)
    return [g for _, g in sorted(groups.items())]


DEFAULT_METHODS = ["reppl-a", "outer-perplexity", "lnpe", "eigenscore-last",
                   "semantic-entropy-lexical", "length"]


def detect_one(backend, tokenizer, cfg, gen: Generation, ex, entailment_model=None,
               sese_models=None, interp: bool = False, logger=None) -> dict:
    out = {"sample_id": gen.sample_id, "methods": {}, "interp": None}
    methods = cfg.get("methods", DEFAULT_METHODS)
    agg_cfg = AggregatorConfig(**(cfg.get("aggregator") or {}))
    eos = set(cfg.get("_eos_ids_", []))
    sample_lens = [valid_len(s.token_ids, eos) for s in gen.samples]

    outer = None
    if any(m in methods for m in ("reppl-a", "reppl-b", "outer-perplexity")):
        outer = compute_outer(gen.greedy, sample_lens, eos_token_ids=eos)

    need_states = any(m in methods for m in ("reppl-a", "eigenscore-last", "reppl-b",
                                             "d-score-last", "reppl-ab"))
    replay_ctx = None
    replay_greedy_out = None
    replay_samples = []
    if need_states:
        rep = backend.replay_last_hidden(gen.prompt_token_ids, gen.greedy.token_ids)
        # u[j] pools PROMPT-side states of the greedy replay (DESIGN section 4.1 step 3);
        # "hidden" alone holds only output-token states.
        replay_ctx = rep.get("hidden_context")
        if replay_ctx is None:
            raise BackendError(
                "backend replay returned no context-side states; required by reppl-a")
        replay_greedy_out = rep["hidden"]
        for s in gen.samples:
            repk = backend.replay_last_hidden(gen.prompt_token_ids, s.token_ids)
            replay_samples.append(repk["hidden"])
    z_reuse = (np.stack([s.mean(axis=0) for s in replay_samples])
               if replay_samples else None)

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

    if "d-score-last" in methods:
        try:
            from .baselines.dscore import dscore_last_risk, VERSION as DSCORE_VERSION
            v = dscore_last_risk(replay_greedy_out,
                                 tau=float((cfg.get("d_score") or {}).get("tau", 10.0)))
            out["methods"]["d-score-last"] = {
                "risk": v, "inner": None, "outer": None, "validity": "ok",
                "reason": str({"version": DSCORE_VERSION,
                               "tau": float((cfg.get("d_score") or {}).get("tau", 10.0)),
                               "matrix": "greedy answer last-layer states (uncentered)"})}
        except BackendError as e:
            out["methods"]["d-score-last"] = {"risk": None, "inner": None, "outer": None,
                                              "validity": "invalid", "reason": str(e)}

    if "sese" in methods:
        try:
            if sese_models is None:
                raise BackendError("sese models not loaded (NLI / sentence-embedding)")
            nli, embedder = sese_models
            scfg = cfg.get("sese") or {}
            v, info = sese_risk([s.text for s in gen.samples], nli, embedder,
                                tree_depth=int(scfg.get("tree_depth", 2)),
                                w_entail=float(scfg.get("w_entail", 0.65)),
                                similarity_threshold=float(scfg.get("similarity_threshold", 0.3)))
            out["methods"]["sese"] = {"risk": v, "inner": None, "outer": None,
                                      "validity": "ok", "reason": str(info)}
        except BackendError as e:
            out["methods"]["sese"] = {"risk": None, "inner": None, "outer": None,
                                      "validity": "invalid", "reason": str(e)}

    if "semantic-energy" in methods:
        try:
            # clusters come from the same grouping the semantic-entropy baseline uses
            _, se_info = semantic_entropy_risk(
                [s.text for s in gen.samples],
                [s.token_logprobs for s in gen.samples],
                entailment_model=entailment_model, eos_token_ids=eos,
                sample_token_ids=[s.token_ids for s in gen.samples])
            clusters = _groups_to_clusters(se_info["semantic_ids"])
            v, info = semantic_energy_risk(
                [s.token_logprobs for s in gen.samples], clusters)
            out["methods"]["semantic-energy"] = {"risk": v, "inner": None, "outer": None,
                                                 "validity": "ok",
                                                 "reason": str({**info,
                                                                "grouping": se_info.get("variant")})}
        except BackendError as e:
            out["methods"]["semantic-energy"] = {"risk": None, "inner": None, "outer": None,
                                                 "validity": "invalid", "reason": str(e)}

    if "rauq" in methods:
        try:
            if not hasattr(backend, "replay_attention"):
                raise BackendError(
                    f"backend {backend.name!r} cannot expose attention weights; "
                    "RAUQ requires the transformers reference environment")
            rep = backend.replay_attention(gen.prompt_token_ids, gen.greedy.token_ids)
            rcfg = cfg.get("rauq") or {}
            v = rauq_risk(rep["attentions"], gen.greedy.token_logprobs,
                          ctx_len=len(gen.prompt_token_ids),
                          alpha=float(rcfg.get("alpha", 0.2)),
                          layers=rcfg.get("layers"),
                          head=str(rcfg.get("head", "max")),
                          token_aggregation=str(rcfg.get("token_aggregation", "meanmin")),
                          aggregation=str(rcfg.get("aggregation", "mean")))
            out["methods"]["rauq"] = {"risk": v["risk"], "inner": None, "outer": None,
                                      "validity": "ok",
                                      "reason": str({k: v[k] for k in
                                                     ("version", "heads", "layers", "alpha")})}
        except BackendError as e:
            out["methods"]["rauq"] = {"risk": None, "inner": None, "outer": None,
                                      "validity": "invalid", "reason": str(e)}

    a_status = None
    a_units = None
    if "reppl-a" in methods or "reppl-ab" in methods:
        try:
            units = segment_input(ex)
            spans = unit_token_spans(units, gen.prompt_text, gen.prompt_token_ids, tokenizer)
            agg, info = reppl_a_inner(
                gen, {"context": replay_ctx, "samples": replay_samples},
                spans, float(cfg.get("association_temperature", 1.0)), agg_cfg, outer)
            out["methods"]["reppl-a"] = {
                "risk": agg.risk, "inner": agg.inner, "outer": agg.outer,
                "validity": agg.validity.status, "reason": agg.validity.reason}
            a_status = agg.validity.status
            a_units = info["units"]
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
            a_status = "invalid"

    if "reppl-ab" in methods:
        try:
            if a_status != "ok" or a_units is None:
                raise BackendError(
                    f"reppl-ab needs a valid reppl-a nomination; reppl-a status={a_status}")
            crit = str((cfg.get("reppl_ab") or {}).get("nominate_by", "r"))
            if crit == "mu":
                vals = agg.mu
            elif crit == "r":
                vals = agg.r
            else:
                raise BackendError(f"reppl_ab.nominate_by must be 'r' or 'mu', got {crit!r}")
            max_j = int(cfg.get("reppl_b", {}).get("max_units", 4))
            order = np.argsort(vals)[::-1][:max_j]
            nominated = [int(a_units[i]["unit_id"]) for i in order]
            res_ab = _detect_b(
                backend, tokenizer, cfg, gen, ex, agg_cfg, outer, eos, interp, out,
                restrict_unit_ids=nominated, z_precomputed=z_reuse,
                nomination={"by": crit, "unit_ids": nominated,
                            "values": [float(vals[i]) for i in order]},
                interp_key="reppl-ab")
            out["methods"]["reppl-ab"] = {k: v for k, v in res_ab.items() if k != "rawq"}
        except (BackendError, SegmentationError, EditError) as e:
            out["methods"]["reppl-ab"] = {"risk": None, "inner": None, "outer": None,
                                          "validity": "invalid", "reason": str(e)}

    if "reppl-b" in methods:
        try:
            res_b = _detect_b(
                backend, tokenizer, cfg, gen, ex, agg_cfg, outer, eos, interp, out,
                z_precomputed=z_reuse)
            out["methods"]["reppl-b"] = {k: v for k, v in res_b.items() if k != "rawq"}
            if res_b.get("rawq") is not None:
                out["methods"]["reppl-b-rawq"] = res_b["rawq"]
        except (BackendError, SegmentationError, EditError) as e:
            out["methods"]["reppl-b"] = {"risk": None, "inner": None, "outer": None,
                                         "validity": "invalid", "reason": str(e)}
    return out


def _detect_b(backend, tokenizer, cfg, gen: Generation, ex, agg_cfg, outer, eos,
              interp: bool, out: dict, restrict_unit_ids=None,
              nomination=None, z_precomputed=None, interp_key="reppl-b") -> dict:
    units = segment_input(ex)
    spans = unit_token_spans(units, gen.prompt_text, gen.prompt_token_ids, tokenizer)
    factual = [u for u in spans if u.get("is_factual")]
    max_j = int(cfg.get("reppl_b", {}).get("max_units", 4))
    if restrict_unit_ids is not None:
        # RepplAB (DESIGN §5.3): only A-nominated units are edited (cost control)
        by_id = {u["unit_id"]: u for u in factual}
        factual = [by_id[i] for i in restrict_unit_ids if i in by_id][:max_j]
    else:
        factual = factual[:max_j]
    if not factual:
        return {"risk": None, "inner": None, "outer": None, "validity": "skipped",
                "reason": "no factual unit eligible for editing", "coverage": 0.0,
                "rawq": None}

    K = len(gen.samples)
    if z_precomputed is not None:
        z = z_precomputed
    else:
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
    agg_raw = res["agg_raw"]
    coverage = len(z_edit) / max(1, len(factual))
    b_interp = {"j_ids": res["j_ids"], "q_raw": raw["q_raw"].tolist(),
                "A": res["A"].tolist(), "edits": edit_records, "coverage": coverage,
                "mu": agg.mu.tolist(), "r": agg.r.tolist()}
    if nomination is not None:
        b_interp["nomination"] = nomination
    if interp and out.get("interp") is not None:
        out["interp"][interp_key] = b_interp
    elif interp:
        out["interp"] = {interp_key: b_interp}
    return {"risk": agg.risk, "inner": agg.inner, "outer": agg.outer,
            "validity": agg.validity.status, "reason": agg.validity.reason,
            "coverage": coverage, "n_units_edited": len(z_edit),
            # raw-q CV ablation (DESIGN §5.1): same aggregator on the unnormalized q
            "rawq": {"risk": agg_raw.risk, "inner": agg_raw.inner, "outer": agg_raw.outer,
                     "validity": agg_raw.validity.status, "reason": agg_raw.validity.reason,
                     "coverage": coverage, "n_units_edited": len(z_edit)}}


def _copy_ex(ex):
    import copy

    e = copy.copy(ex)
    return e
