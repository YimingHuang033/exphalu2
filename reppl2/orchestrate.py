"""Stage orchestration: verify-backend, generate, detect, judge."""
from __future__ import annotations

import platform
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .backends import build_backend
from .cache import RunStore
from .config_loader import require, ConfigError
from .data.adapters import get_dataset_iterator, DATASET_STATUS, IMPLEMENTED_DATASETS
from .judges.local import LocalJudge
from .judges.systemone import SystemOneJudge
from .logging_utils import save_json, load_json
from .baselines.registry import METHOD_REGISTRY
from .pipeline import detect_one, load_generation, DEFAULT_METHODS
from .scoring import (sample_and_score, resolve_chat_template_kwargs, split_think,
                      THINK_CLOSE)
from .types import Example

try:
    import torch
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False


def env_fingerprint() -> dict:
    fp = {"python": platform.python_version()}
    for mod in ("vllm", "transformers", "numpy"):
        try:
            fp[mod] = __import__(mod).__version__
        except Exception:
            fp[mod] = None
    if _TORCH_OK:
        fp["torch"] = torch.__version__
        fp["cuda_available"] = torch.cuda.is_available()
        fp["gpus"] = ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                      if torch.cuda.is_available() else [])
    return fp


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def peak_mem_mib() -> Optional[float]:
    if _TORCH_OK and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    return None


def resolve_model(cfg: dict, name: str | None = None) -> str:
    key = name if name is not None else require(cfg, "model")
    entry = (cfg.get("models") or {}).get(key)
    if entry is None:
        raise ConfigError(f"model '{key}' not registered in config.models")
    path = entry.get("path")
    if not path or not Path(path).exists():
        raise ConfigError(
            f"model '{key}' path '{path}' does not exist on this machine. "
            "Fix config/ or record the model as blocked; no silent substitution.")
    return path


def eos_ids(tokenizer) -> set:
    ids = set()
    if tokenizer.eos_token:
        ids.update(tokenizer(tokenizer.eos_token, add_special_tokens=False)["input_ids"])
    if getattr(tokenizer, "pad_token_id", None) is not None:
        ids.add(tokenizer.pad_token_id)
    return ids


def iter_examples(cfg, logger):
    ds_name = require(cfg, "dataset")
    status = DATASET_STATUS.get(ds_name, "unknown")
    if status != "implemented":
        raise ConfigError(f"dataset '{ds_name}' status={status}; implemented now: {IMPLEMENTED_DATASETS}")
    limit = int(cfg.get("num_samples", 8))
    exs = list(get_dataset_iterator(ds_name, cfg, limit))
    if not exs:
        raise ConfigError(f"dataset '{ds_name}' yielded 0 examples")
    logger.info(f"dataset '{ds_name}': {len(exs)} examples")
    return exs


def cmd_verify_backend(cfg, logger, run_dir) -> dict:
    model_path = resolve_model(cfg)
    backend = build_backend(cfg.get("backend", "vllm"), model_path, cfg.get("backend_cfg", {}), logger)
    tokenizer = load_tokenizer(model_path)
    report = {"backend": backend.name, "model": model_path,
              "capabilities": backend.capabilities(), "checks": []}
    gating_failed = []

    def check(name, ok, detail="", gating=True):
        report["checks"].append({"check": name, "ok": bool(ok), "detail": detail,
                                 "gating": bool(gating)})
        logger.info(f"[verify] {name}: {'OK' if ok else ('WARN' if not gating else 'FAIL')} {detail}")
        if not ok and gating:
            gating_failed.append(name)
        return bool(ok)

    ex = Example(sample_id="verify-0", task="verify",
                 question="What is the capital of France? Answer in one word.",
                 system_prompt="You are a helpful assistant.")
    eos = eos_ids(tokenizer)
    try:
        gen, cost = sample_and_score(
            backend, ex, tokenizer,
            {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 16, "stop_token_ids": list(eos)},
            k=3, sample_seed=7)
        g = gen.greedy
        check("generation_nonempty", len(g.token_ids) > 0)
        lps = np.array(g.token_logprobs, dtype=np.float64)
        check("greedy_logprobs_finite", bool(np.isfinite(lps).all()))
        check("greedy_logprobs_nonpositive", bool((lps <= 1e-6).all()))
        uniq = {tuple(s.token_ids) for s in gen.samples}
        check("sampled_outputs_distinct", len(uniq) >= 2,
              f"{len(uniq)}/3 distinct (peaked outputs are valid low variance, recorded not failed)",
              gating=False)
        sc = backend.score(gen.prompt_token_ids, g.token_ids)
        ok_len = check("score_length_matches", len(sc) == len(g.token_ids),
                       f"{len(sc)} vs {len(g.token_ids)}")
        if ok_len:
            diff = float(np.max(np.abs(np.array(sc) - lps)))
            check("score_alignment_tolerance", diff < 0.5,
                  f"max|diff|={diff:.4f} (sampling vs replay numeric drift)")
        backend.release_for_replay()
        rep = backend.replay_last_hidden(gen.prompt_token_ids, g.token_ids)
        h = rep["hidden"]
        check("hidden_shape", h.ndim == 2 and h.shape[0] == len(g.token_ids), str(h.shape))
        check("hidden_finite", bool(np.isfinite(h).all()))
        check("hidden_dtype_float32", str(h.dtype) == "float32", str(h.dtype))
        report["sample_output_text"] = g.text
    except Exception as e:
        check("end_to_end", False, repr(e)[:400])
    if backend.name == "sglang":
        check("sglang_blocked", False, "sglang not installed; blocked per DESIGN")
    report["status"] = "passed" if not gating_failed else "failed"
    backend.close()
    save_json(run_dir / "verify_backend.json", report)
    return report


def cmd_generate(cfg, logger, run_dir, force=False) -> dict:
    store = RunStore(run_dir, cfg["_config_hash_"])
    if store.up_to_date("generation") and not force:
        logger.info("[generate] stage up-to-date; skipping (use --force to rerun)")
        return store.load_stage("generation")
    model_path = resolve_model(cfg)
    backend = build_backend(cfg.get("backend", "vllm"), model_path, cfg.get("backend_cfg", {}), logger)
    tokenizer = load_tokenizer(model_path)
    examples = iter_examples(cfg, logger)
    # Invalidate before replacing dataset.json, including interrupted forced reruns.
    store.invalidate("generation")
    save_json(run_dir / "dataset.json", {
        "dataset": require(cfg, "dataset"),
        "examples": [e.__dict__ for e in examples],
    })
    sampling_cfg = dict(cfg.get("sampling") or {})
    # per-dataset sampling overrides (e.g. math needs more max_new_tokens), config-driven
    ds_cfg = (cfg.get("datasets") or {}).get(require(cfg, "dataset")) or {}
    sampling_cfg.update(dict(ds_cfg.get("sampling") or {}))
    eos = eos_ids(tokenizer)
    sampling_cfg.setdefault("stop_token_ids", list(eos))
    k = int(sampling_cfg.get("k", 5))
    # reasoning-model support: run-level chat_template_kwargs > model registry entry
    # > None (render_prompt falls back to the name-based default)
    ctk = resolve_chat_template_kwargs(cfg, tokenizer)
    thinking_on = bool((ctk or {}).get("enable_thinking", False))
    if thinking_on:
        logger.info(f"[gen] thinking mode ON (chat_template_kwargs={ctk}); "
                    f"max_new_tokens={sampling_cfg.get('max_new_tokens')} must cover "
                    f"think + answer (DESIGN 9.5.4)")

    gens, meta = [], []
    n_no_answer = 0
    t_all = time.time()
    for ex in examples:
        try:
            gen, cost = sample_and_score(backend, ex, tokenizer, sampling_cfg, k=k,
                                         sample_seed=sampling_cfg.get("seed"),
                                         chat_template_kwargs=ctk)
            _, ans = split_think(gen.greedy.text)
            if thinking_on and (not ans.strip()
                                or THINK_CLOSE not in gen.greedy.text):
                n_no_answer += 1
            gens.append(gen)
            meta.append({"sample_id": ex.sample_id, "validity": "ok",
                         "reason": "", "gen_time_s": cost["gen_time_s"]})
            logger.info(f"[gen] {ex.sample_id}: greedy={len(gen.greedy.token_ids)}tok "
                        f"K={len(gen.samples)} {cost['gen_time_s']:.2f}s")
        except Exception as e:
            meta.append({"sample_id": ex.sample_id, "validity": "failed",
                         "reason": str(e)[:300], "gen_time_s": None})
            logger.warning(f"[gen] {ex.sample_id} FAILED: {e}")
    payload = {
        "model": require(cfg, "model"), "model_path": model_path, "backend": backend.name,
        "sampling_config": sampling_cfg,
        "thinking": {"enabled": thinking_on, "chat_template_kwargs": ctk or {},
                     "n_greedy_no_answer": n_no_answer,
                     "answer_view": "post-think (judge/labels exclude the think part)"},
        "generations": [g.to_dict() for g in gens],
        "per_sample_meta": meta,
        "timing": {"total_s": time.time() - t_all},
        "peak_mem_mib": peak_mem_mib(),
    }
    store.save_stage("generation", payload)
    store.save_stage("trajectory", {
        "model": require(cfg, "model"), "backend": backend.name,
        "trajectories": [{"sample_id": g.sample_id, "prompt_text": g.prompt_text,
                          "greedy_text": g.greedy.text,
                          "greedy_think_text": split_think(g.greedy.text)[0],
                          "greedy_answer_text": split_think(g.greedy.text)[1],
                          "sample_texts": [s.text for s in g.samples]} for g in gens]})
    backend.close()
    logger.info(f"[generate] done: {len(gens)} ok / {len(meta)}"
                + (f"; greedy outputs without answer after think: {n_no_answer} "
                   f"(truncation; NOT natural hallucination, DESIGN 9.5.4)"
                   if thinking_on and n_no_answer else ""))
    return payload


def load_entailment(cfg, logger, shared_nli=None):
    import torch

    path = ((cfg.get("models") or {}).get("deberta-mnli") or {}).get("path")
    if not path or not Path(path).exists():
        logger.warning("[detect] deberta-mnli path missing; semantic-entropy blocked, "
                       "lexical variant used instead (recorded, not substituted silently).")
        return None
    if shared_nli is not None:
        # share the SeSE DebertaNLI instance to avoid a second copy in GPU memory
        class _ArgmaxNLI:
            def check_implication(self, a, b):
                inputs = shared_nli.tokenizer(a, b, return_tensors="pt",
                                              truncation=True, max_length=512).to(shared_nli.device)
                with torch.no_grad():
                    logits = shared_nli.model(**inputs).logits
                return int(logits.argmax(dim=1)[0].item())

        logger.info("[detect] entailment model shared with sese (single instance)")
        return _ArgmaxNLI()
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModelForSequenceClassification.from_pretrained(path).eval()
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        mdl = mdl.to(dev)
        logger.info(f"[detect] entailment model loaded from {path}")

        class NLI:
            def check_implication(self, a, b):
                inputs = tok(a, b, return_tensors="pt", truncation=True, max_length=512).to(dev)
                with torch.no_grad():
                    logits = mdl(**inputs).logits
                return int(logits.argmax(dim=1)[0].item())

        return NLI()
    except Exception as e:
        logger.warning(f"[detect] entailment model failed to load ({e}); "
                       "semantic-entropy blocked, lexical variant used instead.")
        return None


def load_sese_models(cfg, logger):
    """SeSE semantic stack: NLI probability model + sentence embedder.

    Reuses models.deberta-mnli for the NLI part (config-driven dtype/device);
    the embedder defaults to CPU (static embeddings, cheap) to keep GPU memory
    for the engine and the NLI model.
    """
    scfg = cfg.get("sese") or {}
    nli_path = ((cfg.get("models") or {}).get("deberta-mnli") or {}).get("path")
    emb_path = ((cfg.get("models") or {}).get("sent-emb-static-mrl") or {}).get("path")
    if not nli_path or not Path(nli_path).exists():
        raise RuntimeError(f"sese: NLI model path missing/unreadable: {nli_path}")
    if not emb_path:
        raise RuntimeError("sese: sentence-embedding model path missing in config "
                           "(models.sent-emb-static-mrl.path)")
    from .baselines.sese import DebertaNLI, SentenceEmbedder

    nli = DebertaNLI(nli_path,
                     device=scfg.get("nli_device") or "cuda:0",
                     dtype=scfg.get("nli_dtype") or "float16")
    logger.info(f"[detect] sese NLI loaded from {nli_path} "
                f"({scfg.get('nli_dtype') or 'float16'} on {nli.device})")
    embedder = SentenceEmbedder(emb_path)
    embedder.model.to(scfg.get("emb_device") or "cpu")
    logger.info(f"[detect] sese embedder loaded from {emb_path}")
    return nli, embedder


def cmd_detect(cfg, logger, run_dir, force=False) -> dict:
    store = RunStore(run_dir, cfg["_config_hash_"])
    if store.up_to_date("detection") and not force:
        logger.info("[detect] stage up-to-date; skipping")
        return store.load_stage("detection")
    gen_payload = store.load_stage("generation")
    dataset_payload = load_json(run_dir / "dataset.json")
    ex_by_id = {e["sample_id"]: Example(**e) for e in dataset_payload["examples"]}
    model_path = gen_payload["model_path"]
    tokenizer = load_tokenizer(model_path)
    cfg = dict(cfg)
    cfg["_eos_ids_"] = list(eos_ids(tokenizer))

    backend = build_backend(gen_payload["backend"], model_path, cfg.get("backend_cfg", {}), logger)
    methods = cfg.get("methods") or DEFAULT_METHODS
    sese_models = None
    if "sese" in methods:
        sese_models = load_sese_models(cfg, logger)
    entailment_model = None
    if any(m in methods for m in ("semantic-entropy", "semantic-energy")):
        # share the SeSE NLI instance when both methods are requested
        shared = sese_models[0] if sese_models is not None else None
        entailment_model = load_entailment(cfg, logger, shared_nli=shared)

    results = []
    t_all = time.time()
    for g in gen_payload["generations"]:
        gen = load_generation(g)
        t0 = time.time()
        try:
            r = detect_one(backend, tokenizer, cfg, gen, ex_by_id[gen.sample_id],
                           entailment_model=entailment_model, sese_models=sese_models,
                           interp=True, logger=logger)
            r["replay_time_s"] = time.time() - t0
            results.append(r)
            risks = {m: (v.get("risk") if isinstance(v, dict) else None)
                     for m, v in r["methods"].items()}
            logger.info(f"[detect] {gen.sample_id}: {risks} ({r['replay_time_s']:.2f}s)")
        except Exception as e:
            results.append({"sample_id": gen.sample_id, "methods": {},
                            "validity": "failed", "reason": str(e)[:400],
                            "replay_time_s": time.time() - t0})
            logger.warning(f"[detect] {gen.sample_id} FAILED: {e}")
    payload = {
        "model": gen_payload["model"], "backend": gen_payload["backend"],
        "methods_requested": methods,
        "method_registry_status": METHOD_REGISTRY,
        "per_sample": results,
        "timing": {"total_s": time.time() - t_all},
        "peak_mem_mib": peak_mem_mib(),
    }
    store.save_stage("detection", payload)
    backend.close()
    if sese_models is not None:
        del sese_models
    if entailment_model is not None:
        del entailment_model
        if _TORCH_OK:
            torch.cuda.empty_cache()
    logger.info(f"[detect] done: {len(results)} samples")
    return payload


def resolve_judge_model(cfg, logger) -> dict:
    """Resolve the judge provider → a concrete judge construction spec.

    Providers:
      - local_self / gpt_oss_20b (adapter 'local'): run in-process on the
        InferenceBackend contract; target model itself or a separate local model.
      - startlux_local (adapter 'systemone'): HTTP client to an isolated local
        /v1/systemone decision service (llama.cpp + startlux_decision); the
        judge runs there, no GPU backend of this repo is involved.
    Remote providers (api_llm / jev_api) stay blocked.
    """
    judge_cfg = cfg.get("judge") or {}
    provider_name = judge_cfg.get("provider", "local_self")
    provider = (cfg.get("judge_providers") or {}).get(provider_name)
    if provider is None:
        raise ConfigError(f"judge provider '{provider_name}' not registered in judge_providers")
    if provider.get("status") != "implemented":
        raise ConfigError(f"judge provider '{provider_name}' status={provider.get('status')}; blocked")
    adapter = provider.get("adapter")
    spec = {"provider": provider_name, "adapter": adapter}
    if adapter == "local":
        model_key = provider.get("model") or require(cfg, "model")
        spec["model_key"] = model_key
        spec["path"] = resolve_model(cfg, model_key)
        spec["backend_cfg"] = {**dict(cfg.get("backend_cfg") or {}),
                               **dict(provider.get("backend_cfg") or {})}
    elif adapter == "systemone":
        model_key = provider.get("model")
        entry = (cfg.get("models") or {}).get(model_key) or {}
        spec["model_key"] = model_key
        spec["path"] = entry.get("path", "")
        spec["base_url"] = require(cfg, f"judge_providers.{provider_name}.base_url")
        spec["correct_option"] = provider.get("correct_option", "correct")
        spec["hallucinated_option"] = provider.get("hallucinated_option", "hallucinated")
        spec["service"] = provider.get("service") or {}
    else:
        raise ConfigError(f"judge provider '{provider_name}' adapter={adapter!r}; "
                          "only local and systemone adapters are implemented")
    logger.info(f"[judge] provider={provider_name} adapter={adapter} model={spec['model_key']}")
    return spec


def build_judge(cfg, jspec, logger):
    """Construct the judge object from a resolved provider spec."""
    judge_cfg = cfg.get("judge") or {}
    if jspec["adapter"] == "systemone":
        svc = jspec.get("service") or {}
        judge = SystemOneJudge(
            base_url=jspec["base_url"], provider_name=jspec["provider"],
            correct_option=jspec["correct_option"],
            hallucinated_option=jspec["hallucinated_option"],
            prompt_version=judge_cfg.get("prompt_version", "judge-yesno-v2"),
            instructions=judge_cfg.get("systemone_instructions"),
            criteria=judge_cfg.get("systemone_criteria"),
            max_chars=judge_cfg.get("max_chars") or {},
            timeout_s=float(svc.get("timeout_s", 120.0)))
        wait_s = float(svc.get("wait_ready_s", 600.0))
        logger.info(f"[judge] waiting for systemone service {jspec['base_url']} (up to {wait_s:.0f}s)")
        try:
            judge.wait_ready(wait_s)
        except ConnectionError as e:
            raise ConfigError(
                f"systemone judge service not reachable: {e}. Start it with "
                "`bash scripts/setup/startlux_service.sh start`; failing honestly, no fallback judge.")
        return judge, None
    backend = build_backend(cfg.get("backend", "vllm"), jspec["path"],
                            jspec["backend_cfg"], logger)
    tokenizer = load_tokenizer(jspec["path"])
    judge = LocalJudge(backend, tokenizer,
                       prompt_version=judge_cfg.get("prompt_version", "judge-yesno-v2"),
                       reasoning_effort=judge_cfg.get("reasoning_effort"),
                       max_chars=judge_cfg.get("max_chars") or {})
    return judge, backend


VERIFY_JUDGE_CASES = [
    {"case_id": "correct-factual", "question": "What is the capital of France?",
     "answer": "Paris", "gold": ["Paris"], "expected": 0},
    {"case_id": "wrong-factual", "question": "What is the capital of France?",
     "answer": "Berlin", "gold": ["Paris"], "expected": 1},
    {"case_id": "correct-math", "question": "What is 7 times 8?",
     "answer": "Final answer: 56", "gold": ["56"], "expected": 0},
    {"case_id": "wrong-math", "question": "What is 7 times 8?",
     "answer": "Final answer: 54", "gold": ["56"], "expected": 1},
]


def cmd_verify_judge(cfg, logger, run_dir) -> dict:
    """Acceptance of the configured judge provider on canned ground-truth cases
    (works for both local and systemone adapters)."""
    jspec = resolve_judge_model(cfg, logger)
    judge, backend = build_judge(cfg, jspec, logger)
    report = {"provider": jspec["provider"], "adapter": jspec["adapter"],
              "model": jspec["model_key"], "model_path": jspec.get("path", ""),
              "prompt_version": judge.prompt_version, "cases": []}
    if jspec["adapter"] == "local":
        report["final_channel_prefix"] = judge._final_prefix
    gating_failed = []
    for case in VERIFY_JUDGE_CASES:
        r = judge.judge(case["case_id"], case["question"], case["answer"],
                        context="", gold_answers=case["gold"])
        ok = r.hard_verdict == case["expected"]
        gating_failed.append(not ok)
        report["cases"].append({
            "case_id": case["case_id"], "expected": case["expected"],
            "hard_verdict": r.hard_verdict, "continuous_score": r.continuous_score,
            "raw_output": r.raw_output, "ok": bool(ok), "timing_s": r.timing_s,
        })
        logger.info(f"[verify-judge] {case['case_id']}: verdict={r.hard_verdict} "
                    f"expected={case['expected']} cont={r.continuous_score} "
                    f"raw={r.raw_output!r} -> {'OK' if ok else 'FAIL'}")
    cont_ok = all(c["continuous_score"] is not None for c in report["cases"])
    report["checks"] = {
        "hard_verdicts_match_ground_truth": not any(gating_failed),
        "continuous_label_likelihood_available": bool(cont_ok),
    }
    report["status"] = "passed" if not any(gating_failed) else "failed"
    if not cont_ok:
        logger.warning("[verify-judge] continuous label-likelihood unavailable on some cases "
                       "(recorded; hard verdicts remain the label source)")
    if backend is not None:
        backend.close()
    save_json(run_dir / "verify_judge.json", report)
    return report


def cmd_judge(cfg, logger, run_dir, force=False) -> dict:
    store = RunStore(run_dir, cfg["_config_hash_"])
    if store.up_to_date("judge") and not force:
        logger.info("[judge] stage up-to-date; skipping")
        return store.load_stage("judge")
    gen_payload = store.load_stage("generation")
    dataset_payload = load_json(run_dir / "dataset.json")
    ex_by_id = {e["sample_id"]: Example(**e) for e in dataset_payload["examples"]}
    jspec = resolve_judge_model(cfg, logger)
    judge, backend = build_judge(cfg, jspec, logger)

    per_sample = []
    n_no_answer = 0
    n_no_pattern = 0
    # dataset-level answer-format gate (DESIGN 9.5.3): outputs missing the
    # instructed answer marker carry no parseable choice — truncated reasoning or
    # instruction non-compliance is a format error, never a hallucination label
    ds_cfg = (cfg.get("datasets") or {}).get(str(dataset_payload.get("dataset") or "")) or {}
    require_pattern = str(ds_cfg.get("require_pattern") or "")
    # authoritative flag from the generation stage (target-model thinking mode)
    thinking_on = bool((gen_payload.get("thinking") or {}).get("enabled", False))
    t_all = time.time()
    for g in gen_payload["generations"]:
        gen = load_generation(g)
        ex = ex_by_id[gen.sample_id]
        # reference-aware judging sees the POST-THINK answer only; the reasoning
        # trace is the model's own process, not the claim under test
        greedy_text = gen.greedy.text
        _, answer_text = split_think(greedy_text)
        truncated_think = thinking_on and THINK_CLOSE not in greedy_text
        if not answer_text.strip() or truncated_think:
            # thinking output truncated before </think>: judging reasoning ramble
            # (or an empty answer) would fake a hallucination label (DESIGN 9.5.4)
            n_no_answer += 1
            per_sample.append({
                "sample_id": gen.sample_id, "judge_name": "vllm-local",
                "hard_verdict": None, "continuous_score": None,
                "score_type": "label_likelihood", "raw_output": "",
                "coverage": "no_answer_after_think",
                "reason": ("greedy output truncated inside think (no </think>); "
                           "no judgeable answer"),
                "prompt_version": judge.prompt_version, "model_revision": "",
                "timing_s": 0.0})
            logger.warning(f"[judge] {gen.sample_id}: SKIPPED (no answer after think; "
                           "truncation excluded from labels)")
            continue
        if require_pattern and not re.search(require_pattern, answer_text, re.I):
            n_no_pattern += 1
            per_sample.append({
                "sample_id": gen.sample_id, "judge_name": "vllm-local",
                "hard_verdict": None, "continuous_score": None,
                "score_type": "label_likelihood", "raw_output": "",
                "coverage": "no_parseable_answer",
                "reason": (f"answer missing required marker /{require_pattern}/ "
                           "(truncated or instruction non-compliance); excluded "
                           "from labels, not counted as hallucination"),
                "prompt_version": judge.prompt_version, "model_revision": "",
                "timing_s": 0.0})
            logger.warning(f"[judge] {gen.sample_id}: SKIPPED (no parseable answer "
                           f"marker /{require_pattern}/; format exclusion, "
                           "DESIGN 9.5.3)")
            continue
        r = judge.judge(gen.sample_id, ex.question, answer_text,
                        context=ex.context or "", gold_answers=list(ex.gold_answers or []))
        per_sample.append(r.__dict__)
        logger.info(f"[judge] {gen.sample_id}: verdict={r.hard_verdict} "
                    f"score={r.continuous_score} ({r.timing_s:.2f}s) raw={r.raw_output!r}")
    payload = {
        "judge_model": jspec["model_key"], "judge_model_path": jspec.get("path", ""),
        "judge_provider": jspec["provider"], "judge_adapter": jspec["adapter"],
        "prompt_version": judge.prompt_version,
        "reference_aware": True,
        "answer_view": "post-think",
        "n_skipped_no_answer": n_no_answer,
        "n_skipped_no_parseable_answer": n_no_pattern,
        "backend": "systemone-http" if jspec["adapter"] == "systemone" else backend.name,
        "per_sample": per_sample,
        "timing": {"total_s": time.time() - t_all},
    }
    store.save_stage("judge", payload)
    if backend is not None:
        backend.close()
    logger.info("[judge] done")
    return payload
