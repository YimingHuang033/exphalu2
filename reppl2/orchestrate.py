"""Stage orchestration: verify-backend, generate, detect, judge."""
from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .backends import build_backend
from .cache import RunStore
from .config_loader import require, ConfigError
from .data.adapters import get_dataset_iterator, DATASET_STATUS
from .judges.local import LocalJudge
from .logging_utils import save_json, load_json
from .baselines.registry import METHOD_REGISTRY
from .pipeline import detect_one, load_generation, DEFAULT_METHODS
from .scoring import sample_and_score
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


def resolve_model(cfg: dict) -> str:
    name = require(cfg, "model")
    entry = (cfg.get("models") or {}).get(name)
    if entry is None:
        raise ConfigError(f"model '{name}' not registered in config.models")
    path = entry.get("path")
    if not path or not Path(path).exists():
        raise ConfigError(
            f"model '{name}' path '{path}' does not exist on this machine. "
            "Fix config/ or record the model as blocked; no silent substitution.")
    return path


def eos_ids(tokenizer) -> set:
    ids = set()
    if tokenizer.eos_token:
        ids.update(tokenizer(tokenizer.eos_token, add_special_tokens=False)["input_ids"])
    if getattr(tokenizer, "pad_token_id", None):
        ids.add(tokenizer.pad_token_id)
    return ids


def iter_examples(cfg, logger):
    ds_name = require(cfg, "dataset")
    status = DATASET_STATUS.get(ds_name, "unknown")
    if status != "implemented":
        raise ConfigError(f"dataset '{ds_name}' status={status}; available now: triviaqa, synthetic")
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
    save_json(run_dir / "dataset.json", {
        "dataset": require(cfg, "dataset"),
        "examples": [e.__dict__ for e in examples],
    })
    sampling_cfg = dict(cfg.get("sampling") or {})
    eos = eos_ids(tokenizer)
    sampling_cfg.setdefault("stop_token_ids", list(eos))
    k = int(sampling_cfg.get("k", 5))

    gens, meta = [], []
    t_all = time.time()
    for ex in examples:
        try:
            gen, cost = sample_and_score(backend, ex, tokenizer, sampling_cfg, k=k,
                                         sample_seed=sampling_cfg.get("seed"))
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
                          "sample_texts": [s.text for s in g.samples]} for g in gens]})
    backend.close()
    logger.info(f"[generate] done: {len(gens)} ok / {len(meta)}")
    return payload


def load_entailment(cfg, logger):
    path = ((cfg.get("models") or {}).get("deberta-mnli") or {}).get("path")
    if not path or not Path(path).exists():
        logger.warning("[detect] deberta-mnli path missing; semantic-entropy blocked, "
                       "lexical variant used instead (recorded, not substituted silently).")
        return None
    try:
        import torch
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
    entailment_model = None
    if "semantic-entropy" in methods:
        entailment_model = load_entailment(cfg, logger)

    results = []
    t_all = time.time()
    for g in gen_payload["generations"]:
        gen = load_generation(g)
        t0 = time.time()
        try:
            r = detect_one(backend, tokenizer, cfg, gen, ex_by_id[gen.sample_id],
                           entailment_model=entailment_model, interp=True, logger=logger)
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
    if entailment_model is not None:
        del entailment_model
        if _TORCH_OK:
            torch.cuda.empty_cache()
    logger.info(f"[detect] done: {len(results)} samples")
    return payload


def cmd_judge(cfg, logger, run_dir, force=False) -> dict:
    store = RunStore(run_dir, cfg["_config_hash_"])
    if store.up_to_date("judge") and not force:
        logger.info("[judge] stage up-to-date; skipping")
        return store.load_stage("judge")
    gen_payload = store.load_stage("generation")
    dataset_payload = load_json(run_dir / "dataset.json")
    ex_by_id = {e["sample_id"]: Example(**e) for e in dataset_payload["examples"]}
    model_path = gen_payload["model_path"]
    tokenizer = load_tokenizer(model_path)
    judge_cfg = cfg.get("judge") or {}
    backend = build_backend(cfg.get("backend", "vllm"), model_path,
                            cfg.get("backend_cfg", {}), logger)
    judge = LocalJudge(backend, tokenizer,
                       prompt_version=judge_cfg.get("prompt_version", "judge-yesno-v1"))

    per_sample = []
    t_all = time.time()
    for g in gen_payload["generations"]:
        gen = load_generation(g)
        ex = ex_by_id[gen.sample_id]
        r = judge.judge(gen.sample_id, ex.question, gen.greedy.text, context=ex.context or "")
        per_sample.append(r.__dict__)
        logger.info(f"[judge] {gen.sample_id}: verdict={r.hard_verdict} "
                    f"score={r.continuous_score} ({r.timing_s:.2f}s) raw={r.raw_output!r}")
    payload = {
        "judge_model": require(cfg, "model"), "backend": backend.name,
        "prompt_version": judge.prompt_version,
        "per_sample": per_sample,
        "timing": {"total_s": time.time() - t_all},
    }
    store.save_stage("judge", payload)
    backend.close()
    logger.info("[judge] done")
    return payload
