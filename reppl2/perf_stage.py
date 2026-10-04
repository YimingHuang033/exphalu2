"""Perf benchmark: per-stage timing and token counts on the configured dataset sample.

Two-phase structure mirrors the engine constraint: phase 1 (generate runner) does
sampling + scoring for all examples, phase 2 (pooling runner) does state replays.
"""
from __future__ import annotations

import time

import numpy as np

from .backends import build_backend
from .cache import RunStore
from .config_loader import require, ConfigError
from .data.adapters import get_dataset_iterator, DATASET_STATUS
from .logging_utils import save_json
from .orchestrate import eos_ids, load_tokenizer, peak_mem_mib, resolve_model
from .scoring import render_prompt, sample_and_score


def cmd_perf(cfg, logger, run_dir) -> dict:
    import pandas as pd

    import torch

    free_b, total_b = torch.cuda.mem_get_info()
    if free_b < total_b * 0.5:
        logger.warning(f"[perf] GPU busy: {free_b/(1024**3):.1f}/{total_b/(1024**3):.1f} GiB free; "
                       "waiting up to 180s for a previous engine core to exit")
        deadline = time.time() + 180
        while time.time() < deadline:
            free_b, total_b = torch.cuda.mem_get_info()
            if free_b >= total_b * 0.5:
                break
            time.sleep(3.0)
        if free_b < total_b * 0.5:
            raise RuntimeError(
                f"GPU memory still busy ({free_b/(1024**3):.1f} GiB free); "
                "another engine core is resident; aborting instead of reporting fake numbers")

    model_path = resolve_model(cfg)
    backend = build_backend(cfg.get("backend", "vllm"), model_path, cfg.get("backend_cfg", {}), logger)
    tokenizer = load_tokenizer(model_path)
    ds_name = require(cfg, "dataset")
    if DATASET_STATUS.get(ds_name) != "implemented":
        raise ConfigError(f"dataset '{ds_name}' not available (status={DATASET_STATUS.get(ds_name)})")
    examples = list(get_dataset_iterator(ds_name, cfg, int(cfg.get("num_samples", 4))))
    sampling_cfg = dict(cfg.get("sampling") or {})
    sampling_cfg.setdefault("stop_token_ids", list(eos_ids(tokenizer)))
    k = int(sampling_cfg.get("k", 5))

    prompts = {}
    for ex in examples:
        prompt_text, prompt_ids, _ = render_prompt(ex, tokenizer)
        prompts[ex.sample_id] = prompt_ids

    rows = []
    greedy_ids = {}

    # phase 1: generate runner — sampling + scoring
    for ex in examples:
        t0 = time.time()
        gen, _ = sample_and_score(backend, ex, tokenizer, sampling_cfg, k=k,
                                  sample_seed=sampling_cfg.get("seed"))
        t_gen = time.time() - t0
        n_gen_tokens = len(gen.greedy.token_ids) + sum(len(s.token_ids) for s in gen.samples)
        greedy_ids[ex.sample_id] = gen.greedy.token_ids

        t0 = time.time()
        backend.score(prompts[ex.sample_id], gen.greedy.token_ids)
        t_score = time.time() - t0

        rows.append({
            "sample_id": ex.sample_id,
            "prompt_tokens": len(prompts[ex.sample_id]),
            "gen_time_s": t_gen, "gen_tokens": n_gen_tokens,
            "gen_tok_per_s": n_gen_tokens / t_gen,
            "score_time_s": t_score,
        })
        logger.info(f"[perf:phase1] {ex.sample_id}: gen={t_gen:.2f}s score={t_score:.2f}s")

    backend.release_for_replay()

    # phase 2: pooling runner — one replay per sample (greedy sequence)
    for ex in examples:
        t0 = time.time()
        backend.replay_last_hidden(prompts[ex.sample_id], greedy_ids[ex.sample_id])
        t_replay = time.time() - t0
        for row in rows:
            if row["sample_id"] == ex.sample_id:
                row["replay_one_time_s"] = t_replay
                row["replay_all_k_est_s"] = t_replay * (k + 1)
        logger.info(f"[perf:phase2] {ex.sample_id}: replay={t_replay:.2f}s")

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "perf.csv", index=False)
    summary = {
        "model": require(cfg, "model"), "backend": backend.name,
        "k": k, "n_samples": len(rows),
        "gen_p50_s": float(np.percentile(df["gen_time_s"], 50)),
        "gen_p95_s": float(np.percentile(df["gen_time_s"], 95)),
        "score_p50_s": float(np.percentile(df["score_time_s"], 50)),
        "replay_p50_s": float(np.percentile(df["replay_one_time_s"], 50)),
        "peak_mem_mib": peak_mem_mib(),
        "perf_csv": str(run_dir / "perf.csv"),
    }
    save_json(run_dir / "perf_summary.json", summary)
    backend.close()
    logger.info(f"[perf] summary: {summary}")
    return summary
