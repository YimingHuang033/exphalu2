"""Perf benchmark: per-stage timing and token counts on the configured dataset sample."""
from __future__ import annotations

import time

import numpy as np

from .logging_utils import save_json
from .orchestrate import eos_ids, load_tokenizer, peak_mem_mib, resolve_model
from .backends import build_backend
from .cache import RunStore
from .config_loader import require
from .data.adapters import get_dataset_iterator, DATASET_STATUS
from .scoring import render_prompt, sample_and_score


def cmd_perf(cfg, logger, run_dir) -> dict:
    import pandas as pd

    model_path = resolve_model(cfg)
    backend = build_backend(cfg.get("backend", "vllm"), model_path, cfg.get("backend_cfg", {}), logger)
    tokenizer = load_tokenizer(model_path)
    ds_name = require(cfg, "dataset")
    if DATASET_STATUS.get(ds_name) != "implemented":
        from .config_loader import ConfigError

        raise ConfigError(f"dataset '{ds_name}' not available (status={DATASET_STATUS.get(ds_name)})")
    examples = list(get_dataset_iterator(ds_name, cfg, int(cfg.get("num_samples", 4))))
    sampling_cfg = dict(cfg.get("sampling") or {})
    sampling_cfg.setdefault("stop_token_ids", list(eos_ids(tokenizer)))
    k = int(sampling_cfg.get("k", 5))

    rows = []
    for ex in examples:
        prompt_text, prompt_ids, _ = render_prompt(ex, tokenizer)
        t0 = time.time()
        gen, _ = sample_and_score(backend, ex, tokenizer, sampling_cfg, k=k,
                                  sample_seed=sampling_cfg.get("seed"))
        t_gen = time.time() - t0
        n_gen_tokens = len(gen.greedy.token_ids) + sum(len(s.token_ids) for s in gen.samples)

        t0 = time.time()
        backend.score(gen.prompt_token_ids, gen.greedy.token_ids)
        t_score = time.time() - t0

        t0 = time.time()
        backend.replay_last_hidden(gen.prompt_token_ids, gen.greedy.token_ids)
        t_replay_one = time.time() - t0

        rows.append({
            "sample_id": ex.sample_id,
            "prompt_tokens": len(prompt_ids),
            "gen_time_s": t_gen, "gen_tokens": n_gen_tokens,
            "gen_tok_per_s": n_gen_tokens / t_gen,
            "score_time_s": t_score,
            "replay_one_time_s": t_replay_one,
            "replay_all_k_est_s": t_replay_one * (k + 1),
            "backend": backend.name,
        })
        logger.info(f"[perf] {ex.sample_id}: gen={t_gen:.2f}s score={t_score:.2f}s "
                    f"replay={t_replay_one:.2f}s")

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
    store = RunStore(run_dir, cfg["_config_hash_"])
    save_json(run_dir / "perf_summary.json", summary)
    backend.close()
    logger.info(f"[perf] summary: {summary}")
    return summary
