from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .backends import build_backend
from .cache import RunStore
from .config_loader import load_config, require, ConfigError
from .logging_utils import setup_logging, results_dir
from .orchestrate import (cmd_verify_backend, cmd_generate, cmd_detect, cmd_judge,
                          env_fingerprint, resolve_model, load_tokenizer)
from .evaluate_stage import cmd_evaluate
from .types import new_run_id


def _prepare(args, category_default):
    category = getattr(args, "category", None) or category_default
    cfg = load_config(args.config, strict_env=bool(getattr(args, "strict_env", False)))
    if getattr(args, "model", None):
        cfg["model"] = args.model
    if getattr(args, "dataset", None):
        cfg["dataset"] = args.dataset
    if getattr(args, "num_samples", None):
        cfg["num_samples"] = args.num_samples
    if getattr(args, "backend", None):
        cfg["backend"] = args.backend
    if getattr(args, "run_id", None):
        run_id = args.run_id
    else:
        run_id = new_run_id(category)
    run_dir = results_dir(category, run_id)
    logger, log_file, run_id = setup_logging(category, args.command, run_id)
    logger.info(f"command={args.command} category={category} run_id={run_id}")
    logger.info(f"config={args.config} config_hash={cfg['_config_hash_']} log={log_file}")
    store = RunStore(run_dir, cfg["_config_hash_"])
    store.snapshot_config(cfg, env_fingerprint())
    return cfg, logger, run_dir, store


def main(argv=None):
    p = argparse.ArgumentParser(prog="reppl2", description="RePPL 2.0 experiment CLI")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp, category):
        sp.add_argument("--config", required=True, help="config/<name>.yaml")
        sp.add_argument("--category", default=category, help="results/log subfolder")
        sp.add_argument("--run-id", default=None)
        sp.add_argument("--model", default=None)
        sp.add_argument("--dataset", default=None)
        sp.add_argument("--num-samples", type=int, default=None)
        sp.add_argument("--backend", default=None, choices=["vllm", "transformers", "sglang"])
        sp.add_argument("--strict-env", action="store_true")

    sp = sub.add_parser("verify-backend", help="real-GPU acceptance of a backend")
    add_common(sp, "smoke")
    sp = sub.add_parser("generate", help="greedy + K sampled generations")
    add_common(sp, "generation_eval")
    sp.add_argument("--force", action="store_true")
    sp = sub.add_parser("detect", help="RePPL A/B + baselines from generation stage")
    add_common(sp, "generation_eval")
    sp.add_argument("--force", action="store_true")
    sp = sub.add_parser("judge", help="local LLM judge over greedy answers")
    add_common(sp, "generation_eval")
    sp.add_argument("--force", action="store_true")
    sp = sub.add_parser("evaluate", help="metrics table (CSV) from detection/judge stages")
    add_common(sp, "generation_eval")
    sp.add_argument("--label-source", default="auto", choices=["auto", "judge", "em_gold"])
    sp = sub.add_parser("perf", help="stage-level timing/memory benchmark")
    add_common(sp, "perf")

    args = p.parse_args(argv)
    cfg, logger, run_dir, store = _prepare(args, {"verify-backend": "smoke"}.get(args.command, "generation_eval"))

    if args.command == "verify-backend":
        report = cmd_verify_backend(cfg, logger, run_dir)
        if report["status"] != "passed":
            logger.error("verify-backend FAILED")
            return 1
        return 0
    if args.command == "generate":
        cmd_generate(cfg, logger, run_dir, force=args.force)
        return 0
    if args.command == "detect":
        cmd_detect(cfg, logger, run_dir, force=args.force)
        return 0
    if args.command == "judge":
        cmd_judge(cfg, logger, run_dir, force=args.force)
        return 0
    if args.command == "evaluate":
        cmd_evaluate(cfg, logger, run_dir, label_source=args.label_source)
        return 0
    if args.command == "perf":
        from .perf_stage import cmd_perf

        cmd_perf(cfg, logger, run_dir)
        return 0
    logger.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
