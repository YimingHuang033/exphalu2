"""Separate C/D/E stages operating on frozen cloud generations.

Run through scripts/generation_eval/run_cde.sh, not direct experimental commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .cache import content_hash
from .config_loader import load_config, require
from .logging_utils import load_json, save_json, setup_logging, results_dir
from .methods.reppl_c import reppl_c_inner, unit_vector
from .methods.reppl_d import reppl_d_inner
from .methods.reppl_e import uncertainty_features, grouped_oof
from .semantic_views import (public_question, split_options, canonical_answer,
                             choose_competitor, validate_annotation, edited_question, full_question)
from .semantic_readout import SemanticReadout
from .baselines.probability import outer_perplexity_risk

VERSION = "cde-v1-native-propagation"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source(cfg):
    path = Path(require(cfg, "cde.source_run")).expanduser().resolve()
    dataset, generation = load_json(path / "dataset.json"), load_json(path / "generation.json")
    if dataset["dataset"] not in ("supergpqa", "popqa"):
        raise ValueError("CDE currently supports SuperGPQA and PopQA only")
    examples = {r["sample_id"]: r for r in dataset["examples"]}
    ids = [g["sample_id"] for g in generation["generations"]]
    if len(examples) != len(dataset["examples"]) or len(set(ids)) != len(ids):
        raise ValueError("duplicate source IDs")
    if set(ids) - set(examples):
        raise ValueError("generation without dataset example")
    limit = int(cfg["cde"]["limit"])
    if limit < 0:
        raise ValueError("negative limit")
    gens = generation["generations"][:limit] if limit else generation["generations"]
    if not gens:
        raise ValueError("frozen source contains no generations")
    source = {"path": str(path), "dataset_sha256": _sha(path / "dataset.json"),
              "generation_sha256": _sha(path / "generation.json"), "version": VERSION,
              "generation_model": generation.get("model"),
              "generation_model_path": generation.get("model_path"),
              "thinking": generation.get("thinking", {})}
    return path, dataset, examples, generation, gens, source


def _output(cfg, source):
    run_id = require(cfg, "cde.run_id")
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a simple directory name")
    category = require(cfg, "cde.category")
    if category not in ("generation_eval", "smoke"):
        raise ValueError("CDE category must be generation_eval or smoke")
    out = results_dir(category, run_id)
    if out.resolve() == Path(source["path"]):
        raise ValueError("CDE output must not overwrite source run")
    identity = content_hash({"config_hash": cfg["_config_hash_"], "source": source})
    snapshot = out / "config_snapshot.json"
    if snapshot.exists() and load_json(snapshot).get("cde_identity") != identity:
        raise ValueError("run identity changed; choose a new run_id instead of mixing caches")
    save_json(snapshot, {**cfg, "cde_identity": identity, "source": source})
    return out, identity


def _backend(cfg, generation, logger):
    from .backends import build_backend
    from .orchestrate import load_tokenizer, resolve_model
    name = require(cfg, "backend")
    if name not in ("vllm", "sglang"):
        raise ValueError("CDE cannot use Transformers model forward")
    path = resolve_model(cfg)
    # The core experiment uses the same generator, not an undeclared external observer.
    if cfg["model"] != generation["model"]:
        raise ValueError("observer model key must equal frozen generator model")
    if Path(path).resolve() != Path(generation["model_path"]).resolve():
        raise ValueError("observer path differs from frozen generator; audit model before relocation")
    tokenizer = load_tokenizer(path)  # tokenizer only; no Transformers model forward
    bcfg = dict(cfg.get("backend_cfg", {}))
    bcfg["model_impl"] = "vllm" if name == "vllm" else "sglang"
    backend = build_backend(name, path, bcfg, logger)
    caps = backend.capabilities()
    if not caps.get("token_last_hidden") or not caps.get("generation"):
        backend.close()
        raise ValueError(f"native capabilities unavailable: {caps}")
    return backend, tokenizer


def json_request(backend, tokenizer, instruction, data, config):
    messages = [{"role": "system", "content": config["preparation_system"]},
                {"role": "user", "content": instruction + "\n\n" + json.dumps(data, ensure_ascii=False)}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **config["preparation_chat_kwargs"])
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) + config["prepare_sampling"]["max_new_tokens"] > config["max_prepare_context"]:
        raise ValueError("preparation request exceeds configured context budget")
    t0 = time.monotonic()
    result = backend.sample(ids, config["prepare_sampling"], n=1, seed=config["seed"])
    if result["finish_reasons"][0] == "length":
        raise ValueError("truncated preparation JSON")
    raw = result["texts"][0].strip()
    if raw.startswith("```json") and raw.endswith("```"):
        raw = raw[7:-3].strip()
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("expected JSON object")
    return obj, {"input_tokens": len(ids), "output_tokens": len(result["token_ids"][0]),
                 "seconds": time.monotonic() - t0}


def prepare_one(example, request, config):
    public = public_question(example)
    proposal, cost = request(config["annotation_prompt"],
                             {**public, "max_units": config["max_units"],
                              "n_paraphrases": config["views"] - 1})
    annotation = validate_annotation(public, proposal, config)
    checks, costs, accepted = [], [cost], [annotation["views"][0]]
    for view in annotation["views"][1:]:
        try:
            check, cost = request(config["validation_prompt"],
                                  {"original": public["question_stem"], "paraphrase": view["question_stem"],
                                   "original_units": annotation["units"], "paraphrase_units": view["units"],
                                   "options_block": public["options_block"]})
        except (ValueError, KeyError, TypeError) as exc:
            annotation["rejected_views"].append({"view_id": view["view_id"], "reason": str(exc)})
            continue
        costs.append(cost)
        checks.append({"view_id": view["view_id"], "check": check})
        if check.get("equivalent") is True and check.get("units_correspond") is True:
            view["validation"] = "model_checked_not_human_gold"
            accepted.append(view)
        else:
            annotation["rejected_views"].append({"view_id": view["view_id"], "reason": "semantic check rejected"})
    annotation["views"] = accepted
    return {"sample_id": example["sample_id"], "question_hash": content_hash(public),
            "status": "ok", "annotation": annotation, "validation_checks": checks,
            "cost": costs, "human_reviewed": False}


def _stage_rows(path, identity):
    if not path.exists():
        return {}
    payload = load_json(path)
    if payload.get("identity") != identity:
        raise ValueError("stage identity mismatch")
    rows = payload["per_sample"]
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate stage IDs")
    return {r["sample_id"]: r for r in rows}


def prepare_stage(cfg, logger, out, identity, examples, generation, gens, force=False):
    path = out / "cde_prepared.json"
    rows = {} if force else _stage_rows(path, identity)
    if all(g["sample_id"] in rows for g in gens):
        return
    # Changed preparation invalidates downstream identity even if config is unchanged.
    backend = tokenizer = None
    try:
        backend, tokenizer = _backend(cfg, generation, logger)
        for g in gens:
            sid = g["sample_id"]
            if sid in rows:
                continue
            try:
                rows[sid] = prepare_one(examples[sid],
                    lambda prompt, data: json_request(backend, tokenizer, prompt, data, cfg["cde"]), cfg["cde"])
            except Exception as exc:
                rows[sid] = {"sample_id": sid, "status": "invalid", "reason": str(exc)}
                logger.warning("prepare %s: %s", sid, exc)
            save_json(path, {"identity": identity, "per_sample": list(rows.values())})
            logger.info("prepared %s: %s", sid, rows[sid]["status"])
    finally:
        if backend is not None:
            backend.close()


def _result(inner, outer, config):
    if inner is None:
        return {"risk": None, "inner": None, "outer": outer, "validity": "low_signal"}
    return {"risk": float((inner + config["epsilon"]) * outer), "inner": float(inner),
            "outer": float(outer), "validity": "ok"}


def detect_one_cde(example, gen, prepared, readout, config, eos, thinking=False):
    """Pure per-example orchestration, testable with a native-contract fake backend."""
    methods = {name: {"risk": None, "inner": None, "validity": "invalid", "reason": "not computed"}
               for name in ("reppl-c", "reppl-d", "reppl-c-inner", "reppl-d-inner")}
    row = {"sample_id": gen["sample_id"], "methods": methods, "propagation": {}}
    outer = outer_perplexity_risk(gen["greedy"]["token_logprobs"], eos, gen["greedy"]["token_ids"])
    methods["outer-perplexity"] = {"risk": outer, "validity": "ok"}
    if prepared.get("status") != "ok":
        raise ValueError(f"preparation unavailable: {prepared.get('reason')}")
    if prepared["question_hash"] != content_hash(public_question(example)):
        raise ValueError("prepared question mismatch")
    ann = prepared["annotation"]
    # Revalidate hand-edited artifacts before indexing spans or accepting views.
    raw = {"units": ann["units"], "views": ann["views"][1:]}
    checked = validate_annotation(public_question(example), raw, config)
    if checked["rejected_views"]:
        raise ValueError("prepared view no longer passes structural validation")
    views = checked["views"]
    accepted = {v["question_stem"]: v for v in ann["views"]}
    views = [v for v in views if v["view_id"] == "original" or
             accepted[v["question_stem"]].get("validation") == "model_checked_not_human_gold" or
             (prepared.get("human_reviewed") is True and accepted[v["question_stem"]].get("validation") == "human")]
    if config["require_human_review"] and prepared.get("human_reviewed") is not True:
        raise ValueError("preparation awaits human semantic review")
    _, options, block = split_options(example)
    target = canonical_answer(gen["greedy"], options, thinking)
    row.update({"frozen_target": target, "original_greedy_text": gen["greedy"]["text"],
                "units": checked["units"], "views": views, "edits": []})

    def responses(view_list, candidates):
        ds, base = [], []
        for view, candidate in zip(view_list, candidates):
            orig = readout.observe(full_question(view, block), candidate["text"])
            base.append(orig)
            changes = []
            for u in view["units"]:
                replacement = config["replacements"].get(u["role"], config["replacements"]["default"])
                edited = edited_question(view, u, replacement, block)
                edit = {"view_id": view["view_id"], "unit_id": u["unit_id"],
                        "replacement": replacement, "edited_question": edited}
                if edit not in row["edits"]:
                    row["edits"].append(edit)
                changes.append(orig - readout.observe(edited, candidate["text"]))
            ds.append(changes)
        return np.asarray(ds), np.asarray(base)

    try:
        samples, rejected_samples = [], []
        for k, sample in enumerate(gen["samples"]):
            try:
                samples.append(canonical_answer(sample, options, thinking))
            except ValueError as exc:
                rejected_samples.append({"k": k, "reason": str(exc)})
        row["c_sampling_coverage"] = {"n_total": len(gen["samples"]),
                                      "n_valid": len(samples), "rejected": rejected_samples}
        if len(samples) < 2:
            raise ValueError("C requires >=2 valid sampled answers")
        competitor = choose_competitor(target, samples, options, config["seed"])
        if competitor is None:
            raise ValueError("C unavailable: no nonidentical PopQA competitor (no gold fallback)")
        direction = unit_vector(readout.concept(target["text"]) - readout.concept(competitor["text"]),
                                config["vector_floor"])
        d, _ = responses([views[0]] * len(samples), samples)
        stats = reppl_c_inner(d, direction, config["tau_c"], config["min_signal_c"])
        row["propagation"]["c"] = stats
        row["contrast_candidate"] = competitor
        row["candidate_equivalence"] = "MCQ option ID; PopQA literal identity only, aliases not certified"
        row["unique_sample_answers"] = len({s["id"] for s in samples})
        seed = config["seed"] + int(content_hash(gen["sample_id"])[:8], 16)
        random_direction = np.random.default_rng(seed).normal(size=d.shape[-1])
        random_stats = reppl_c_inner(d, random_direction, config["tau_c"], config["min_signal_c"])
        methods["c-random-direction-inner"] = {"risk": random_stats["inner"], "validity": random_stats["validity"]}
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        row["propagation"]["c"] = {"validity": "invalid", "inner": None, "reason": str(exc)}
    try:
        if len(views) < config["min_views"]:
            raise ValueError("D requires enough semantically validated views")
        d, base = responses(views, [target] * len(views))
        stats = reppl_d_inner(d, config["tau_d"], config["min_signal_d"])
        row["propagation"]["d"] = stats
        methods["d-norm-only-inner"] = {"risk": stats["norm_only_inner"], "validity": stats["validity"]}
        output_stats = reppl_d_inner(base[:, None, :], config["tau_d"], 0.0)
        methods["d-output-variance-inner"] = {"risk": output_stats["inner"], "validity": output_stats["validity"]}
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        row["propagation"]["d"] = {"validity": "invalid", "inner": None, "reason": str(exc)}
    for letter in ("c", "d"):
        stats = row["propagation"][letter]
        res = _result(stats.get("inner"), outer, config)
        if stats["validity"] != "ok":
            res.update(validity=stats["validity"], reason=stats.get("reason", "below signal floor"))
        methods[f"reppl-{letter}"] = res
        methods[f"reppl-{letter}-inner"] = {**res, "risk": res["inner"]}
        methods[f"reppl-{letter}-gentle"] = {**res, "risk": None if res["inner"] is None
                                            else outer * (1 + config["gentle_lambda"] * res["inner"])}
    row["cost"] = readout.cost()
    return row


def detect_stage(cfg, logger, out, identity, examples, generation, gens, force=False):
    prep_path, path = out / "cde_prepared.json", out / "cde_detection.json"
    prepared = _stage_rows(prep_path, identity)
    detection_identity = content_hash({"identity": identity, "prepared_sha": _sha(prep_path)})
    rows = {} if force else _stage_rows(path, detection_identity)
    if all(g["sample_id"] in rows for g in gens):
        return
    backend = tokenizer = None
    try:
        from .orchestrate import eos_ids
        backend, tokenizer = _backend(cfg, generation, logger)
        backend.release_for_replay()
        eos = eos_ids(tokenizer)
        for g in gens:
            sid = g["sample_id"]
            if sid in rows:
                continue
            t0 = time.monotonic()
            try:
                readout = SemanticReadout(backend, tokenizer, cfg["cde"])
                rows[sid] = detect_one_cde(examples[sid], g, prepared.get(sid, {}), readout,
                    cfg["cde"], eos, bool(generation.get("thinking", {}).get("enabled", False)))
                n_valid = sum(rows[sid]["methods"][m]["validity"] == "ok"
                              for m in ("reppl-c", "reppl-d"))
                rows[sid]["status"] = "ok" if n_valid == 2 else "partial" if n_valid else "invalid"
            except Exception as exc:
                methods = {m: {"risk": None, "validity": "invalid", "reason": str(exc)}
                           for m in ("reppl-c", "reppl-d", "reppl-c-inner", "reppl-d-inner")}
                try:
                    outer = outer_perplexity_risk(g["greedy"]["token_logprobs"], eos, g["greedy"]["token_ids"])
                    methods["outer-perplexity"] = {"risk": outer, "validity": "ok"}
                except Exception as outer_exc:
                    methods["outer-perplexity"] = {"risk": None, "validity": "invalid", "reason": str(outer_exc)}
                rows[sid] = {"sample_id": sid, "status": "invalid", "reason": str(exc), "methods": methods}
                logger.warning("CDE %s: %s", sid, exc)
            rows[sid]["elapsed_s"] = time.monotonic() - t0
            save_json(path, {"identity": detection_identity, "parent_identity": identity,
                             "prepared_sha": _sha(prep_path), "per_sample": list(rows.values())})
            logger.info("CDE %s: %s", sid, rows[sid]["status"])
    finally:
        if backend is not None:
            backend.close()


def verified_detection(out, identity):
    payload = load_json(out / "cde_detection.json")
    expected = content_hash({"identity": identity, "prepared_sha": _sha(out / "cde_prepared.json")})
    if payload["identity"] != expected:
        raise ValueError("detection stale after changing preparation")
    return payload


def evaluation_labels(cfg, source_path, examples, generation, gens):
    source = cfg["cde"]["label_source"]
    if source == "em_gold":
        from .evaluate_stage import labels_from_gold
        ds = load_json(source_path / "dataset.json")["dataset"]
        pattern = cfg.get("datasets", {}).get(ds, {}).get("require_pattern", "")
        labels = labels_from_gold(examples, gens, bool(generation.get("thinking", {}).get("enabled", False)), pattern)
        labels = {g["sample_id"]: labels[g["sample_id"]] for g in gens
                  if g["sample_id"] in labels and g["greedy"].get("finish_reason") != "length"}
        fingerprint = content_hash({sid: labels.get(sid) for sid in sorted(examples)})
    elif source == "judge":
        path = source_path / "judge.json"
        payload = load_json(path)
        if not generation.get("_config_hash_") or payload.get("_config_hash_") != generation["_config_hash_"]:
            raise ValueError("judge and frozen generation config hashes must match")
        rows = payload["per_sample"]
        if len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError("duplicate judge sample IDs")
        if any(r.get("hard_verdict") not in (None, 0, 1) for r in rows):
            raise ValueError("judge verdict must be binary, never rounded")
        labels = {r["sample_id"]: int(r["hard_verdict"]) for r in rows if r.get("hard_verdict") is not None}
        fingerprint = _sha(path)
    else:
        raise ValueError("label_source must explicitly be em_gold or judge (no auto)")
    if set(labels.values()) - {0, 1}:
        raise ValueError("labels must be binary")
    return labels, {"label_source": source, "label_hash": fingerprint}


def sample_group(ex):
    if ex.get("meta", {}).get("dataset") == "popqa":
        group = ex["meta"].get("subj_id") or ex["meta"].get("subj")
        if group is None or not str(group).strip():
            raise ValueError("PopQA subject group missing; cannot safely split or bootstrap")
        return f"popqa:{str(group).casefold()}"
    return content_hash(" ".join(ex["question"].casefold().split()))


def train_e_stage(cfg, out, identity, examples, labels, label_meta):
    detection = verified_detection(out, identity)
    mode = cfg["cde"]["e"]["mode"]
    X, y, ids, groups, excluded = [], [], [], [], []
    for row in detection["per_sample"]:
        sid = row["sample_id"]
        stats = row.get("propagation", {})
        features = uncertainty_features(stats.get("c"), stats.get("d"), mode)
        if features is None or sid not in labels:
            excluded.append(sid)
            continue
        group = sample_group(examples[sid])
        X.append(features); y.append(labels[sid]); ids.append(sid); groups.append(group)
    report = grouped_oof(X, y, groups, ids, mode, cfg["cde"]["e"])
    report.update({"detection_identity": detection["identity"],
                   "detection_sha": _sha(out / "cde_detection.json"), **label_meta,
                   "excluded_ids": excluded, "total_n": len(detection["per_sample"])})
    save_json(out / "cde_e_oof.json", report)


def evaluate_stage(cfg, out, identity, labels, label_meta, examples):
    from .evaluation.metrics import evaluate_method
    from sklearn.metrics import roc_auc_score
    detection = verified_detection(out, identity)
    rows = detection["per_sample"]
    methods = sorted({m for row in rows for m in row["methods"]})
    scores = {m: {r["sample_id"]: r["methods"].get(m, {}).get("risk") for r in rows} for m in methods}
    ep = out / "cde_e_oof.json"
    if ep.exists():
        e = load_json(ep)
        if (e["detection_identity"] != detection["identity"] or e["label_hash"] != label_meta["label_hash"]
                or e["label_source"] != label_meta["label_source"]
                or e["detection_sha"] != _sha(out / "cde_detection.json")):
            raise ValueError("E predictions stale or evaluated against different labels")
        scores["reppl-e-inner"] = {r["sample_id"]: r["inner"] for r in e["predictions"]}
        scores["reppl-e"] = {sid: (value + cfg["cde"]["epsilon"]) * scores["outer-perplexity"][sid]
                             for sid, value in scores["reppl-e-inner"].items()}
        scores["reppl-e-gentle"] = {
            sid: (1 + cfg["cde"]["gentle_lambda"] * value) * scores["outer-perplexity"][sid]
            for sid, value in scores["reppl-e-inner"].items()}
    ids = [r["sample_id"] for r in rows if r["sample_id"] in labels]
    y = [labels[sid] for sid in ids]
    summary, paired = {}, {}
    for m, smap in scores.items():
        summary[m] = evaluate_method(m, [smap.get(sid) for sid in ids], y)
        summary[m]["auroc_ci_note"] = "legacy row bootstrap; use paired_outer group bootstrap for comparisons"
        shared = [sid for sid in ids if smap.get(sid) is not None
                  and scores.get("outer-perplexity", {}).get(sid) is not None]
        yy = np.array([labels[sid] for sid in shared])
        if len(set(yy)) < 2:
            paired[m] = {"status": "insufficient_classes", "n_shared": len(shared)}
            continue
        a = np.array([smap[sid] for sid in shared])
        b = np.array([scores["outer-perplexity"][sid] for sid in shared])
        rng = np.random.default_rng(cfg["cde"]["seed"])
        diffs = []
        groups = np.array([sample_group(examples[sid]) for sid in shared])
        group_indices = [np.flatnonzero(groups == g) for g in sorted(set(groups))]
        if len(group_indices) < 2:
            paired[m] = {"status": "insufficient_groups", "n_shared": len(shared), "n_groups": len(group_indices)}
            continue
        for _ in range(cfg["cde"]["bootstrap_repeats"]):
            idx = np.concatenate([group_indices[i] for i in
                                  rng.integers(0, len(group_indices), len(group_indices))])
            if len(set(yy[idx])) == 2:
                diffs.append(roc_auc_score(yy[idx], a[idx]) - roc_auc_score(yy[idx], b[idx]))
        paired[m] = {"n_shared": len(shared), "method_auroc": roc_auc_score(yy, a),
                     "outer_auroc_same_ids": roc_auc_score(yy, b),
                     "delta": roc_auc_score(yy, a) - roc_auc_score(yy, b),
                     "ci95": np.quantile(diffs, [.025, .975]).tolist() if diffs else None,
                     "n_groups": len(group_indices),
                     "bootstrap": "paired subject/question groups; E intervals conditional on fitted OOF models"}
    save_json(out / "cde_evaluation.json", {**label_meta, "n_total": len(rows), "n_labeled": len(ids),
              "positive_rate": float(np.mean(y)) if y else None, "per_method": summary,
              "paired_outer": paired, "detection_sha": _sha(out / "cde_detection.json")})


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "detect", "train-e", "evaluate"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true", help="rerun stage, never alter frozen source")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    source_path, dataset, examples, generation, gens, source = load_source(cfg)
    out, identity = _output(cfg, source)
    logger, _, _ = setup_logging(cfg["cde"]["category"], f"cde-{args.stage}", cfg["cde"]["run_id"])
    logger.info("CDE %s source=%s output=%s", args.stage, source_path, out)
    if args.stage == "prepare":
        prepare_stage(cfg, logger, out, identity, examples, generation, gens, args.force)
    elif args.stage == "detect":
        detect_stage(cfg, logger, out, identity, examples, generation, gens, args.force)
    else:
        labels, label_meta = evaluation_labels(cfg, source_path, examples, generation, gens)
        if args.stage == "train-e":
            train_e_stage(cfg, out, identity, examples, labels, label_meta)
        else:
            evaluate_stage(cfg, out, identity, labels, label_meta, examples)
    logger.info("completed %s", args.stage)


if __name__ == "__main__":
    main()
