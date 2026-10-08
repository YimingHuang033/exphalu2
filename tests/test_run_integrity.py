"""CPU regressions for cache lineage, output integrity and evaluation labels."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from reppl2.cache import RunStore
from reppl2.cli import _prepare
from reppl2.config_loader import ConfigError, config_hash
from reppl2.evaluate_stage import labels_from_gold, cmd_evaluate
from reppl2.logging_utils import save_json, load_json
from reppl2.types import Generation, SampledOutput


def generation(text):
    return Generation("s", "prompt", [1], SampledOutput("s", 0, [2], text, [-1]), [], {}).to_dict()


def test_old_cache_version_is_not_reused():
    cfg = {"model": "original"}
    legacy = hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    assert config_hash(cfg) != legacy


@pytest.mark.parametrize("script", ["scripts/smoke/unit_tests.sh", "scripts/setup/test_env.sh",
                                    "scripts/generation_eval/run_pipeline.sh"])
def test_shell_syntax(script):
    root = Path(__file__).resolve().parent.parent
    subprocess.run(["bash", "-n", str(root / script)], check=True)


@pytest.mark.parametrize("field,value", [("model", "new"), ("dataset", "new"),
    ("backend", "transformers"), ("num_samples", 0), ("judge_provider", "new")])
def test_cli_hash_includes_overrides(tmp_path, monkeypatch, field, value):
    import reppl2.cli as cli
    config = tmp_path / "config.yaml"
    config.write_text("_include_base_: false\nmodel: original\nnum_samples: 2\n")
    monkeypatch.setattr(cli, "results_dir", lambda *args: tmp_path)
    monkeypatch.setattr(cli, "setup_logging", lambda *args: (logging.getLogger("test"), tmp_path / "log", "run"))
    monkeypatch.setattr(cli, "env_fingerprint", lambda: {})
    args = argparse.Namespace(config=str(config), command="generate", run_id="run", **{field: value})
    cfg, _, _, store = _prepare(args, "smoke")
    assert cfg["_config_hash_"] == config_hash(cfg) == store.config_hash
    base = {"model": "original", "num_samples": 2}
    assert cfg["_config_hash_"] != config_hash(base)
    assert load_json(tmp_path / "config_snapshot.json")["_config_hash_"] == config_hash(cfg)
    if field == "num_samples":
        assert cfg[field] == 0


def test_stage_config_mismatch_is_rejected(tmp_path):
    RunStore(tmp_path, "old").save_stage("generation", {})
    store = RunStore(tmp_path, "new")
    assert not store.up_to_date("generation")
    with pytest.raises(ConfigError, match="config mismatch"):
        store.load_stage("generation")


@pytest.mark.parametrize("stage,invalidated,retained", [
    ("generation", ["trajectory", "detection", "judge", "evaluation", "interp"], []),
    ("detection", ["evaluation", "interp"], ["generation", "judge"]),
    ("judge", ["evaluation"], ["generation", "detection"]),
])
def test_stage_replacement_invalidates_dependents(tmp_path, stage, invalidated, retained):
    store = RunStore(tmp_path, "hash")
    for name in store.STAGES:
        save_json(store.path(name), {"_config_hash_": "hash"})
    (tmp_path / "eval.csv").write_text("stale")
    store.save_stage(stage, {"_config_hash_": "cannot override", "value": 1})
    assert store.load_stage(stage)["value"] == 1
    assert all(not store.path(name).exists() for name in invalidated)
    assert all(store.up_to_date(name) for name in retained)
    assert not (tmp_path / "eval.csv").exists()


def test_corrupt_cache_is_not_current(tmp_path):
    store = RunStore(tmp_path, "hash")
    store.path("generation").write_text("[]")
    assert not store.up_to_date("generation")


def test_json_nonfinite_values_are_null(tmp_path):
    path = tmp_path / "result.json"
    save_json(path, {"a": float("nan"), "b": [float("inf"), np.float32(-np.inf)],
                     "c": np.array([1, np.nan]), "d": np.bool_(True)})
    def reject(value):
        raise AssertionError(f"nonstandard JSON constant {value}")
    data = json.loads(path.read_text(), parse_constant=reject)
    assert data == {"a": None, "b": [None, None], "c": [1, None], "d": True}


def test_failed_json_write_preserves_previous_file(tmp_path, monkeypatch):
    import reppl2.logging_utils as io
    path = save_json(tmp_path / "result.json", {"old": True})
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(io.json, "dump", fail)
    with pytest.raises(OSError):
        save_json(path, {"new": True})
    assert load_json(path) == {"old": True}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("answer,gold,expected", [
    ("5", "56", 1), ("156", "56", 1), ("-56", "56", 1),
    ("5.6", "56", 1), ("Answer: B", "A", 1), ("Paris.", "Paris", 0),
    ("work\nFinal answer: 56", "56", 0), ("George", "George Orwell", 1),
])
def test_gold_labels_do_not_accept_substrings(answer, gold, expected):
    assert labels_from_gold({"s": {"gold_answers": [gold]}}, [generation(answer)]) == {"s": expected}


@pytest.mark.parametrize("examples", [{}, {"s": {}}, {"s": {"gold_answers": None}}])
def test_missing_gold_is_unlabeled(examples):
    assert labels_from_gold(examples, [generation("answer")]) == {}


def test_require_pattern_labels_matching_answers():
    gen = generation("work\nFinal answer: B")
    labels = labels_from_gold({"s": {"gold_answers": ["A"]}}, [gen],
                              require_pattern=r"final answer\s*:")
    assert labels == {"s": 1}


@pytest.mark.parametrize("answer", [
    "primary cementite consists of",       # truncated mid-reasoning, no marker
    "The correct choice is B.",            # free-form answer without the marker
])
def test_require_pattern_excludes_format_errors(answer):
    # DESIGN 9.5.3: instruction non-compliance / truncation is a format error,
    # never an automatic hallucination label
    assert labels_from_gold({"s": {"gold_answers": ["A"]}}, [generation(answer)],
                            require_pattern=r"final answer\s*:") == {}


def test_require_pattern_case_insensitive():
    gen = generation("FINAL ANSWER: A")
    assert labels_from_gold({"s": {"gold_answers": ["A"]}}, [gen],
                            require_pattern=r"final answer\s*:") == {"s": 0}


def test_cmd_evaluate_applies_dataset_require_pattern(tmp_path):
    store = RunStore(tmp_path, "hash")
    store.save_stage("generation", {"generations": [generation("rambling without marker")]})
    store.save_stage("detection", {"methods_requested": [], "per_sample": []})
    save_json(tmp_path / "dataset.json", {
        "dataset": "supergpqa",
        "examples": [{"sample_id": "s", "gold_answers": ["A"]}]})
    cfg = {"_config_hash_": "hash",
           "datasets": {"supergpqa": {"require_pattern": r"final answer\s*:"}}}
    with pytest.raises(ConfigError, match="no labeled samples"):
        cmd_evaluate(cfg, logging.getLogger("test"), tmp_path, "em_gold")


def test_failed_method_still_appears_in_evaluation(tmp_path):
    store = RunStore(tmp_path, "hash")
    store.save_stage("generation", {"generations": [generation("wrong")]})
    store.save_stage("detection", {"methods_requested": ["reppl-a"],
                                   "per_sample": [{"sample_id": "s", "methods": {}}]})
    save_json(tmp_path / "dataset.json", {"examples": [{"sample_id": "s", "gold_answers": ["right"]}]})
    cmd_evaluate({"_config_hash_": "hash"}, logging.getLogger("test"), tmp_path, "em_gold")
    result = store.load_stage("evaluation")["per_method"]["reppl-a"]
    assert result["n_invalid_scores"] == 1
    assert result["auroc"] is None
    assert result["status"] == "invalid"


@pytest.mark.parametrize("fail_stage", ["", "detect"])
def test_judge_service_starts_after_detection(tmp_path, fail_stage):
    root = Path(__file__).resolve().parent.parent
    for folder in ["scripts/generation_eval", "scripts/setup", "bin", "miniconda3/etc/profile.d"]:
        (tmp_path / folder).mkdir(parents=True)
    script = tmp_path / "scripts/generation_eval/run_pipeline.sh"
    shutil.copyfile(root / "scripts/generation_eval/run_pipeline.sh", script)
    (tmp_path / "miniconda3/etc/profile.d/conda.sh").write_text("conda() { return 0; }\n")
    (tmp_path / "scripts/setup/startlux_service.sh").write_text(
        '[ "$1" = status ] && exit 1\necho "service-$1" >> "$TRACE"\n')
    python = tmp_path / "bin/python"
    python.write_text('#!/bin/bash\necho "$3" >> "$TRACE"\n[ "$3" != "$FAIL_STAGE" ]\n')
    python.chmod(0o755)
    trace = tmp_path / "trace"
    env = {**os.environ, "HOME": str(tmp_path), "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
           "TRACE": str(trace), "FAIL_STAGE": fail_stage}
    result = subprocess.run(["bash", str(script), "cfg", "model", "vllm", "2", "ds", "startlux_local"],
                            env=env, capture_output=True, text=True)
    events = trace.read_text().splitlines()
    assert events[:3] == ["verify-backend", "generate", "detect"]
    if fail_stage:
        assert result.returncode != 0
        assert "service-start" not in events
    else:
        assert result.returncode == 0
        assert events[3:] == ["service-start", "judge", "evaluate", "service-stop"]
