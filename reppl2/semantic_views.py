"""Strict, label-blind semantic unit/view contracts for C and D."""
from __future__ import annotations

import re
from collections import Counter

from .scoring import split_think


def split_options(example):
    question = example["question"]
    if example["task"] != "mcq":
        return question, [], ""
    # The cloud's saved questions include a rendered option block. New adapters
    # need not be rerun. Fail rather than silently treating letters as semantics.
    hits = list(re.finditer(r"(?m)^([A-L])\.\s+", question))
    n = int(example.get("meta", {}).get("n_options", len(hits)))
    if len(hits) != n or n < 2 or [h[1] for h in hits] != list("ABCDEFGHIJKL"[:n]):
        raise ValueError("ambiguous MCQ option block")
    options = [question[h.end():hits[i + 1].start() if i + 1 < n else len(question)].strip()
               for i, h in enumerate(hits)]
    if any(not c for c in options):
        raise ValueError("empty MCQ option")
    return question[:hits[0].start()].strip(), options, question[hits[0].start():]


def public_question(example):
    """Explicit allowlist: never serialize Example/gold/meta into a model prompt."""
    stem, _, block = split_options(example)
    return {"question_stem": stem, "options_block": block}


def canonical_answer(sample, options, thinking=False):
    text = sample["text"]
    if thinking and "</think>" not in text:
        raise ValueError("truncated thinking: missing final answer")
    if sample.get("finish_reason") == "length":
        raise ValueError("length-truncated answer")
    answer = split_think(text)[1].strip()
    explicit = re.findall(r"(?im)^\s*(?:final answer|answer)\s*:\s*(.+)$", answer)
    answer = explicit[-1].strip() if explicit else answer
    if not answer or "\n" in answer:
        raise ValueError("missing/ambiguous short answer")
    if options:
        match = re.fullmatch(r"\(?([A-L])\)?[.\s]*", answer)
        if match and ord(match[1]) - ord("A") < len(options):
            i = ord(match[1]) - ord("A")
            return {"id": f"option-{i}", "text": options[i]}
        matches = [i for i, value in enumerate(options) if value.casefold() == answer.casefold()]
        if len(matches) == 1:
            return {"id": f"option-{matches[0]}", "text": options[matches[0]]}
        raise ValueError("unparseable MCQ answer")
    # Conservative only: no gold alias list; preserve numbers, signs and units.
    identity = " ".join(answer.casefold().split()).rstrip(".!?")
    return {"id": identity, "text": answer}


def choose_competitor(target, samples, options, seed):
    import random
    alternatives = [c for c in samples if c["id"] != target["id"]]
    if alternatives:
        counts = Counter(c["id"] for c in alternatives)
        best = sorted(counts, key=lambda key: (-counts[key], key))[0]
        return next(c for c in alternatives if c["id"] == best)
    if options:
        candidates = [{"id": f"option-{i}", "text": t} for i, t in enumerate(options)
                      if f"option-{i}" != target["id"]]
        return random.Random(seed).choice(candidates)
    return None


def _units(stem, units, max_units):
    if not isinstance(units, list) or not 1 <= len(units) <= max_units:
        raise ValueError("invalid number of semantic units")
    result, spans, ids = [], [], set()
    for unit in units:
        uid, text = unit.get("unit_id"), unit.get("text")
        if not isinstance(uid, str) or not uid or uid in ids:
            raise ValueError("unit_id must be unique nonempty string")
        if not isinstance(text, str) or not text.strip() or stem.count(text) != 1:
            raise ValueError("unit text must occur exactly once in the stem")
        start, end = stem.index(text), stem.index(text) + len(text)
        if any(start < b and end > a for a, b in spans):
            raise ValueError("overlapping semantic units")
        if text.strip() == stem.strip():
            raise ValueError("refuse to mask the entire question")
        spans.append((start, end))
        ids.add(uid)
        result.append({"unit_id": uid, "role": str(unit.get("role", "unspecified")),
                       "text": text, "start": start, "end": end})
    return result


def validate_annotation(question, annotation, config):
    """Mechanical integrity only; does not certify semantic equivalence."""
    stem = question["question_stem"]
    units = _units(stem, annotation["units"], config["max_units"])
    base_ids = [u["unit_id"] for u in units]
    views = [{"view_id": "original", "question_stem": stem, "units": units,
              "validation": "original"}]
    rejected = []
    seen = {stem}
    numbers = Counter(re.findall(r"[-+]?\d+(?:\.\d+)?", stem))
    for i, view in enumerate(annotation.get("views", [])):
        try:
            text = view["question_stem"]
            if not isinstance(text, str) or not text.strip() or text in seen:
                raise ValueError("empty/duplicate view")
            if Counter(re.findall(r"[-+]?\d+(?:\.\d+)?", text)) != numbers:
                raise ValueError("changed numeric constants")
            vus = _units(text, view["units"], config["max_units"])
            by_id = {u["unit_id"]: u for u in vus}
            if set(by_id) != set(base_ids):
                raise ValueError("view semantic unit correspondence mismatch")
            for u in units:
                if by_id[u["unit_id"]]["role"] != u["role"]:
                    raise ValueError("semantic unit role changed")
            # Exact entity/number preservation is stronger than fuzzy identity.
            for u in units:
                if u["role"] in config["protected_roles"] and by_id[u["unit_id"]]["text"] != u["text"]:
                    raise ValueError("protected entity/quantity changed")
            views.append({"view_id": f"view-{i + 1}", "question_stem": text,
                          "units": [by_id[key] for key in base_ids], "validation": "structural_only"})
            seen.add(text)
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({"view_index": i, "reason": str(exc)})
    if len(views) > config["views"]:
        raise ValueError("too many views; refusing silent truncation")
    return {"units": units, "views": views, "rejected_views": rejected,
            "options_block": question["options_block"]}


def edited_question(view, unit, replacement, options_block):
    text = view["question_stem"]
    if not replacement or replacement == unit["text"]:
        raise ValueError("empty/no-op replacement")
    if text[unit["start"]:unit["end"]] != unit["text"]:
        raise ValueError("unit span mismatch")
    changed = text[:unit["start"]] + replacement + text[unit["end"]:]
    return changed + ("\n" + options_block if options_block else "")


def full_question(view, options_block):
    return view["question_stem"] + ("\n" + options_block if options_block else "")
