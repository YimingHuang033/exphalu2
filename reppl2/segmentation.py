from __future__ import annotations

import hashlib
import json
import re
from typing import Optional


class SegmentationError(RuntimeError):
    pass


_SENT_SPLIT = None


def _split_sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [p for p in parts if p.strip()]


_CLAUSE_SPLIT = re.compile(r"[,;:，；：]")


def segment_input(example, tokenizer=None, max_units: int = 8) -> list[dict]:
    """Produce input units with char spans. Sources: system (non-factual), question sentences,
    RAG paragraphs. Single-sentence questions without context are split into clause
    fragments so that J>=2; with J=1 the cross-unit softmax is degenerate (w==1 for the
    only unit, Inner==0) and that degenerate case must not pass silently."""
    units: list[dict] = []
    uid = 0
    if example.system_prompt:
        units.append({
            "unit_id": uid, "text": example.system_prompt, "char_start": 0, "char_end": len(example.system_prompt),
            "source": "system", "is_factual": False,
        })
        uid += 1
    if example.context:
        paras = [p for p in example.context.split("\n\n") if p.strip()]
        for p in paras[:max_units]:
            start = example.context.find(p)
            units.append({
                "unit_id": uid, "text": p, "char_start": start, "char_end": start + len(p),
                "source": "rag_paragraph", "is_factual": True,
            })
            uid += 1
    factual_units = []
    for s in _split_sentences(example.question)[:max_units]:
        start = example.question.find(s)
        factual_units.append({
            "unit_id": uid, "text": s, "char_start": start, "char_end": start + len(s),
            "source": "question_sentence", "is_factual": True,
        })
        uid += 1
    if len(factual_units) == 1 and not example.context:
        u = factual_units[0]
        clauses = [c for c in _CLAUSE_SPLIT.split(u["text"]) if c.strip()]
        if len(clauses) >= 2:
            factual_units = []
            cursor = 0
            for c in clauses:
                c = c.strip()
                start = u["text"].find(c, cursor)
                factual_units.append({
                    "unit_id": uid, "text": c,
                    "char_start": u["char_start"] + start,
                    "char_end": u["char_start"] + start + len(c),
                    "source": "question_clause", "is_factual": True,
                })
                uid += 1
                cursor = start + len(c)
    units.extend(factual_units)
    if not any(u["is_factual"] for u in units):
        raise SegmentationError("no factual input unit found")
    return units


def _prompt_span(u: dict, prompt_text: str) -> tuple[int, int, str]:
    """Translate a unit's char span (relative to its source string, e.g. the raw
    question or system prompt) into coordinates of the rendered prompt_text.

    Chat templates prepend/append template symbols, so source-relative spans do NOT
    index prompt_text directly. Prefer locating the unit text verbatim; fall back to
    the raw span only as a last resort (recorded as unverified).
    """
    text = u.get("text") or ""
    if text and text in prompt_text:
        i = prompt_text.find(text)
        return i, i + len(text), "text_search"
    return int(u["char_start"]), int(u["char_end"]), "source_relative_unverified"


def unit_token_spans(units: list[dict], prompt_text: str, prompt_token_ids: list[int],
                     tokenizer) -> list[dict]:
    """Resolve token spans of units against the rendered prompt via character offsets.

    Uses tokenizer offset mapping when available; falls back to a leading-text
    incremental search. Units that cannot be located keep token_start=-1 and are
    excluded downstream (recorded, never silently zeroed).
    """
    out = []
    fast = hasattr(tokenizer, "__call__")
    encoded = None
    if fast:
        try:
            encoded = tokenizer(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
        except Exception:
            try:
                enc = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
                encoded = enc
            except Exception:
                encoded = None
    offsets = encoded.get("offset_mapping") if isinstance(encoded, dict) else getattr(encoded, "offset_mapping", None) if encoded is not None else None
    if offsets is not None and len(offsets) == len(prompt_token_ids):
        for u in units:
            cs, ce, span_src = _prompt_span(u, prompt_text)
            toks = [i for i, (a, b) in enumerate(offsets) if a < ce and b > cs]
            u2 = dict(u)
            u2["char_start_in_prompt"], u2["char_end_in_prompt"] = cs, ce
            u2["span_source"] = span_src
            if toks:
                u2["token_start"], u2["token_end"] = min(toks), max(toks) + 1
            else:
                u2["token_start"], u2["token_end"] = -1, -1
            out.append(u2)
        return out
    cur = 0
    token_texts = None
    try:
        token_texts = tokenizer.convert_ids_to_tokens(prompt_token_ids)
    except Exception:
        token_texts = None
    for u in units:
        u2 = dict(u)
        cs, ce, span_src = _prompt_span(u, prompt_text)
        u2["char_start_in_prompt"], u2["char_end_in_prompt"] = cs, ce
        u2["span_source"] = span_src
        if token_texts is None:
            u2["token_start"], u2["token_end"] = -1, -1
        else:
            pos, tstart, tend = 0, -1, -1
            for i, tt in enumerate(token_texts):
                clean = tt.replace("Ġ", " ").replace("▁", " ")
                nxt = pos + len(clean)
                if nxt > cs and pos < ce:
                    if tstart == -1:
                        tstart = i
                    tend = i + 1
                pos = nxt
                if pos > ce and tstart != -1:
                    break
            u2["token_start"], u2["token_end"] = tstart, tend
        out.append(u2)
    return out
