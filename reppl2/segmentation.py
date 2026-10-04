from __future__ import annotations

import hashlib
import json
from typing import Optional


class SegmentationError(RuntimeError):
    pass


_SENT_SPLIT = None


def _split_sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [p for p in parts if p.strip()]


def segment_input(example, tokenizer=None, max_units: int = 8) -> list[dict]:
    """Produce input units with char spans. Sources: system (non-factual), question sentences, RAG paragraphs."""
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
    for s in _split_sentences(example.question)[:max_units]:
        start = example.question.find(s)
        units.append({
            "unit_id": uid, "text": s, "char_start": start, "char_end": start + len(s),
            "source": "question_sentence", "is_factual": True,
        })
        uid += 1
    if not any(u["is_factual"] for u in units):
        raise SegmentationError("no factual input unit found")
    return units


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
            cs, ce = u["char_start"], u["char_end"]
            toks = [i for i, (a, b) in enumerate(offsets) if a < ce and b > cs]
            u2 = dict(u)
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
        cs, ce = u["char_start"], u["char_end"]
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
