from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class EditError(RuntimeError):
    pass


@dataclass
class EditRecord:
    unit_id: int
    edit_type: str
    original_text: str
    replacement_text: str
    char_start: int
    char_end: int
    structure_ok: bool
    note: str = ""


def neutralize_unit(context: str, question: str, unit: dict, filler: str = "[MASKED]") -> tuple[str, str, EditRecord]:
    """Deterministic, auditable edit: replace the unit's factual text with an explicit unknown marker."""
    text = unit["text"]
    if not text.strip():
        raise EditError("unit text is empty; cannot edit")
    if unit["source"] == "system":
        raise EditError("system units are not factual inputs; not edited by default")
    target_field = "context" if unit["source"] == "rag_paragraph" else "question"
    base = context if target_field == "context" else question
    if base[unit["char_start"]:unit["char_end"]] != text:
        start = base.find(text)
        if start == -1:
            raise EditError("unit text not found in its source field; span mismatch")
    else:
        start = unit["char_start"]
    edited = base[:start] + filler + base[start + len(text):]
    rec = EditRecord(
        unit_id=unit["unit_id"], edit_type="neutralize",
        original_text=text, replacement_text=filler,
        char_start=start, char_end=start + len(filler),
        structure_ok=True,
    )
    new_context = edited if target_field == "context" else context
    new_question = edited if target_field == "question" else question
    return new_context, new_question, rec
