"""Human instruction/slot review tasks mirroring the paper's Appendix B.2."""

from __future__ import annotations

from collections import Counter
from typing import Any

from loto_pke.evaluation import extract_steps


LABELS = ("CORRECT", "PARTIAL", "WRONG", "N-SURE")
FIELDS = ("action", "target", "tool", "location")


def human_review_template(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for record in records:
        for step in extract_steps(record):
            output.append({
                "document_id": str(record["document_id"]),
                "procedure_id": str(step.get("procedure_id") or "unknown"),
                "step_id": str(step["step_id"]),
                "instruction": step.get("action"),
                "predicted_slots": {field: step.get("predicted_action") or step.get("action")
                                    if field == "action" else step.get(field) for field in FIELDS},
                "instruction_rating": None,
                "slot_ratings": {field: None for field in FIELDS},
                "error_notes": "",
                "annotator_id": None,
                "annotation_status": "needs_human_review",
            })
    return output


def summarize_human_review(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {field: Counter() for field in ("instruction", *FIELDS)}
    evaluated = 0
    for record in records:
        if record.get("annotation_status") != "verified":
            continue
        evaluated += 1
        observed = {"instruction": record.get("instruction_rating"), **record.get("slot_ratings", {})}
        for field, value in observed.items():
            if field not in counts:
                continue
            if value is not None and value not in LABELS:
                raise ValueError(f"Invalid human-review label {value!r}; expected one of {LABELS}")
            if value:
                counts[field][value] += 1
    result = {}
    for field, counter in counts.items():
        n = sum(counter.values())
        result[field] = {
            "n_rated": n,
            "distribution": {label: counter[label] for label in LABELS},
            "correct_rate": counter["CORRECT"] / n if n else None,
            "correct_or_partial_rate": (counter["CORRECT"] + counter["PARTIAL"]) / n if n else None,
        }
    return {
        "evaluation": "human_review",
        "verified_step_count": evaluated,
        "label_schema": list(LABELS),
        "criteria": result,
        "status": "evaluated" if evaluated else "not_evaluable_no_verified_reviews",
    }
