"""Inter-annotator agreement for independently collected public annotations.

No second annotator labels are shipped in public_data. A conceptual-normalizer
must be explicit, versioned and researcher-approved; string casefolding alone
is not a valid surrogate for ontology-level conceptual agreement.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from loto_pke.evaluation import extract_steps


SLOTS = ("action", "target", "tool", "location")


def _step_index(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for record in records:
        if "step_id" in record:
            steps = [record]
        else:
            steps = extract_steps(record)
        for step in steps:
            key = (str(step["document_id"]), str(step["step_id"]))
            if key in result:
                raise ValueError(f"Duplicate annotated step ID: {key}")
            result[key] = step
    return result


def _normalize(value: Any, field: str, normalizer: dict[str, Any] | None) -> str:
    text = str(value).strip() if value is not None else "<NULL>"
    if normalizer is None:
        return text
    canonical = text.casefold().strip()
    mappings = normalizer.get(field, {})
    if not isinstance(mappings, dict):
        raise ValueError(f"Concept normalization for {field} must be a mapping")
    return str(mappings.get(canonical, canonical))


def _kappa(first: list[str], second: list[str]) -> float | None:
    if len(first) != len(second):
        raise ValueError("Cohen kappa requires paired observations")
    if not first:
        return None
    observed = sum(a == b for a, b in zip(first, second)) / len(first)
    fa, fb = Counter(first), Counter(second)
    expected = sum(fa[label] * fb[label] for label in set(fa) | set(fb)) / len(first) ** 2
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1 - expected)


def _jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def _score_pairs(
    left: dict[tuple[str, str], dict[str, Any]],
    right: dict[tuple[str, str], dict[str, Any]],
    normalizer: dict[str, Any] | None,
) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    pair_rows = []
    field_scores = {}
    for field in SLOTS:
        first = [_normalize(left[key].get(field), field, normalizer) for key in common]
        second = [_normalize(right[key].get(field), field, normalizer) for key in common]
        field_scores[field] = {
            "n_shared_steps": len(common),
            "exact_or_normalized_accuracy": sum(a == b for a, b in zip(first, second)) / len(common) if common else None,
            "cohen_kappa": _kappa(first, second),
        }

    by_document: dict[str, tuple[set[tuple[str, ...]], set[tuple[str, ...]]]] = defaultdict(lambda: (set(), set()))
    for key, step in left.items():
        by_document[key[0]][0].add(tuple(_normalize(step.get(field), field, normalizer) for field in SLOTS))
    for key, step in right.items():
        by_document[key[0]][1].add(tuple(_normalize(step.get(field), field, normalizer) for field in SLOTS))

    for document_id, (first, second) in sorted(by_document.items()):
        pair_rows.append({"document_id": document_id, "action_tuple_set_jaccard": _jaccard(first, second),
                          "annotator_a_tuple_count": len(first), "annotator_b_tuple_count": len(second)})

    first_tuples = ["|".join(_normalize(left[key].get(field), field, normalizer) for field in SLOTS) for key in common]
    second_tuples = ["|".join(_normalize(right[key].get(field), field, normalizer) for field in SLOTS) for key in common]
    return {
        "shared_step_count": len(common),
        "annotator_a_step_count": len(left),
        "annotator_b_step_count": len(right),
        "unmatched_a_step_ids": [key[1] for key in sorted(set(left) - set(right))],
        "unmatched_b_step_ids": [key[1] for key in sorted(set(right) - set(left))],
        "field_agreement": field_scores,
        "tuple_exact_or_normalized_cohen_kappa": _kappa(first_tuples, second_tuples),
        "tuple_set_jaccard_macro_document": mean(item["action_tuple_set_jaccard"] for item in pair_rows) if pair_rows else None,
        "per_document": pair_rows,
    }


def inter_annotator_agreement(
    records_a: list[dict[str, Any]], records_b: list[dict[str, Any]],
    conceptual_normalization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if conceptual_normalization is not None:
        if not conceptual_normalization.get("approved_by") or not conceptual_normalization.get("version"):
            raise ValueError("Conceptual mappings require nonempty approved_by and version fields")
        if not any(conceptual_normalization.get(field) for field in SLOTS):
            raise ValueError("Conceptual mapping has no approved equivalences")
    left, right = _step_index(records_a), _step_index(records_b)
    result = {
        "evaluation": "inter_annotator_agreement",
        "exact_lexical": _score_pairs(left, right, None),
        "conceptual_normalized": _score_pairs(left, right, conceptual_normalization) if conceptual_normalization is not None else None,
        "conceptual_normalization_status": (
            "user_supplied_approved_mappings" if conceptual_normalization is not None else "not_evaluable_missing_approved_mappings"
        ),
        "alignment_rule": "shared stable step_id for Cohen kappa; per-document tuple sets for Jaccard",
        "original_paper_iaa_reproducible": False,
        "original_paper_iaa_reason": "Independent annotator exports and approved conceptual equivalence mapping not in public_data",
    }
    return result
