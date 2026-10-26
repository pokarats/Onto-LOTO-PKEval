"""Optional integration with Google's SMART evaluation implementation.

SMART uses a chosen sentence-pair matching function. The original draft used
BLEURT-20; a CHRF-backed SMART run is useful as a sanity check but must not be
reported under the draft's BLEURT-backed score label.
"""

from __future__ import annotations

from typing import Any

from loto_pke.common import require_module


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if hasattr(value, "_asdict"):
        return _json_compatible(value._asdict())
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value


def smart_scores(
    reference: str, candidate: str, matcher: str = "chrf",
    bleurt_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Run official smart_eval.SmartScorer with a requested matcher.

    The BLEURT callback follows Google's BLEURT `BleurtScorer.score` API, but
    must be verified against the installed SMART release before publication.
    """

    scoring = require_module("smart_eval.scorer", "Install the official Google smart-eval package")
    if matcher == "chrf":
        matching_functions = require_module("smart_eval.matching_functions", "Install the official Google smart-eval package")
        matching_fn = matching_functions.chrf_matcher
    elif matcher == "bleurt20":
        if not bleurt_checkpoint:
            raise ValueError("BLEURT-20 matcher requires --bleurt-checkpoint")
        bleurt = require_module("bleurt.score", "Install Google's BLEURT and checkpoint BLEURT-20")
        model = bleurt.BleurtScorer(checkpoint=bleurt_checkpoint)

        def matching_fn(text_a: str, text_b: str) -> float:
            return float(model.score(references=[text_a], candidates=[text_b])[0])

    else:
        raise ValueError(f"Unsupported SMART matcher {matcher!r}")

    instance = scoring.SmartScorer(matching_fn=matching_fn)
    return {
        "matcher": matcher,
        "raw_smart_scores": _json_compatible(instance.smart_score(reference, candidate)),
        "paper_bleurt20_parity_verified": False,
    }


def evaluate_smart_parsing(
    parsed_records: list[dict[str, Any]], reference_records: list[dict[str, Any]],
    matcher: str = "chrf", bleurt_checkpoint: str | None = None,
) -> dict[str, Any]:
    gold_index = {(str(item["document_id"]), int(item["page_number"])): item
                  for item in reference_records if item.get("annotation_status") == "verified"}
    rows = []
    missing = []
    for record in parsed_records:
        for page in record.get("pages", []):
            key = (str(record["document_id"]), int(page["page_number"]))
            gold = gold_index.get(key)
            if gold is None or not isinstance(gold.get("gold_page_text"), str):
                missing.append({"document_id": key[0], "page_number": key[1]})
                continue
            score = smart_scores(
                gold["gold_page_text"], str(page.get("text") or ""),
                matcher=matcher, bleurt_checkpoint=bleurt_checkpoint,
            )
            rows.append({"document_id": key[0], "page_number": key[1], **score})
    return {
        "evaluation": "smart_text_quality",
        "matcher": matcher,
        "status": "evaluated" if rows else "not_evaluable_missing_verified_page_transcriptions",
        "n_evaluated_pages": len(rows),
        "n_missing_reference_pages": len(missing),
        "missing_reference_pages": missing,
        "per_page": rows,
        "aggregation_note": "Use raw scorer keys from the installed SMART package; no guessed key remapping",
        "paper_bleurt20_parity_verified": False,
    }
