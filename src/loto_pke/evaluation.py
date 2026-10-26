"""Experiment-level evaluations with macro/micro denominators and coverage.

Results computed here are independently implemented comparisons against the
public TXT annotations. They are not the published paper's original scores.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Sequence

from loto_pke.metrics import (
    TextSimilarity,
    bbox_iou,
    expert_agreement,
    graph_overlap_metrics,
    instruction_metrics,
    match_items,
    normalized_graph_edit_distance,
    precision_recall_f1,
    rouge1_f1,
    rouge_l_f1,
    triplet_metrics,
)
from loto_pke.pkg import build_graph


NODE_TYPES = ("Step", "Action", "Target", "Tool", "Location", "Verification")
SLOT_TYPES = ("action", "target", "tool", "location", "verification")
DEFAULT_THRESHOLDS = {"exact": 1.0, "fuzzy": 0.85, "embedding": 0.90}


def index_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for record in records:
        identifier = str(record["document_id"])
        if identifier in result:
            raise ValueError(f"Duplicate document_id in input: {identifier}")
        result[identifier] = record
    return result


def unpack_graph(record: dict[str, Any]) -> dict[str, Any] | None:
    graph = record.get("graph", record)
    return graph if all(key in graph for key in ("nodes", "edges", "triplets")) else None


def _graph_groups(graph: dict[str, Any]) -> list[dict[str, Any]] | None:
    """For annotated and structured baselines only; direct VLM has no groups."""

    if graph.get("output_mode") == "direct_graph_from_images":
        return None
    steps = graph.get("steps")
    if not isinstance(steps, list):
        return None
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        by_group[str(step.get("procedure_id") or "unknown")].append(step)
    return [build_graph(str(graph["document_id"]), group) for group in by_group.values()]


def _comparison_units(
    predicted: dict[str, Any], gold: dict[str, Any], granularity: str,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], str]], str | None]:
    if granularity == "document":
        return [(predicted, gold, "document")], None
    predicted_groups, gold_groups = _graph_groups(predicted), _graph_groups(gold)
    if predicted_groups is None or gold_groups is None:
        return [], "Procedure segmentation unavailable in a direct/unstructured graph"
    empty = build_graph(str(gold["document_id"]), [])
    pairs = []
    for index in range(max(len(predicted_groups), len(gold_groups))):
        pairs.append((
            predicted_groups[index] if index < len(predicted_groups) else empty,
            gold_groups[index] if index < len(gold_groups) else empty,
            f"ordinal_procedure_{index + 1}",
        ))
    return pairs, None


def _mean_of(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in rows if item.get(key) is not None]
    return mean(values) if values else None


def _node_metrics(
    predicted: dict[str, Any], gold: dict[str, Any],
    similarity: TextSimilarity, threshold: float,
) -> dict[str, dict[str, Any]]:
    result = {}
    for node_type in NODE_TYPES:
        p = [node["label"] for node in predicted.get("nodes", []) if node["type"] == node_type]
        g = [node["label"] for node in gold.get("nodes", []) if node["type"] == node_type]
        result[node_type] = match_items(p, g, similarity.score, threshold)
    return result


def evaluate_graphs(
    predicted_records: list[dict[str, Any]],
    gold_records: list[dict[str, Any]],
    methods: Sequence[str] = ("exact", "fuzzy"),
    thresholds: dict[str, float] | None = None,
    granularity: str = "document",
    ged: bool = False,
    ged_timeout: float = 5.0,
    embedding_model: str = "google/embeddinggemma-300m",
    offline: bool = True,
) -> dict[str, Any]:
    if granularity not in ("document", "procedure_index"):
        raise ValueError("Granularity must be document or procedure_index")
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    predicted_index, gold_index = index_records(predicted_records), index_records(gold_records)
    similarities = {
        method: TextSimilarity(method, model_name=embedding_model, offline=offline)
        for method in methods
    }
    rows: list[dict[str, Any]] = []
    unevaluable: list[dict[str, str]] = []
    missing_prediction = []

    for document_id, reference in gold_index.items():
        prediction = predicted_index.get(document_id)
        predicted_graph = unpack_graph(prediction) if prediction else None
        gold_graph = unpack_graph(reference)
        if gold_graph is None:
            raise ValueError(f"Gold graph is invalid for document {document_id}")
        if predicted_graph is None:
            missing_prediction.append(document_id)
            continue

        pairs, reason = _comparison_units(predicted_graph, gold_graph, granularity)
        if reason:
            unevaluable.append({"document_id": document_id, "reason": reason})
            continue
        for predicted_unit, gold_unit, unit_id in pairs:
            for method, similarity in similarities.items():
                threshold = thresholds[method]
                result = triplet_metrics(predicted_unit, gold_unit, similarity, threshold)
                overlap = graph_overlap_metrics(predicted_unit, gold_unit)
                node_scores = _node_metrics(predicted_unit, gold_unit, similarity, threshold)
                row: dict[str, Any] = {
                    "document_id": document_id,
                    "unit_id": unit_id,
                    "method": method,
                    "threshold": threshold,
                    "triplets": result,
                    "node_types": node_scores,
                    "graph_overlap_exact": overlap,
                }
                if ged:
                    row["graph_edit_distance"] = normalized_graph_edit_distance(
                        predicted_unit, gold_unit, timeout=ged_timeout,
                        similarity=similarity, threshold=threshold,
                    )
                rows.append(row)

    aggregates = {}
    for method in methods:
        matched = [row for row in rows if row["method"] == method]
        tmetrics = [row["triplets"] for row in matched]
        micro = precision_recall_f1(
            sum(item["matched"] for item in tmetrics),
            sum(item["predicted"] for item in tmetrics),
            sum(item["gold"] for item in tmetrics),
        )
        aggregates[method] = {
            "unit_count": len(matched),
            "triplet_macro": {
                name: _mean_of(tmetrics, name)
                for name in ("precision", "recall", "f1", "average_matched_similarity")
            },
            "triplet_micro": micro,
            "graph_overlap_exact_macro": {
                name: _mean_of([row["graph_overlap_exact"] for row in matched], name)
                for name in ("node_jaccard", "edge_jaccard")
            },
            "node_types_macro": {
                kind: {
                    metric: _mean_of([row["node_types"][kind] for row in matched], metric)
                    for metric in ("precision", "recall", "f1")
                }
                for kind in NODE_TYPES
            },
            "normalized_ged_macro": _mean_of(
                [row.get("graph_edit_distance", {}) for row in matched], "normalized_ged",
            ) if ged else None,
        }

    return {
        "evaluation": "public_reference_pkg",
        "matching_thresholds": {method: thresholds[method] for method in methods},
        "granularity": granularity,
        "procedure_pairing": "ordinal_not_annotator_verified" if granularity != "document" else None,
        "embedding_model": embedding_model if "embedding" in methods else None,
        "graph_overlap_note": "Exact typed node/edge signatures; not fuzzy-normalized Jaccard",
        "ged_note": "NetworkX GED with method-specific node-label substitution, exact typed edge relations",
        "gold_document_count": len(gold_index),
        "predicted_document_count": len(predicted_index),
        "missing_or_invalid_prediction_document_ids": missing_prediction,
        "not_evaluable": unevaluable,
        "aggregate": aggregates,
        "per_unit": rows,
    }


def extract_steps(record: dict[str, Any]) -> list[dict[str, Any]]:
    if "steps" in record:
        if not isinstance(record["steps"], list):
            raise ValueError("steps must be an array")
        return record["steps"]
    if "procedures" in record:
        return [step for proc in record["procedures"] for step in proc.get("steps", [])]
    if "graph" in record:
        return extract_steps(record["graph"])
    raise ValueError("No predicted steps in this record")


def evaluate_instructions(
    predicted_records: list[dict[str, Any]], gold_records: list[dict[str, Any]],
    similarity: TextSimilarity,
) -> dict[str, Any]:
    predicted_index, gold_index = index_records(predicted_records), index_records(gold_records)
    rows = []
    missing = []
    for document_id, gold in gold_index.items():
        prediction = predicted_index.get(document_id)
        if prediction is None or prediction.get("status") == "error":
            missing.append(document_id)
            continue
        try:
            p = [str(step.get("action") or "") for step in extract_steps(prediction)]
            g = [str(step.get("action") or "") for step in extract_steps(gold)]
        except ValueError:
            missing.append(document_id)
            continue
        rows.append({"document_id": document_id, **instruction_metrics(p, g, similarity)})

    names = ("diag_mean", "alignment_accuracy", "off_diag_mean", "margin_difference",
             "rouge1_f1", "rougeL_f1", "bleu4", "meteor")
    return {
        "evaluation": "instruction_extraction",
        "similarity_method": similarity.method,
        "gold_document_count": len(gold_index),
        "missing_or_invalid_prediction_document_ids": missing,
        "matching_rule": "Document reading order; evaluate first min(predicted,gold) steps, as in paper's index-aligned metrics",
        "macro": {name: _mean_of(rows, name) for name in names},
        "smart_status": "not_evaluable_without_official_BLEURT_SMART_and_transcribed_page_reference",
        "per_document": rows,
    }


def evaluate_slots(
    predicted_records: list[dict[str, Any]], gold_records: list[dict[str, Any]],
    similarity: TextSimilarity, threshold: float,
) -> dict[str, Any]:
    """Require matched actions for a role match, penalizing unmatched steps."""

    p_index, g_index = index_records(predicted_records), index_records(gold_records)
    rows, missing = [], []
    for document_id, gold in g_index.items():
        prediction = p_index.get(document_id)
        if prediction is None or prediction.get("status") == "error":
            missing.append(document_id)
            continue
        try:
            psteps, gsteps = extract_steps(prediction), extract_steps(gold)
        except ValueError:
            missing.append(document_id)
            continue
        paired = match_items(
            psteps, gsteps,
            lambda p, g: similarity.score(str(p.get("predicted_action") or p.get("action") or ""), str(g.get("action") or "")),
            threshold,
        )
        slot_scores = {}
        for field in SLOT_TYPES:
            pred_count = sum(bool(step.get("predicted_action") or step.get("action")) if field == "action"
                             else bool(step.get(field)) for step in psteps)
            gold_count = sum(bool(step.get(field)) for step in gsteps)
            matched = 0
            for pair in paired["pairs"]:
                p, g = psteps[pair["prediction_index"]], gsteps[pair["gold_index"]]
                left = p.get("predicted_action") or p.get("action") if field == "action" else p.get(field)
                right = g.get(field)
                if left and right and similarity.score(str(left), str(right)) >= threshold:
                    matched += 1
            slot_scores[field] = {
                **precision_recall_f1(matched, pred_count, gold_count),
                "matched": matched, "predicted": pred_count, "gold": gold_count,
            }
        rows.append({"document_id": document_id, "action_alignment": paired,
                     "slots": slot_scores})

    aggregate = {
        field: {
            "macro_f1": _mean_of([row["slots"][field] for row in rows], "f1"),
            "micro": precision_recall_f1(
                sum(row["slots"][field]["matched"] for row in rows),
                sum(row["slots"][field]["predicted"] for row in rows),
                sum(row["slots"][field]["gold"] for row in rows),
            ),
        }
        for field in SLOT_TYPES
    }
    return {
        "evaluation": "slot_extraction_action_aligned",
        "similarity_method": similarity.method,
        "threshold": threshold,
        "gold_document_count": len(g_index),
        "missing_or_invalid_prediction_document_ids": missing,
        "aggregate": aggregate,
        "per_document": rows,
    }


def evaluate_parsing(
    predicted_records: list[dict[str, Any]],
    reference_records: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if reference_records is None:
        return {
            "evaluation": "document_parsing",
            "status": "not_evaluable",
            "reason": "Public TXT slots are not transcribed source pages or human layout boxes",
            "smart_1_f1": None, "smart_l_f1": None, "layout_iou": None,
        }
    indexed = {(str(item["document_id"]), int(item["page_number"])): item
               for item in reference_records}
    rows = []
    for prediction in predicted_records:
        for page in prediction.get("pages", []):
            key = (str(prediction["document_id"]), int(page["page_number"]))
            ref = indexed.get(key)
            row: dict[str, Any] = {"document_id": key[0], "page_number": key[1]}
            if ref is None or ref.get("annotation_status") != "verified":
                row.update({"status": "not_evaluable", "reason": "missing_verified_reference"})
            elif ref.get("coordinate_system") != page.get("coordinate_system"):
                row.update({"status": "not_evaluable", "reason": "coordinate_system_mismatch"})
            else:
                gold_blocks = ref.get("gold_blocks")
                if not isinstance(gold_blocks, list):
                    row.update({"status": "not_evaluable", "reason": "missing_gold_boxes"})
                else:
                    predicted_blocks = [b for b in page.get("blocks", []) if b.get("bbox")]
                    pairs = match_items(
                        predicted_blocks, gold_blocks,
                        lambda p, g: bbox_iou(p["bbox"], g["bbox"]), threshold=0.5,
                    )
                    row.update({"status": "evaluated", "bbox_iou_at_0_5": pairs})
                    ref_text = ref.get("gold_page_text")
                    if isinstance(ref_text, str):
                        row["rouge1_f1_proxy"] = rouge1_f1(str(page.get("text") or ""), ref_text)
                        row["rougeL_f1_proxy"] = rouge_l_f1(str(page.get("text") or ""), ref_text)
            rows.append(row)

    evaluated = [row for row in rows if row["status"] == "evaluated"]
    return {
        "evaluation": "document_parsing",
        "status": "evaluated" if evaluated else "not_evaluable",
        "evaluated_pages": len(evaluated),
        "reference_matching": "Greedy one-to-one max-cardinality IoU>=0.5; same coordinate system required",
        "bbox_macro_f1_at_0_5": _mean_of([row["bbox_iou_at_0_5"] for row in evaluated], "f1"),
        "rouge1_f1_proxy": _mean_of(rows, "rouge1_f1_proxy"),
        "rougeL_f1_proxy": _mean_of(rows, "rougeL_f1_proxy"),
        "smart_1_f1": None,
        "smart_l_f1": None,
        "smart_status": "Official BLEURT-backed SMART not implemented; ROUGE is a distinct proxy",
        "per_page": rows,
    }


def layout_review_template(parsed_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = []
    for record in parsed_records:
        for page in record.get("pages", []):
            tasks.append({
                "document_id": str(record["document_id"]),
                "page_number": int(page["page_number"]),
                "coordinate_system": page.get("coordinate_system"),
                "gold_page_text": None,
                "gold_blocks": [],
                "annotation_status": "needs_human_annotation",
                "extracted_text_preview": str(page.get("text") or "")[:300],
            })
    return tasks


def expert_review_template(safety_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "document_id": str(record["document_id"]),
            "procedure_id": str(block["procedure_id"]),
            "ordered_steps": block.get("ordered_steps", []),
            "energy_type_hints": block.get("detected_energy_types", []),
            "energy_source_correct": None,
            "action_order_correct": None,
            "missing_required_steps": None,
            "structural_error_severity": None,
            "approve_block": None,
            "comments": "",
            "annotation_status": "needs_certified_expert_review",
        }
        for record in safety_records
        for block in record.get("procedure_blocks", [])
    ]


def _binary_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip().casefold()
    if text in {"yes", "true", "correct"}:
        return True
    if text in {"no", "false", "wrong", "partially", "partial"}:
        return False
    return None


def evaluate_expert_agreement(
    safety_records: list[dict[str, Any]], expert_labels: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    predictions = []
    for record in safety_records:
        for block in record.get("procedure_blocks", []):
            predictions.append({
                **block,
                "document_id": str(record["document_id"]),
            })
    if expert_labels is None:
        return expert_agreement(predictions, None)
    # The expert's "Partially" for energy source is treated as No, per the
    # paper's Appendix B.4; unverified/pending labels are explicitly ignored.
    verified = [
        {
            **label,
            **{
                field: _binary_value(label.get(field))
                for field in ("energy_source_correct", "action_order_correct", "missing_required_steps")
            },
        }
        for label in expert_labels
        if label.get("annotation_status") == "verified"
    ]
    results = expert_agreement(predictions, verified)
    results["n_verified_expert_procedure_blocks"] = len(verified)
    results["partial_policy"] = "Expert Partially -> No, per paper appendix"
    results["safety_certification"] = "NOT_PERFORMED_BY_THIS_CODE"
    return results
