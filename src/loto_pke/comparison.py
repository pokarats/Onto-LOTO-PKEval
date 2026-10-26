"""Paired multi-step versus end-to-end VLM comparisons on common documents."""

from __future__ import annotations

import random
from statistics import mean
from typing import Any

from loto_pke.evaluation import evaluate_graphs


def _row_lookup(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for row in report["per_unit"]:
        key = (row["document_id"], row["method"])
        if row["unit_id"] != "document":
            raise ValueError("Paired pipeline comparison requires document-level units")
        result[key] = row
    return result


def _paired_bootstrap_ci(
    differences: list[float], samples: int, seed: int = 42,
) -> list[float] | None:
    if not differences:
        return None
    if samples <= 0:
        return None
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(samples))
    return [estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]]


def compare_pipelines(
    multi: list[dict[str, Any]], baseline: list[dict[str, Any]],
    gold: list[dict[str, Any]], method: str = "fuzzy", threshold: float = 0.85,
    bootstrap_samples: int = 2000, seed: int = 42,
    embedding_model: str = "google/embeddinggemma-300m", offline: bool = True,
) -> dict[str, Any]:
    config = dict(
        methods=(method,), thresholds={method: threshold},
        embedding_model=embedding_model, offline=offline,
        granularity="document", ged=False,
    )
    multi_report = evaluate_graphs(multi, gold, **config)
    baseline_report = evaluate_graphs(baseline, gold, **config)
    by_multi = _row_lookup(multi_report)
    by_baseline = _row_lookup(baseline_report)
    shared = sorted(set(by_multi) & set(by_baseline))
    paired = []
    for key in shared:
        mrow, brow = by_multi[key], by_baseline[key]
        mf1, bf1 = float(mrow["triplets"]["f1"]), float(brow["triplets"]["f1"])
        paired.append({
            "document_id": key[0],
            "method": method,
            "multistage_triplet_f1": mf1,
            "end_to_end_triplet_f1": bf1,
            "delta_multistage_minus_end_to_end": mf1 - bf1,
            "multistage_action_node_f1": mrow["node_types"]["Action"]["f1"],
            "end_to_end_action_node_f1": brow["node_types"]["Action"]["f1"],
            "delta_action_node_f1": (
                mrow["node_types"]["Action"]["f1"] - brow["node_types"]["Action"]["f1"]
            ),
        })
    differences = [row["delta_multistage_minus_end_to_end"] for row in paired]
    action_differences = [row["delta_action_node_f1"] for row in paired]
    return {
        "evaluation": "paired_multistage_vs_end_to_end",
        "method": method,
        "threshold": threshold,
        "n_common_evaluable_documents": len(paired),
        "n_gold_documents": len(gold),
        "multistage_missing_or_invalid": multi_report["missing_or_invalid_prediction_document_ids"],
        "baseline_missing_or_invalid": baseline_report["missing_or_invalid_prediction_document_ids"],
        "triplet_f1_delta_mean": mean(differences) if differences else None,
        "triplet_f1_delta_bootstrap_95ci": _paired_bootstrap_ci(differences, bootstrap_samples, seed),
        "action_node_f1_delta_mean": mean(action_differences) if action_differences else None,
        "action_node_f1_delta_bootstrap_95ci": _paired_bootstrap_ci(action_differences, bootstrap_samples, seed),
        "conditional_f1": None,
        "conditional_status": "Public TXT annotations have no executable/conditional step_type gold labels",
        "cross_modal_f1": None,
        "cross_modal_status": "No verified gold action/entity-to-visual region links in public TXT",
        "paired_documents": paired,
        "comparison_note": "Identical public document gold and matching thresholds; not the paper's unknown subset",
    }
