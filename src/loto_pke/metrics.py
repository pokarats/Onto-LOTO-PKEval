"""Reference matching and graph metrics with explicit, reproducible choices.

The metrics in this module are independently implemented variants. The original
paper's missing matching code and human judgments prevent exact score recovery.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Callable, Sequence

from loto_pke.common import require_module


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.casefold(), flags=re.UNICODE)


def exact_similarity(first: str, second: str) -> float:
    return 1.0 if first.strip() == second.strip() else 0.0


def fuzzy_similarity(first: str, second: str) -> float:
    return SequenceMatcher(None, first.casefold().strip(), second.casefold().strip()).ratio()


class TextSimilarity:
    """Exact, fuzzy, or locally loaded SentenceTransformer representations."""

    def __init__(
        self, method: str = "exact", model_name: str = "google/embeddinggemma-300m",
        offline: bool = True,
    ) -> None:
        if method not in {"exact", "fuzzy", "embedding"}:
            raise ValueError(f"Unsupported similarity: {method}")
        self.method = method
        self.model_name = model_name if method == "embedding" else None
        self._embeddings: dict[str, Any] = {}
        self._model = None
        if method == "embedding":
            module = require_module("sentence_transformers", "Install sentence-transformers")
            self._model = module.SentenceTransformer(model_name, local_files_only=offline)

    def _vector(self, text: str) -> Any:
        if text not in self._embeddings:
            if self._model is None:
                raise RuntimeError("Embedding model is not loaded")
            self._embeddings[text] = self._model.encode(text, normalize_embeddings=True)
        return self._embeddings[text]

    def score(self, first: str, second: str) -> float:
        if self.method == "exact":
            return exact_similarity(first, second)
        if self.method == "fuzzy":
            return fuzzy_similarity(first, second)
        first_vector, second_vector = self._vector(first), self._vector(second)
        # NumPy arrays, torch tensors, and Python sequences all allow iteration.
        return max(-1.0, min(1.0, float(sum(float(a) * float(b) for a, b in zip(first_vector, second_vector)))))


def maximum_cardinality_pairs(
    scores: list[list[float]], threshold: float,
) -> list[tuple[int, int, float]]:
    """Deterministic one-to-one maximum-cardinality matching.

    Augmenting paths maximize *number* of accepted pairs. Within that objective,
    candidates are visited by descending similarity. This is not guaranteed
    to maximize the sum of accepted similarities.
    """

    if not scores or not scores[0]:
        return []
    n_gold = len(scores[0])
    if any(len(row) != n_gold for row in scores):
        raise ValueError("Similarity matrix must be rectangular")

    owner: dict[int, int] = {}

    def try_assign(predicted_index: int, visited: set[int]) -> bool:
        candidates = sorted(
            (j for j in range(n_gold) if scores[predicted_index][j] >= threshold),
            key=lambda j: (-scores[predicted_index][j], j),
        )
        for gold_index in candidates:
            if gold_index in visited:
                continue
            visited.add(gold_index)
            if gold_index not in owner or try_assign(owner[gold_index], visited):
                owner[gold_index] = predicted_index
                return True
        return False

    for i in range(len(scores)):
        try_assign(i, set())
    return sorted((i, j, scores[i][j]) for j, i in owner.items())


def precision_recall_f1(matched: int, predicted: int, gold: int) -> dict[str, float]:
    if predicted == 0 and gold == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    precision = matched / predicted if predicted else 0.0
    recall = matched / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def match_items(
    predicted: list[Any], gold: list[Any], scorer: Callable[[Any, Any], float], threshold: float,
) -> dict[str, Any]:
    matrix = [[scorer(p, g) for g in gold] for p in predicted]
    matches = maximum_cardinality_pairs(matrix, threshold)
    result = precision_recall_f1(len(matches), len(predicted), len(gold))
    return {
        **result,
        "matched": len(matches),
        "predicted": len(predicted),
        "gold": len(gold),
        "average_matched_similarity": sum(score for _, _, score in matches) / len(matches) if matches else None,
        "pairs": [{"prediction_index": i, "gold_index": j, "similarity": score} for i, j, score in matches],
    }


def _triple_score(predicted: dict[str, str], gold: dict[str, str], similarity: TextSimilarity) -> float:
    if predicted.get("relation") != gold.get("relation"):
        return 0.0
    return (similarity.score(str(predicted.get("subject", "")), str(gold.get("subject", ""))) +
            similarity.score(str(predicted.get("object", "")), str(gold.get("object", "")))) / 2


def triplet_metrics(
    predicted: dict[str, Any], gold: dict[str, Any], similarity: TextSimilarity, threshold: float,
) -> dict[str, Any]:
    return match_items(
        list(predicted.get("triplets", [])),
        list(gold.get("triplets", [])),
        lambda p, g: _triple_score(p, g, similarity),
        threshold,
    )


def _jaccard(first: set[Any], second: set[Any]) -> float:
    return len(first & second) / len(first | second) if first or second else 1.0


def graph_overlap_metrics(predicted: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    def graph_signatures(graph: dict[str, Any]) -> tuple[set[Any], set[Any]]:
        node_map = {node["id"]: (node["type"], node["label"]) for node in graph.get("nodes", [])}
        node_signatures = set(node_map.values())
        edge_signatures = {
            (node_map[edge["source"]], edge["relation"], node_map[edge["target"]])
            for edge in graph.get("edges", [])
            if edge["source"] in node_map and edge["target"] in node_map
        }
        return node_signatures, edge_signatures

    predicted_nodes, predicted_edges = graph_signatures(predicted)
    gold_nodes, gold_edges = graph_signatures(gold)
    return {
        "node_jaccard": _jaccard(predicted_nodes, gold_nodes),
        "edge_jaccard": _jaccard(predicted_edges, gold_edges),
    }


def normalized_graph_edit_distance(
    predicted: dict[str, Any], gold: dict[str, Any], timeout: float = 5.0,
    similarity: TextSimilarity | None = None, threshold: float = 1.0,
) -> dict[str, Any]:
    """Actual NetworkX GED with method-dependent node substitution costs."""

    networkx = require_module("networkx", "Install networkx for graph edit distance")

    def to_graph(record: dict[str, Any]) -> Any:
        graph = networkx.DiGraph()
        for node in record.get("nodes", []):
            graph.add_node(node["id"], type=node["type"], label=node["label"])
        for edge in record.get("edges", []):
            graph.add_edge(edge["source"], edge["target"], relation=edge["relation"])
        return graph

    first, second = to_graph(predicted), to_graph(gold)
    def node_cost(a: dict[str, str], b: dict[str, str]) -> float:
        if a["type"] != b["type"]:
            return 1.0
        if similarity is None:
            return 0.0 if a["label"] == b["label"] else 1.0
        return 0.0 if similarity.score(a["label"], b["label"]) >= threshold else 1.0

    value = networkx.graph_edit_distance(
        first, second,
        node_subst_cost=node_cost,
        edge_subst_cost=lambda a, b: 0 if a["relation"] == b["relation"] else 1,
        node_del_cost=lambda _: 1,
        node_ins_cost=lambda _: 1,
        edge_del_cost=lambda _: 1,
        edge_ins_cost=lambda _: 1,
        timeout=timeout,
    )
    denominator = max(first.number_of_nodes() + first.number_of_edges(), second.number_of_nodes() + second.number_of_edges(), 1)
    return {
        "ged": float(value) if value is not None else None,
        "normalized_ged": min(1.0, float(value) / denominator) if value is not None else None,
        "timeout_seconds": timeout,
        "node_matching_method": similarity.method if similarity else "exact",
        "node_matching_threshold": threshold,
        "normalization": "GED / max(V_pred+E_pred, V_gold+E_gold, 1)",
    }


def rouge1_f1(predicted: str, reference: str) -> float:
    pred_tokens, ref_tokens = tokenize(predicted), tokenize(reference)
    matched = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    return precision_recall_f1(matched, len(pred_tokens), len(ref_tokens))["f1"]


def rouge_l_f1(predicted: str, reference: str) -> float:
    first, second = tokenize(predicted), tokenize(reference)
    previous = [0] * (len(second) + 1)
    for word in first:
        row = [0]
        for j, gold_word in enumerate(second, start=1):
            row.append(previous[j - 1] + 1 if word == gold_word else max(previous[j], row[j - 1]))
        previous = row
    return precision_recall_f1(previous[-1], len(first), len(second))["f1"]


def sentence_bleu4(predicted: str, reference: str) -> float:
    """Smoothed sentence BLEU-4; published BLEU tokenization may differ."""

    pred, gold = tokenize(predicted), tokenize(reference)
    if not pred or not gold:
        return 1.0 if not pred and not gold else 0.0
    precisions = []
    for n in range(1, 5):
        pred_ngrams = Counter(tuple(pred[i:i+n]) for i in range(len(pred) - n + 1))
        gold_ngrams = Counter(tuple(gold[i:i+n]) for i in range(len(gold) - n + 1))
        hits = sum((pred_ngrams & gold_ngrams).values())
        total = sum(pred_ngrams.values())
        precisions.append((hits + 1) / (total + 1) if total else 1.0)
    brevity = min(1.0, math.exp(1.0 - len(gold) / len(pred)))
    return brevity * math.exp(sum(math.log(value) for value in precisions) / 4)


def meteor_score_optional(predicted: str, reference: str) -> float | None:
    try:
        module = require_module("nltk.translate.meteor_score", "Install nltk and its WordNet resources")
        return float(module.meteor_score([tokenize(reference)], tokenize(predicted)))
    except (RuntimeError, LookupError):
        return None


def step_alignment_metrics(
    predicted: Sequence[str], gold: Sequence[str], similarity: TextSimilarity,
) -> dict[str, Any]:
    """Paper-style index-diagonal alignment over the first L=min(n,m) items."""

    length = min(len(predicted), len(gold))
    if not length:
        return {
            "diag_mean": None, "alignment_accuracy": None,
            "off_diag_mean": None, "margin_difference": None,
            "matched_length": 0,
        }
    matrix = [
        [similarity.score(predicted[i], gold[j]) for j in range(length)]
        for i in range(length)
    ]
    diagonal = [matrix[i][i] for i in range(length)]
    off_diagonal = [matrix[i][j] for i in range(length) for j in range(length) if i != j]
    margins = [
        matrix[i][i] - max(matrix[i][j] for j in range(length) if i != j)
        for i in range(length)
    ] if length > 1 else []
    return {
        "diag_mean": sum(diagonal) / length,
        "alignment_accuracy": sum(max(range(length), key=lambda j: matrix[i][j]) == i for i in range(length)) / length,
        "off_diag_mean": sum(off_diagonal) / len(off_diagonal) if off_diagonal else None,
        "margin_difference": sum(margins) / len(margins) if margins else None,
        "matched_length": length,
        "prediction_length": len(predicted),
        "gold_length": len(gold),
    }


def instruction_metrics(
    predicted: Sequence[str], gold: Sequence[str], similarity: TextSimilarity,
) -> dict[str, Any]:
    alignment = step_alignment_metrics(predicted, gold, similarity)
    length = min(len(predicted), len(gold))
    if not length:
        return {**alignment, "rouge1_f1": None, "rougeL_f1": None, "bleu4": None, "meteor": None}
    pairs = list(zip(predicted[:length], gold[:length]))
    meteor = [meteor_score_optional(p, g) for p, g in pairs]
    return {
        **alignment,
        "rouge1_f1": sum(rouge1_f1(p, g) for p, g in pairs) / length,
        "rougeL_f1": sum(rouge_l_f1(p, g) for p, g in pairs) / length,
        "bleu4": sum(sentence_bleu4(p, g) for p, g in pairs) / length,
        "meteor": sum(value for value in meteor if value is not None) / length if all(value is not None for value in meteor) else None,
        "SMART_1_F1": None,
        "SMART_L_F1": None,
        "smart_status": "Official BLEURT-backed SMART scorer and original reference alignments unavailable",
    }


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != 4 or len(second) != 4:
        raise ValueError("Bounding boxes must be [x0,y0,x1,y1]")
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    if ax1 < ax0 or ay1 < ay0 or bx1 < bx0 or by1 < by0:
        raise ValueError("Bounding box coordinates must be ordered")
    overlap = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    area_first = (ax1 - ax0) * (ay1 - ay0)
    area_second = (bx1 - bx0) * (by1 - by0)
    union = area_first + area_second - overlap
    return overlap / union if union else 0.0


def layout_iou_metrics(
    predicted: list[dict[str, Any]], gold: list[dict[str, Any]] | None,
    coordinate_system: str, gold_coordinate_system: str | None,
) -> dict[str, Any]:
    if gold is None:
        return {"status": "not_evaluable", "reason": "missing_gold_bounding_boxes"}
    if coordinate_system != gold_coordinate_system:
        raise ValueError("Cannot compare layout boxes with different coordinate systems")
    result = match_items(predicted, gold, lambda p, g: bbox_iou(p["bbox"], g["bbox"]), threshold=0.5)
    return {"status": "evaluated", **result}


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    xmean, ymean = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - xmean) * (y - ymean) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - xmean) ** 2 for x in xs) * sum((y - ymean) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranked = [0.0] * len(values)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and ordered[end][1] == ordered[position][1]:
            end += 1
        mean_rank = (position + 1 + end) / 2
        for offset in range(position, end):
            ranked[ordered[offset][0]] = mean_rank
        position = end
    return ranked


def expert_agreement(
    predictions: list[dict[str, Any]], expert_labels: list[dict[str, Any]] | None,
    criteria: Sequence[str] = ("energy_source_correct", "action_order_correct", "missing_required_steps"),
) -> dict[str, Any]:
    """Compare matched explicit boolean judgments; never infer expert labels."""

    if expert_labels is None:
        return {"status": "not_evaluable", "reason": "missing_expert_judgments"}
    if not expert_labels:
        return {"status": "not_evaluable", "reason": "no_verified_expert_judgments"}
    index = {(str(row["document_id"]), str(row["procedure_id"])): row for row in expert_labels}
    results = {}
    for criterion in criteria:
        pairs = []
        for prediction in predictions:
            key = (str(prediction.get("document_id")), str(prediction.get("procedure_id")))
            if key not in index:
                continue
            actual = index[key].get(criterion)
            predicted = prediction.get(criterion)
            if isinstance(actual, bool) and isinstance(predicted, bool):
                pairs.append((float(predicted), float(actual)))
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        results[criterion] = {
            "n": len(pairs),
            "accuracy": sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
            "pearson_r": _pearson(xs, ys),
            "spearman_rho": _pearson(_ranks(xs), _ranks(ys)) if pairs else None,
        }
    return {"status": "evaluated", "criteria": results}
