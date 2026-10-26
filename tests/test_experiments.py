"""Dependency-light tests for each scientific stage and failure policy."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from loto_pke.baselines import normalize_direct_graph
from loto_pke.comparison import compare_pipelines
from loto_pke.common import read_jsonl
from loto_pke.evaluation import (
    evaluate_expert_agreement,
    evaluate_graphs,
    evaluate_instructions,
    evaluate_parsing,
    evaluate_slots,
    expert_review_template,
)
from loto_pke.extraction import decompose_action, extract_instructions, extract_slots
from loto_pke.metrics import TextSimilarity, bbox_iou, match_items
from loto_pke.pipeline import _run_items
from loto_pke.pkg import build_graph
from loto_pke.safety import deterministic_safety_audit
from loto_pke.vision import ImageCandidate, grounding_accuracy, rank_visual_evidence


class FakeGenerator:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)

    def generate(self, system: str, prompt: str, images=None) -> str:
        return json.dumps(self.responses.pop(0))


def fixture_steps() -> list[dict]:
    return [
        {"document_id": "1", "procedure_id": "p1", "step_id": "s1", "action": "Open E-1 disconnect",
         "target": "E-1", "tool": None, "location": "east panel", "verification": None},
        {"document_id": "1", "procedure_id": "p1", "step_id": "s2", "action": "Apply padlock to E-1",
         "target": "E-1", "tool": "padlock", "verification": "Verify lock holds"},
    ]


class TestExperiments(unittest.TestCase):
    def test_two_stage_extraction(self) -> None:
        parsed = {"document_id": "1", "backend": "tesseract", "pages": [
            {"page_number": 1, "text": "1. Open E-1 disconnect"},
        ]}
        first = FakeGenerator([{"procedures": [{"title": "LOTO", "steps": [
            {"step_order": 1, "action": "Open E-1 disconnect", "step_type": "executable"},
        ]}]}])
        instructions = extract_instructions(parsed, first)
        self.assertEqual(len(instructions["procedures"]), 1)
        second = FakeGenerator([{"action": "Open", "target": "E-1", "tool": None,
                                 "location": None, "verification": None, "description": None,
                                 "energy_type": "electrical", "step_type": "executable"}])
        slots = extract_slots(instructions, second)
        self.assertEqual(slots["steps"][0]["predicted_action"], "Open")
        graph = build_graph("1", slots["steps"])
        self.assertIn({"subject": "Open", "relation": "acts_on", "object": "E-1"}, graph["triplets"])
        self.assertEqual(graph["nodes"][0]["label"], "Open E-1 disconnect")

    def test_atomic_decomposition_arguments_and_precedence(self) -> None:
        fake = FakeGenerator([{"atoms": [
            {"verb": "open", "target": "disconnect", "tool": None, "location": "cabinet"},
            {"verb": "lock", "target": "disconnect", "tool": "padlock", "location": None},
        ]}])
        atoms = decompose_action("Open and lock disconnect", fake)
        graph = build_graph("1", [{**fixture_steps()[0], "atoms": atoms}])
        self.assertIn({"subject": "lock", "relation": "uses", "object": "padlock"}, graph["triplets"])
        self.assertIn({"subject": "open", "relation": "precedes", "object": "lock"}, graph["triplets"])

    def test_graph_scores_self_match(self) -> None:
        graph = build_graph("1", fixture_steps())
        report = evaluate_graphs([graph], [graph], methods=("exact", "fuzzy"))
        self.assertEqual(report["aggregate"]["exact"]["triplet_micro"]["f1"], 1.0)
        self.assertEqual(report["aggregate"]["fuzzy"]["graph_overlap_exact_macro"]["edge_jaccard"], 1.0)

    def test_procedure_index_alignment_and_unsupported_direct_graph(self) -> None:
        graph = build_graph("1", fixture_steps())
        report = evaluate_graphs([graph], [graph], methods=("exact",), granularity="procedure_index")
        self.assertEqual(report["aggregate"]["exact"]["unit_count"], 1)
        direct = normalize_direct_graph({"nodes": [{"id": "a", "type": "Action", "label": "Open"}], "edges": []}, "1")
        unavailable = evaluate_graphs([direct], [graph], methods=("exact",), granularity="procedure_index")
        self.assertEqual(len(unavailable["not_evaluable"]), 1)

    def test_paired_pipeline_bootstrap_comparison(self) -> None:
        gold = build_graph("1", fixture_steps())
        baseline = normalize_direct_graph({"nodes": [
            {"id": "a", "type": "Action", "label": "Something unrelated"},
        ], "edges": []}, "1")
        report = compare_pipelines([gold], [{"document_id": "1", "graph": baseline}], [gold],
                                   method="exact", threshold=1.0, bootstrap_samples=100)
        self.assertEqual(report["n_common_evaluable_documents"], 1)
        self.assertGreater(report["triplet_f1_delta_mean"], 0)
        self.assertIsNone(report["cross_modal_f1"])

    def test_instruction_and_slot_scores(self) -> None:
        record = {"document_id": "1", "steps": fixture_steps()}
        self.assertEqual(evaluate_instructions([record], [record], TextSimilarity("fuzzy"))["macro"]["diag_mean"], 1.0)
        self.assertEqual(evaluate_slots([record], [record], TextSimilarity("fuzzy"), 0.85)
                         ["aggregate"]["target"]["micro"]["f1"], 1.0)

    def test_missing_layout_and_expert_gold_are_not_fake_scores(self) -> None:
        no_layout = evaluate_parsing([{"document_id": "1", "pages": []}], None)
        self.assertEqual(no_layout["status"], "not_evaluable")
        self.assertIsNone(no_layout["smart_1_f1"])
        no_expert = evaluate_expert_agreement([], None)
        self.assertEqual(no_expert["status"], "not_evaluable")

    def test_bbox_coordinates_and_one_to_one_matching(self) -> None:
        self.assertEqual(bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(match_items(["one", "one"], ["one"], TextSimilarity("exact").score, 1)["matched"], 1)

    def test_exploratory_safety_flags_and_expert_tasks(self) -> None:
        graph = build_graph("1", fixture_steps())
        rules = {
            "expert_verified": False,
            "action_categories": {"isolate": ["open"], "lock": ["padlock"], "verify": ["verify"]},
            "required_categories": ["isolate", "lock", "verify"],
            "ordered_pairs": [["isolate", "lock"]],
            "energy_types": {"electrical": ["disconnect"]},
        }
        safety = deterministic_safety_audit(graph, rules)
        block = safety["procedure_blocks"][0]
        self.assertTrue(block["missing_required_steps"])
        self.assertTrue(block["action_order_correct"])
        self.assertEqual(safety["safety_certification"], "NOT_PERFORMED")
        task = expert_review_template([safety])[0]
        self.assertEqual(len(task["ordered_steps"]), 2)
        self.assertEqual(task["annotation_status"], "needs_certified_expert_review")

    def test_verified_expert_agreement_requires_manual_status(self) -> None:
        review = {"document_id": "1", "procedure_blocks": [{
            "procedure_id": "p1", "energy_source_correct": True,
            "action_order_correct": True, "missing_required_steps": False,
        }]}
        gold = [{"document_id": "1", "procedure_id": "p1", "annotation_status": "verified",
                 "energy_source_correct": "Partially", "action_order_correct": "Yes", "missing_required_steps": "No"}]
        report = evaluate_expert_agreement([review], gold)
        self.assertEqual(report["criteria"]["energy_source_correct"]["accuracy"], 0.0)
        self.assertEqual(report["criteria"]["action_order_correct"]["accuracy"], 1.0)

    def test_visual_retrieval_proxy_separate_from_gold(self) -> None:
        class FakeScorer:
            method = "synthetic_cosine"

            def score(self, image_path: Path, text: str) -> float:
                return 0.9 if image_path.stem == "page1" else 0.3

        candidates = [ImageCandidate("1:page1", "1", Path("page1.png")),
                      ImageCandidate("2:page1", "2", Path("other.png"))]
        ranking = rank_visual_evidence([{"step_id": "s1", "document_id": "1", "text": "Open"}], candidates, FakeScorer())
        self.assertTrue(ranking[0]["document_proxy_hit_at_1"])
        self.assertIsNone(ranking[0]["fine_grained_acc_at_k"])
        self.assertEqual(grounding_accuracy(ranking, None)["status"], "not_evaluable")
        verified = [{"step_id": "s1", "gold_image_ids": ["1:page1"], "annotation_status": "verified"}]
        self.assertEqual(grounding_accuracy(ranking, verified)["acc_at_1"], 1.0)

    def test_incremental_output_and_resume_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stage.jsonl"
            args = argparse.Namespace(
                output=output, overwrite=False, resume=False, retry_errors=False, command="mock",
            )
            counter = {"n": 0}

            def generate(item: str) -> dict:
                counter["n"] += 1
                return {"document_id": item, "status": "ok"}

            _run_items(args, ["1", "2"], lambda x: x, generate)
            self.assertEqual(counter["n"], 2)
            self.assertEqual(len(read_jsonl(output)), 2)
            with self.assertRaises(FileExistsError):
                _run_items(args, ["1", "2"], lambda x: x, generate)
            args.resume = True
            _run_items(args, ["1", "2"], lambda x: x, generate)
            self.assertEqual(counter["n"], 2)


if __name__ == "__main__":
    unittest.main()
