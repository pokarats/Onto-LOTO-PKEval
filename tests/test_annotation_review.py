"""Independent annotator and four-label human-review evaluation."""

from __future__ import annotations

import unittest

from loto_pke.annotation_agreement import inter_annotator_agreement
from loto_pke.human_review import human_review_template, summarize_human_review


class TestAnnotationReview(unittest.TestCase):
    def setUp(self) -> None:
        self.annotator_a = [{
            "document_id": "1", "procedure_id": "p1", "step_id": "s1", "action": "Close valve",
            "target": "Main valve", "tool": "padlock", "location": "pump room",
        }]
        self.annotator_b = [{
            "document_id": "1", "procedure_id": "p1", "step_id": "s1", "action": "Close valve",
            "target": "main valve", "tool": "padlock", "location": "pump room",
        }]

    def test_exact_agreement_and_missing_concept_mapping(self) -> None:
        result = inter_annotator_agreement(self.annotator_a, self.annotator_b)
        self.assertEqual(result["exact_lexical"]["field_agreement"]["target"]["exact_or_normalized_accuracy"], 0.0)
        self.assertIsNone(result["conceptual_normalized"])

    def test_explicit_expert_concept_mapping(self) -> None:
        mapping = {"version": "v1", "approved_by": "trained reviewer", "target": {"main valve": "VALVE_MAIN"}}
        result = inter_annotator_agreement(self.annotator_a, self.annotator_b, mapping)
        self.assertEqual(result["conceptual_normalized"]["field_agreement"]["target"]["exact_or_normalized_accuracy"], 1.0)

    def test_unapproved_map_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            inter_annotator_agreement(self.annotator_a, self.annotator_b, {"target": {"main valve": "VALVE_MAIN"}})

    def test_human_task_and_summary(self) -> None:
        records = [{"document_id": "1", "steps": self.annotator_a}]
        task = human_review_template(records)[0]
        self.assertEqual(task["predicted_slots"]["target"], "Main valve")
        self.assertEqual(task["annotation_status"], "needs_human_review")
        task["annotation_status"] = "verified"
        task["instruction_rating"] = "CORRECT"
        task["slot_ratings"]["target"] = "PARTIAL"
        result = summarize_human_review([task])
        self.assertEqual(result["criteria"]["instruction"]["correct_rate"], 1.0)
        self.assertEqual(result["criteria"]["target"]["correct_or_partial_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
