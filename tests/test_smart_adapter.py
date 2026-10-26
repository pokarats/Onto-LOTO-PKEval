"""SMART adapter contract tests (the real optional package is not bundled)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from loto_pke.smart_optional import evaluate_smart_parsing, smart_scores


class FakeSmartScorer:
    def __init__(self, matching_fn) -> None:
        self.matching_fn = matching_fn

    def smart_score(self, reference: str, candidate: str) -> dict:
        return {"smart1": {"f1": self.matching_fn(reference, candidate)}}


class TestSmartAdapter(unittest.TestCase):
    def test_chrf_official_api_call_shape(self) -> None:
        scoring = SimpleNamespace(SmartScorer=FakeSmartScorer)
        matching = SimpleNamespace(chrf_matcher=lambda a, b: 1.0 if a == b else 0.0)
        with patch("loto_pke.smart_optional.require_module", side_effect=[scoring, matching]):
            report = smart_scores("Reference", "Reference", matcher="chrf")
        self.assertEqual(report["raw_smart_scores"]["smart1"]["f1"], 1.0)
        self.assertFalse(report["paper_bleurt20_parity_verified"])

    def test_missing_reference_does_not_need_smart_dependency(self) -> None:
        result = evaluate_smart_parsing(
            [{"document_id": "1", "pages": [{"page_number": 1, "text": "Text"}]}],
            [{"document_id": "1", "page_number": 1,
              "annotation_status": "needs_human_annotation", "gold_page_text": None}],
        )
        self.assertEqual(result["n_evaluated_pages"], 0)
        self.assertIn("not_evaluable", result["status"])


if __name__ == "__main__":
    unittest.main()
