"""Test the one-call VLM outputs without requiring a GPU or model weights."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from loto_pke.baselines import (
    add_visual_links,
    normalize_direct_graph,
    normalize_structured_baseline,
    run_vlm_baseline,
)


class FakeGenerator:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = [json.dumps(item) for item in responses]
        self.calls = []

    def generate(self, system: str, prompt: str, images=None) -> str:
        self.calls.append((system, prompt, images))
        return self.responses.pop(0)


class TestBaselines(unittest.TestCase):
    def test_structured_baseline_graph(self) -> None:
        graph = normalize_structured_baseline({"procedures": [{"title": "Lockout", "steps": [
            {"step_order": 1, "source_page": 1, "action": "Close E-1", "target": "E-1"},
            {"step_order": 2, "source_page": 1, "action": "Lock E-1", "tool": "lock"},
        ]}]}, "1", 1)
        self.assertEqual(len(graph["steps"]), 2)
        self.assertIn({"subject": "Close E-1", "relation": "acts_on", "object": "E-1"}, graph["triplets"])
        self.assertEqual(graph["output_mode"], "structured_steps_shared_graph_projection")

    def test_direct_graph_validation(self) -> None:
        answer = {"nodes": [
            {"id": "a", "type": "Action", "label": "Turn off"},
            {"id": "b", "type": "Target", "label": "E-1"},
        ], "edges": [{"source": "a", "relation": "acts_on", "target": "b"}]}
        graph = normalize_direct_graph(answer, "1")
        self.assertEqual(graph["triplets"], [{"subject": "Turn off", "relation": "acts_on", "object": "E-1"}])
        with self.assertRaises(ValueError):
            normalize_direct_graph({**answer, "edges": [{"source": "a", "relation": "uses", "target": "missing"}]}, "1")

    def test_visual_links_validated(self) -> None:
        graph = normalize_direct_graph({"nodes": [{"id": "a", "type": "Action", "label": "Lockout"}], "edges": []}, "1")
        updated = add_visual_links(graph, {"links": [{"node_id": "a", "page_number": 1, "bbox": [0.1, 0.1, 0.8, 0.9]}]}, 1)
        self.assertEqual(len(updated["edges"]), 1)
        self.assertEqual(updated["edges"][0]["bbox_coordinate_system"], "normalized_image_pixels")
        with self.assertRaises(ValueError):
            add_visual_links(graph, {"links": [{"node_id": "a", "page_number": 2}]}, 1)

    def test_end_to_end_single_pass(self) -> None:
        generator = FakeGenerator([{"nodes": [{"id": "a", "type": "Action", "label": "Stop machine"}], "edges": []}])
        result = run_vlm_baseline("1", [Path("page_1.png")], generator)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(len(generator.calls[0][2]), 1)


if __name__ == "__main__":
    unittest.main()
