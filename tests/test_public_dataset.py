"""Regression tests for public-only dataset ingestion and audit outputs."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from loto_pke.audit import summarize, write_audit
from loto_pke.public_dataset import load_public_dataset, read_selection_file


class PublicDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "public_data"
        self.root.mkdir()

    def _add_document(self, document_id: str, annotation: str, extension: str = ".webp") -> None:
        folder = self.root / document_id
        folder.mkdir()
        (folder / f"{document_id}{extension}").write_bytes(b"synthetic source bytes")
        (folder / f"{document_id}.txt").write_text(annotation, encoding="utf-8")

    def test_preserves_multiple_procedures_and_local_step_numbers(self) -> None:
        self._add_document(
            "1",
            """step_order: 1
action: Shut down equipment.
title: Application
tool: null
target: equipment
location: null
description: null
verification: null

step_order: 2
action: Turn switch to OFF: disconnect power.
title: Application
tool: padlock
target: switch
location: control panel
description: null
verification: null

step_order: 1
action: Check personnel are safe.
title: Removal
tool: null
target: personnel
location: null
description: null
verification: null
""",
        )

        dataset = load_public_dataset(self.root)
        document = dataset.documents[0]

        self.assertEqual(len(document.procedures), 2)
        self.assertEqual([step.step_order for group in document.procedures for step in group.steps], [1, 2, 1])
        self.assertEqual(document.procedures[0].title, "Application")
        self.assertEqual(document.procedures[1].title, "Removal")
        self.assertEqual(document.procedures[0].steps[1].action, "Turn switch to OFF: disconnect power.")
        self.assertEqual(document.procedures[0].steps[1].tool, "padlock")
        self.assertEqual(len({step.step_id for group in document.procedures for step in group.steps}), 3)
        self.assertEqual(dataset.issues, ())

    def test_missing_title_is_flagged_without_invention(self) -> None:
        self._add_document(
            "3",
            """step_order: 1
action: Isolate power.
title: Machine shutdown
target: breaker

step_order: 2
action: Verify isolation.
target: breaker
""",
        )

        dataset = load_public_dataset(self.root)
        self.assertEqual(len(dataset.documents[0].procedures), 1)
        second_step = dataset.documents[0].procedures[0].steps[1]
        self.assertIsNone(second_step.title)
        self.assertEqual([issue.code for issue in dataset.issues], ["MISSING_TITLE"])
        self.assertEqual(dataset.issues[0].annotation_line, 6)

    def test_pdf_and_rendered_pages_are_distinct_assets(self) -> None:
        self._add_document("20", "step_order: 1\naction: Lock out.\ntitle: Steps\n", extension=".pdf")
        folder = self.root / "20"
        (folder / "page_2.png").write_bytes(b"second rendered page")
        (folder / "page_1.png").write_bytes(b"first rendered page")

        dataset = load_public_dataset(self.root)
        document = dataset.documents[0]
        self.assertEqual(document.source_kind, "pdf")
        self.assertEqual(document.source.path, "20/20.pdf")
        self.assertEqual([asset.path for asset in document.rendered_pages], ["20/page_1.png", "20/page_2.png"])
        self.assertEqual(len(document.source.sha256), 64)
        self.assertEqual(summarize(dataset)["available_visual_page_file_count"], 2)

    def test_selection_marks_but_does_not_drop_excluded_documents(self) -> None:
        for document_id in ("1", "2"):
            self._add_document(
                document_id,
                "step_order: 1\naction: Verify zero energy.\ntitle: Steps\n",
            )

        selection_file = Path(self.temporary.name) / "selected.txt"
        selection_file.write_text("# evaluation subset\n2\n", encoding="utf-8")
        dataset = load_public_dataset(self.root, selected_ids=read_selection_file(selection_file))
        summary = summarize(dataset)

        self.assertEqual(summary["document_count"], 2)
        self.assertEqual(summary["selected_document_ids"], ["2"])
        self.assertEqual(summary["excluded_document_ids"], ["1"])
        self.assertEqual(summary["selected_step_record_count"], 1)

        output_dir = Path(self.temporary.name) / "report"
        write_audit(dataset, output_dir)
        steps = [json.loads(line) for line in (output_dir / "steps.jsonl").read_text().splitlines()]
        self.assertEqual([step["selected"] for step in steps], [False, True])
        self.assertNotIn(str(self.root), (output_dir / "documents.jsonl").read_text())

    def test_unknown_selection_and_missing_source_are_reported(self) -> None:
        folder = self.root / "4"
        folder.mkdir()
        (folder / "4.txt").write_text("step_order: 1\naction: Stop.\ntitle: Steps\n")

        dataset = load_public_dataset(self.root, selected_ids={"4", "99"})
        self.assertEqual(
            {issue.code for issue in dataset.issues},
            {"UNKNOWN_SELECTED_ID", "MISSING_SOURCE"},
        )
        self.assertEqual(dataset.documents[0].source, None)

    def test_duplicate_field_does_not_silently_overwrite(self) -> None:
        self._add_document(
            "5",
            """step_order: 1
action: Close valve.
action: Open valve.
title: Steps
""",
        )

        dataset = load_public_dataset(self.root)
        self.assertIn("DUPLICATE_FIELD", {issue.code for issue in dataset.issues})
        self.assertEqual(dataset.documents[0].procedures[0].steps[0].action, "Close valve.")

    def test_empty_annotation_and_page_gap_are_reported(self) -> None:
        self._add_document("20", "\n", extension=".pdf")
        folder = self.root / "20"
        (folder / "page_1.png").write_bytes(b"first")
        (folder / "page_3.png").write_bytes(b"third")

        dataset = load_public_dataset(self.root)
        self.assertEqual(
            {issue.code for issue in dataset.issues},
            {"EMPTY_ANNOTATION", "RENDERED_PAGE_SEQUENCE"},
        )


if __name__ == "__main__":
    unittest.main()
