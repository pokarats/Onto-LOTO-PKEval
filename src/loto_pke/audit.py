"""CLI for reproducible public-data manifests and annotation quality checks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from loto_pke.public_dataset import load_public_dataset, read_selection_file
from loto_pke.schema import PublicDataset


SLOT_FIELDS = ("action", "target", "tool", "location", "description", "verification", "title")


def summarize(dataset: PublicDataset) -> dict[str, Any]:
    documents = dataset.documents
    all_steps = [step for document in documents for procedure in document.procedures for step in procedure.steps]
    all_groups = [group for document in documents for group in document.procedures]
    selected_docs = [doc for doc in documents if doc.selected]

    source_kinds = Counter(doc.source_kind or "missing" for doc in documents)
    source_exts = Counter(
        Path(doc.source.path).suffix.lower()
        for doc in documents
        if doc.source is not None
    )
    issue_severity = Counter(issue.severity for issue in dataset.issues)
    issue_codes = Counter(issue.code for issue in dataset.issues)

    return {
        "schema_version": "public-loto-0.1.0",
        "dataset": "public_data",
        "document_count": len(documents),
        "document_ids": [doc.document_id for doc in documents],
        "selected_document_count": len(selected_docs),
        "selected_document_ids": [doc.document_id for doc in selected_docs],
        "excluded_document_ids": [doc.document_id for doc in documents if not doc.selected],
        "requested_selection": list(dataset.requested_selection) if dataset.requested_selection is not None else None,
        "inferred_procedure_group_count": len(all_groups),
        "annotated_step_record_count": len(all_steps),
        "selected_step_record_count": sum(doc.step_count for doc in selected_docs),
        "selected_inferred_procedure_group_count": sum(len(doc.procedures) for doc in selected_docs),
        "source_kind_counts": dict(sorted(source_kinds.items())),
        "source_extension_counts": dict(sorted(source_exts.items())),
        "available_visual_page_file_count": sum(
            1 if doc.source_kind == "image" else len(doc.rendered_pages)
            for doc in documents
        ),
        "non_null_field_counts": {
            name: sum(getattr(step, name) is not None for step in all_steps)
            for name in SLOT_FIELDS
        },
        "issue_severity_counts": dict(sorted(issue_severity.items())),
        "issue_code_counts": dict(sorted(issue_codes.items())),
        "visual_annotation_status": "No bounding-box or entity-to-image alignment labels in the supplied TXT schema",
        "grouping_rule": "New group on title change or non-increasing step_order; inferred, not manually verified",
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_audit(dataset: PublicDataset, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(dataset)

    _write_jsonl(output_dir / "documents.jsonl", (doc.to_record() for doc in dataset.documents))
    _write_jsonl(
        output_dir / "procedures.jsonl",
        (group.to_record(doc.selected) for doc in dataset.documents for group in doc.procedures),
    )
    _write_jsonl(
        output_dir / "steps.jsonl",
        (
            {**step.to_dict(), "selected": doc.selected}
            for doc in dataset.documents
            for group in doc.procedures
            for step in group.steps
        ),
    )
    _write_jsonl(output_dir / "issues.jsonl", (issue.to_dict() for issue in dataset.issues))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="loto-pke",
        description="Inventory and validate the public LOTO dataset without model inference.",
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Path to public_data or its parent")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for JSON and JSONL reports")
    parser.add_argument(
        "--selection-file",
        type=Path,
        help="Optional newline-separated document IDs defining the evaluation subset",
    )
    parser.add_argument("--fail-on-warning", action="store_true", help="Return a failure status for warnings too")
    args = parser.parse_args(argv)

    selection = read_selection_file(args.selection_file) if args.selection_file else None
    dataset = load_public_dataset(args.data_root, selected_ids=selection)
    summary = write_audit(dataset, args.output_dir)

    print(
        f"Public LOTO audit: {summary['document_count']} documents, "
        f"{summary['inferred_procedure_group_count']} inferred groups, "
        f"{summary['annotated_step_record_count']} step records."
    )
    print(f"Selected documents: {summary['selected_document_count']}")
    print(f"Warnings/errors: {summary['issue_severity_counts']}")
    print(f"Reports: {args.output_dir.resolve()}")

    if summary["issue_severity_counts"].get("error", 0):
        return 1
    if args.fail_on_warning and summary["issue_severity_counts"].get("warning", 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
