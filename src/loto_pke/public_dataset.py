"""Read the supplied public LOTO annotations without changing source text.

No proprietary dataset paths, model dependencies, or implicit safety judgments
are required. Source files are hashed and referenced by relative paths only.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loto_pke.schema import (
    AnnotatedStep,
    ProcedureGroup,
    PublicDataset,
    PublicDocument,
    SourceAsset,
    ValidationIssue,
)


FIELDS = (
    "step_order",
    "action",
    "title",
    "tool",
    "target",
    "location",
    "description",
    "verification",
)
SOURCE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".pdf"}
FIELD_LINE = re.compile(r"^([a-z_]+)\s*:\s*(.*)$")
PAGE_IMAGE = re.compile(r"^page_(\d+)\.(png|jpg|jpeg|webp)$", re.IGNORECASE)


@dataclass(frozen=True)
class _RawStep:
    fields: dict[str, str | None]
    line_number: int


def _issue(
    severity: Literal["warning", "error"],
    code: str,
    message: str,
    document_id: str | None,
    line: int | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        severity=severity,
        code=code,
        message=message,
        document_id=document_id,
        annotation_line=line,
    )


def _nullable(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() == "null":
        return None
    return stripped


def _asset(path: Path, root: Path) -> SourceAsset:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return SourceAsset(
        path=path.relative_to(root).as_posix(),
        sha256=checksum.hexdigest(),
        size_bytes=path.stat().st_size,
    )


def _parse_annotation(
    path: Path,
    document_id: str,
) -> tuple[list[_RawStep], list[ValidationIssue]]:
    """Each step_order begins a record; blanks do not define boundaries."""

    issues: list[ValidationIssue] = []
    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        content = path.read_text(encoding="utf-8-sig", errors="replace")
        issues.append(
            _issue(
                "error",
                "INVALID_UTF8",
                "Annotation contains undecodable bytes; replacement characters were used.",
                document_id,
            )
        )

    records: list[_RawStep] = []
    current: dict[str, str | None] = {}
    current_line = 1
    previous_field: str | None = None

    def finish_record() -> None:
        if not current:
            return
        for field in ("step_order", "action", "title"):
            if field not in current or _nullable(current.get(field)) is None:
                severity = "warning" if field == "title" else "error"
                issues.append(
                    _issue(
                        severity,
                        "MISSING_" + field.upper(),
                        f"Missing or empty {field} in an annotation record.",
                        document_id,
                        current_line,
                    )
                )
        records.append(_RawStep(dict(current), current_line))

    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue

        match = FIELD_LINE.match(line)
        if match:
            name, value = match.groups()
            if name not in FIELDS:
                issues.append(
                    _issue(
                        "warning",
                        "UNKNOWN_FIELD",
                        f"Unrecognized annotation field: {name}",
                        document_id,
                        line_number,
                    )
                )
                previous_field = None
                continue

            if name == "step_order" and current:
                finish_record()
                current = {}

            if not current:
                current_line = line_number

            if name in current:
                issues.append(
                    _issue(
                        "error",
                        "DUPLICATE_FIELD",
                        f"Duplicate {name} field; the first value was retained.",
                        document_id,
                        line_number,
                    )
                )
            else:
                current[name] = _nullable(value)
            previous_field = name
        elif previous_field and current:
            # Preserve unanticipated wrapped values instead of dropping them.
            prior = current.get(previous_field) or ""
            current[previous_field] = prior + "\n" + line.strip()
            issues.append(
                _issue(
                    "warning",
                    "CONTINUATION_LINE",
                    f"Line was appended to {previous_field}; verify the annotation.",
                    document_id,
                    line_number,
                )
            )
        else:
            issues.append(
                _issue(
                    "error",
                    "UNPARSED_LINE",
                    "Line could not be associated with an annotation field.",
                    document_id,
                    line_number,
                )
            )

    finish_record()
    return records, issues


def _step_order(raw: _RawStep, document_id: str, issues: list[ValidationIssue]) -> int | None:
    original = raw.fields.get("step_order")
    if original is None:
        return None
    try:
        value = int(original)
    except ValueError:
        value = -1
    if value < 1:
        issues.append(
            _issue(
                "error",
                "INVALID_STEP_ORDER",
                f"Expected a positive integer, received {original!r}.",
                document_id,
                raw.line_number,
            )
        )
        return None
    return value


def _build_procedures(
    raw_steps: list[_RawStep],
    document_id: str,
    issues: list[ValidationIssue],
) -> tuple[ProcedureGroup, ...]:
    grouped: list[list[tuple[_RawStep, int | None]]] = []
    titles: list[str | None] = []
    previous_order: int | None = None

    for raw in raw_steps:
        order = _step_order(raw, document_id, issues)
        title = raw.fields.get("title")

        title_changed = bool(grouped and title and titles[-1] and title != titles[-1])
        order_reset = bool(grouped and order is not None and previous_order is not None and order <= previous_order)
        if not grouped or title_changed or order_reset:
            grouped.append([])
            titles.append(title)
            previous_order = None

        if titles[-1] is None and title is not None:
            titles[-1] = title

        if previous_order is not None and order is not None and order > previous_order + 1:
            issues.append(
                _issue(
                    "warning",
                    "STEP_ORDER_GAP",
                    f"Step order jumps from {previous_order} to {order} within a group.",
                    document_id,
                    raw.line_number,
                )
            )

        grouped[-1].append((raw, order))
        previous_order = order

    procedures: list[ProcedureGroup] = []
    source_index = 0
    for group_number, group in enumerate(grouped, start=1):
        procedure_id = f"public-{int(document_id):03d}-p{group_number:02d}"
        steps: list[AnnotatedStep] = []
        for step_index, (raw, order) in enumerate(group, start=1):
            source_index += 1
            fields = raw.fields
            steps.append(
                AnnotatedStep(
                    document_id=document_id,
                    procedure_id=procedure_id,
                    step_id=f"{procedure_id}-s{step_index:03d}",
                    source_index=source_index,
                    step_index_within_procedure=step_index,
                    step_order=order,
                    annotation_line=raw.line_number,
                    title=fields.get("title"),
                    action=fields.get("action"),
                    target=fields.get("target"),
                    tool=fields.get("tool"),
                    location=fields.get("location"),
                    description=fields.get("description"),
                    verification=fields.get("verification"),
                )
            )
        procedures.append(
            ProcedureGroup(
                document_id=document_id,
                procedure_id=procedure_id,
                procedure_index=group_number,
                title=titles[group_number - 1],
                steps=tuple(steps),
            )
        )

    return tuple(procedures)


def _document_sort_key(path: Path) -> tuple[int, int, str]:
    if path.name.isdecimal():
        return (0, int(path.name), path.name)
    return (1, 0, path.name)


def _load_document(
    folder: Path,
    root: Path,
    selected: bool,
) -> tuple[PublicDocument, list[ValidationIssue]]:
    document_id = folder.name
    issues: list[ValidationIssue] = []

    primary = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.stem == document_id and path.suffix.lower() in SOURCE_EXTENSIONS
    )
    source: SourceAsset | None = None
    source_kind: Literal["image", "pdf"] | None = None
    if len(primary) != 1:
        issues.append(
            _issue(
                "error",
                "MISSING_SOURCE" if not primary else "AMBIGUOUS_SOURCE",
                f"Expected one original source file; found {len(primary)}.",
                document_id,
            )
        )
    else:
        source = _asset(primary[0], root)
        source_kind = "pdf" if primary[0].suffix.lower() == ".pdf" else "image"

    rendered = [
        (int(match.group(1)), path)
        for path in folder.iterdir()
        if path.is_file() and (match := PAGE_IMAGE.fullmatch(path.name))
    ]
    rendered.sort(key=lambda item: (item[0], item[1].name))
    rendered_pages = tuple(_asset(path, root) for _, path in rendered)
    rendered_numbers = [number for number, _ in rendered]
    if rendered_numbers and rendered_numbers != list(range(1, len(rendered_numbers) + 1)):
        issues.append(
            _issue(
                "warning",
                "RENDERED_PAGE_SEQUENCE",
                "Rendered page images must have unique consecutive page numbers starting at 1.",
                document_id,
            )
        )

    annotation_path = folder / f"{document_id}.txt"
    annotation: SourceAsset | None = None
    procedures: tuple[ProcedureGroup, ...] = ()
    if annotation_path.exists():
        annotation = _asset(annotation_path, root)
        raw_steps, parsing_issues = _parse_annotation(annotation_path, document_id)
        issues.extend(parsing_issues)
        if not raw_steps:
            issues.append(
                _issue(
                    "error",
                    "EMPTY_ANNOTATION",
                    "Annotation file contains no recognizable step records.",
                    document_id,
                )
            )
        procedures = _build_procedures(raw_steps, document_id, issues)
    else:
        issues.append(
            _issue(
                "error",
                "MISSING_ANNOTATION",
                "No annotation text file matching the document ID was found.",
                document_id,
            )
        )

    return (
        PublicDocument(
            document_id=document_id,
            source_kind=source_kind,
            source=source,
            annotation=annotation,
            rendered_pages=rendered_pages,
            procedures=procedures,
            selected=selected,
        ),
        issues,
    )


def load_public_dataset(
    data_root: Path,
    selected_ids: set[str] | None = None,
) -> PublicDataset:
    """Inventory all public files, marking an optional subset as selected.

    A subset does not remove other documents from the manifest. Selection is
    therefore reversible and transparent when reconciling paper sample sizes.
    """

    root = data_root.expanduser().resolve()
    if root.name != "public_data" and (root / "public_data").is_dir():
        root = root / "public_data"
    if not root.is_dir():
        raise NotADirectoryError(f"Public-data directory not found: {root}")

    document_dirs = sorted(
        (path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=_document_sort_key,
    )
    issues: list[ValidationIssue] = []
    numeric_dirs = [folder for folder in document_dirs if folder.name.isdecimal()]
    for folder in document_dirs:
        if not folder.name.isdecimal():
            issues.append(
                _issue(
                    "warning",
                    "NON_NUMERIC_DOCUMENT_DIR",
                    f"Ignored directory: {folder.name}",
                    None,
                )
            )

    available_ids = {folder.name for folder in numeric_dirs}
    canonical_ids: dict[int, str] = {}
    for folder in numeric_dirs:
        key = int(folder.name)
        if key in canonical_ids:
            issues.append(
                _issue(
                    "error",
                    "DOCUMENT_ID_COLLISION",
                    f"Directories {canonical_ids[key]!r} and {folder.name!r} map to the same stable ID.",
                    folder.name,
                )
            )
        canonical_ids[key] = folder.name
    if selected_ids is not None:
        for unknown_id in sorted(selected_ids - available_ids):
            issues.append(
                _issue(
                    "error",
                    "UNKNOWN_SELECTED_ID",
                    f"Selected document ID is not present: {unknown_id}",
                    unknown_id,
                )
            )

    documents: list[PublicDocument] = []
    for folder in numeric_dirs:
        selected = selected_ids is None or folder.name in selected_ids
        document, doc_issues = _load_document(folder, root, selected)
        documents.append(document)
        issues.extend(doc_issues)

    return PublicDataset(
        documents=tuple(documents),
        issues=tuple(issues),
        requested_selection=tuple(sorted(selected_ids, key=lambda item: (not item.isdecimal(), int(item) if item.isdecimal() else item))) if selected_ids is not None else None,
    )


def read_selection_file(path: Path) -> set[str]:
    """Read a list of document IDs; blank lines and # comments are ignored."""

    selected = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.split("#", maxsplit=1)[0].strip()
        if value:
            selected.add(value)
    if not selected:
        raise ValueError(f"Selection file contains no document IDs: {path}")
    return selected
