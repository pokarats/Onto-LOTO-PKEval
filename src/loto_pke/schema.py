"""Stable, model-independent data structures for public LOTO annotations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class SourceAsset:
    """A file referenced relative to the public_data directory."""

    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationIssue:
    severity: Literal["warning", "error"]
    code: str
    message: str
    document_id: str | None = None
    annotation_line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnnotatedStep:
    """One source-faithful annotation, not necessarily an atomic action."""

    document_id: str
    procedure_id: str
    step_id: str
    source_index: int
    step_index_within_procedure: int
    step_order: int | None
    annotation_line: int
    title: str | None
    action: str | None
    target: str | None
    tool: str | None
    location: str | None
    description: str | None
    verification: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProcedureGroup:
    """A contiguous title/order segment inferred from an annotation file."""

    document_id: str
    procedure_id: str
    procedure_index: int
    title: str | None
    steps: tuple[AnnotatedStep, ...]

    def to_record(self, selected: bool) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "procedure_id": self.procedure_id,
            "procedure_index": self.procedure_index,
            "title": self.title,
            "selected": selected,
            "step_ids": [step.step_id for step in self.steps],
            "step_count": len(self.steps),
        }


@dataclass(frozen=True)
class PublicDocument:
    document_id: str
    source_kind: Literal["image", "pdf"] | None
    source: SourceAsset | None
    annotation: SourceAsset | None
    rendered_pages: tuple[SourceAsset, ...]
    procedures: tuple[ProcedureGroup, ...]
    selected: bool

    @property
    def step_count(self) -> int:
        return sum(len(group.steps) for group in self.procedures)

    def to_record(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "selected": self.selected,
            "source_kind": self.source_kind,
            "source": self.source.to_dict() if self.source else None,
            "annotation": self.annotation.to_dict() if self.annotation else None,
            "rendered_pages": [asset.to_dict() for asset in self.rendered_pages],
            "procedure_ids": [group.procedure_id for group in self.procedures],
            "inferred_procedure_group_count": len(self.procedures),
            "annotated_step_count": self.step_count,
        }


@dataclass(frozen=True)
class PublicDataset:
    documents: tuple[PublicDocument, ...]
    issues: tuple[ValidationIssue, ...]
    requested_selection: tuple[str, ...] | None
