"""Public document parsing with separate PDF, OCR, and Docling backends.

Image-only sources are never silently treated as text-native PDFs. Bboxes
include a coordinate system to prevent invalid cross-backend IoU comparisons.
"""

from __future__ import annotations

import csv
import io
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from loto_pke.common import require_module
from loto_pke.schema import PublicDocument


def _absolute_asset(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    path = (resolved_root / relative_path).resolve()
    if not path.is_relative_to(resolved_root):
        raise ValueError(f"Asset escapes public-data directory: {relative_path}")
    return path


def source_images(document: PublicDocument, root: Path, tmpdir: Path | None = None) -> list[Path]:
    """Return each page image, rendering a PDF only when no renders exist."""

    if document.source is None:
        raise ValueError(f"Document {document.document_id} has no source file")
    if document.source_kind == "image":
        return [_absolute_asset(root, document.source.path)]
    if document.rendered_pages:
        return [_absolute_asset(root, page.path) for page in document.rendered_pages]
    if tmpdir is None:
        raise ValueError("A temporary directory is required to render a PDF")

    fitz = require_module("fitz", "Install pymupdf")
    source = _absolute_asset(root, document.source.path)
    images: list[Path] = []
    with fitz.open(source) as pdf:
        for index, page in enumerate(pdf, start=1):
            path = tmpdir / f"page_{index}.png"
            page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False).save(str(path))
            images.append(path)
    return images


def parse_pymupdf(document: PublicDocument, root: Path) -> dict[str, Any]:
    """Text/layout from PDF pages; raster-only sources have no extractable text."""

    if document.source is None:
        raise ValueError("Document source missing")
    if document.source_kind != "pdf":
        return {
            "document_id": document.document_id,
            "backend": "pymupdf",
            "text": "",
            "pages": [{"page_number": 1, "text": "", "blocks": [], "coordinate_system": "image_pixels"}],
            "warnings": ["Raster source has no native PDF text; use tesseract OCR or Docling explicitly."],
        }

    fitz = require_module("fitz", "Install pymupdf")
    source = _absolute_asset(root, document.source.path)
    pages = []
    with fitz.open(source) as pdf:
        for index, page in enumerate(pdf, start=1):
            blocks = [
                {"text": str(item[4]).strip(), "bbox": [float(value) for value in item[:4]], "kind": "text"}
                for item in page.get_text("blocks", sort=True)
                if len(item) >= 5 and isinstance(item[4], str) and item[4].strip()
            ]
            pages.append(
                {
                    "page_number": index,
                    "width": float(page.rect.width),
                    "height": float(page.rect.height),
                    "coordinate_system": "pdf_points_top_left",
                    "text": "\n".join(block["text"] for block in blocks),
                    "blocks": blocks,
                }
            )
    return {
        "document_id": document.document_id,
        "backend": "pymupdf",
        "text": "\n\n".join(page["text"] for page in pages),
        "pages": pages,
        "warnings": [],
    }


def _tesseract_tsv(image_path: Path, language: str) -> list[dict[str, Any]]:
    """Run installed Tesseract and group word boxes into reading-order lines."""

    completed = subprocess.run(
        ["tesseract", str(image_path), "stdout", "-l", language, "tsv"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Tesseract failed for {image_path.name}: {completed.stderr.strip()}")

    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    reader = csv.DictReader(io.StringIO(completed.stdout), delimiter="\t")
    for row in reader:
        token = (row.get("text") or "").strip()
        if not token:
            continue
        try:
            confidence = float(row["conf"])
            if confidence < 0:
                continue
            word = {
                "text": token,
                "bbox": [
                    int(row["left"]), int(row["top"]),
                    int(row["left"]) + int(row["width"]),
                    int(row["top"]) + int(row["height"]),
                ],
                "confidence": confidence,
            }
            group = (int(row["block_num"]), int(row["par_num"]), int(row["line_num"]))
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Invalid Tesseract TSV row: {row}") from exc
        groups[group].append(word)

    blocks = []
    for _, words in sorted(groups.items()):
        x0 = min(item["bbox"][0] for item in words)
        y0 = min(item["bbox"][1] for item in words)
        x1 = max(item["bbox"][2] for item in words)
        y1 = max(item["bbox"][3] for item in words)
        blocks.append(
            {
                "text": " ".join(word["text"] for word in words),
                "bbox": [x0, y0, x1, y1],
                "kind": "ocr_line",
                "confidence": sum(word["confidence"] for word in words) / len(words),
            }
        )
    return blocks


def parse_tesseract(document: PublicDocument, root: Path, language: str = "eng") -> dict[str, Any]:
    image_module = require_module("PIL.Image", "Install Pillow")
    pages = []
    with tempfile.TemporaryDirectory(prefix="loto_ocr_") as directory:
        tmpdir = Path(directory)
        for index, image_path in enumerate(source_images(document, root, tmpdir), start=1):
            with image_module.open(image_path) as image:
                converted = image.convert("RGB")
                width, height = converted.size
                temporary_png = tmpdir / f"ocr_input_{index}.png"
                converted.save(temporary_png)
            blocks = _tesseract_tsv(temporary_png, language)
            pages.append(
                {
                    "page_number": index,
                    "width": width,
                    "height": height,
                    "coordinate_system": "image_pixels",
                    "text": "\n".join(item["text"] for item in blocks),
                    "blocks": blocks,
                }
            )
    return {
        "document_id": document.document_id,
        "backend": "tesseract",
        "text": "\n\n".join(page["text"] for page in pages),
        "pages": pages,
        "warnings": [],
        "ocr_language": language,
    }


def parse_docling(document: PublicDocument, root: Path) -> dict[str, Any]:
    docling = require_module("docling.document_converter", "Install docling")
    if document.source is None:
        raise ValueError("Document source missing")
    source = _absolute_asset(root, document.source.path)

    with tempfile.TemporaryDirectory(prefix="loto_docling_") as directory:
        # Some Docling installations do not accept WebP; convert the format,
        # preserving the original source file and provenance.
        converted = False
        if source.suffix.lower() == ".webp":
            image_module = require_module("PIL.Image", "Install Pillow")
            converted_source = Path(directory) / f"{source.stem}.png"
            with image_module.open(source) as image:
                image.convert("RGB").save(converted_source)
            source = converted_source
            converted = True

        result = docling.DocumentConverter().convert(str(source))
        doc = result.document
        text = doc.export_to_markdown()
        pages: dict[int, dict[str, Any]] = {}
        # Docling provenance uses page_no and bbox with l/t/r/b.  Text
        # remains available even if a newer Docling schema changes provenance.
        for item in getattr(doc, "texts", []):
            value = getattr(item, "text", "")
            for provenance in getattr(item, "prov", []) or []:
                number = int(getattr(provenance, "page_no", 1))
                page = pages.setdefault(
                    number,
                    {"page_number": number, "text": "", "blocks": [], "coordinate_system": "docling_native"},
                )
                bbox = getattr(provenance, "bbox", None)
                coordinates = (
                    [float(getattr(bbox, field)) for field in ("l", "t", "r", "b")]
                    if bbox is not None else None
                )
                page["blocks"].append({"text": value, "bbox": coordinates, "kind": "text"})
                page["text"] += value + "\n"
        if not pages:
            pages[1] = {
                "page_number": 1, "text": text, "blocks": [], "coordinate_system": "docling_native",
            }
    return {
        "document_id": document.document_id,
        "backend": "docling",
        "text": text,
        "pages": [pages[number] for number in sorted(pages)],
        "warnings": ["WebP transcoded to PNG before parsing"] if converted else [],
    }


def parse_document(
    document: PublicDocument,
    data_root: Path,
    backend: str,
    ocr_language: str = "eng",
) -> dict[str, Any]:
    if backend == "pymupdf":
        return parse_pymupdf(document, data_root)
    if backend == "tesseract":
        return parse_tesseract(document, data_root, ocr_language)
    if backend == "docling":
        return parse_docling(document, data_root)
    raise ValueError(f"Unsupported parsing backend: {backend}")
