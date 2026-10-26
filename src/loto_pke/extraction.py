"""Versioned two-stage instruction and semantic-role extraction."""

from __future__ import annotations

import json
from typing import Any

from loto_pke.common import sha256_text
from loto_pke.llm import TextGenerator, parse_json_response


SYSTEM_PROMPT = (
    "You extract procedural knowledge from industrial Lockout-Tagout documents. "
    "Use ONLY explicitly visible or written evidence; never infer missing safety steps, "
    "tools, targets, device IDs, or locations. Preserve identifiers exactly. "
    "Return valid JSON with null for unknown values."
)

INSTRUCTION_PROMPT_VERSION = "instruction-v1"
INSTRUCTION_PROMPT = """Identify actionable and conditional LOTO instructions in the following page.
Return JSON with this structure:
{{"procedures":[{{"title":"Section title or null","steps":[{{"step_order":1,
"action":"document-grounded instruction","step_type":"executable|conditional|other"}}]}}]}}
Keep the original reading order. Step numbers may restart for different sections.
Do not summarize multiple steps into one or add implicit actions.

Examples:
Input: "1. Open the disconnect and lock out. 2. Verify zero voltage."
Output: {{"procedures":[{{"title":null,"steps":[
{{"step_order":1,"action":"Open the disconnect and lock out.","step_type":"executable"}},
{{"step_order":2,"action":"Verify zero voltage.","step_type":"executable"}}]}}]}}

DOCUMENT ID: {document_id}
PAGE: {page_number}
SOURCE TEXT:
{text}
"""

SLOT_PROMPT_VERSION = "slots-v1"
SLOT_PROMPT = """Extract semantic roles for this ONE documented procedural instruction.
Use a null value for an unstated role. Keep the original equipment identifiers.
Return JSON with exactly these keys: action, target, tool, location,
verification, description, energy_type, step_type.

Definitions: action = operation; target = acted-upon component; tool = required
physical tool; location = place; verification = explicitly stated check;
description = energy or contextual annotation; energy_type = textual energy
category if explicit; step_type = executable, conditional, or other.

Example:
Instruction: "Turn Ball Valve W-1 closed and secure with a padlock."
Output: {{"action":"Turn Ball Valve W-1 closed and secure with a padlock.",
"target":"Ball Valve W-1","tool":"padlock","location":null,
"verification":null,"description":null,"energy_type":null,"step_type":"executable"}}

DOCUMENT ID: {document_id}
PROCEDURE: {title}
INSTRUCTION: {action}
"""

ATOM_PROMPT_VERSION = "atoms-v1"
ATOM_PROMPT = """Decompose this documented instruction into minimal ordered atomic actions.
Do not invent steps or arguments. Return JSON:
{{"atoms":[{{"verb":"action verb","target":null,"tool":null,"location":null}}]}}

Instruction: {action}
"""


def _nullable(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected text or null; got {type(value).__name__}")
    cleaned = value.strip()
    return cleaned if cleaned and cleaned.lower() != "null" else None


def normalize_procedures(
    result: dict[str, Any], document_id: str, page_number: int,
) -> list[dict[str, Any]]:
    procedures = result.get("procedures")
    if not isinstance(procedures, list):
        raise ValueError("Model output requires a procedures list")

    normalized: list[dict[str, Any]] = []
    for procedure_index, proc in enumerate(procedures, start=1):
        if not isinstance(proc, dict) or not isinstance(proc.get("steps"), list):
            raise ValueError("Each procedure requires a steps list")
        title = _nullable(proc.get("title"))
        steps = []
        for step_index, step in enumerate(proc["steps"], start=1):
            if not isinstance(step, dict):
                raise ValueError("Each step must be a JSON object")
            action = _nullable(step.get("action"))
            if action is None:
                raise ValueError("Model returned an empty action")
            order = step.get("step_order")
            if order is not None and (not isinstance(order, int) or order < 1):
                raise ValueError(f"Invalid step_order: {order!r}")
            identifier = f"public-{int(document_id):03d}-pg{page_number:03d}-p{procedure_index:02d}-s{step_index:03d}"
            steps.append(
                {
                    "document_id": document_id,
                    "procedure_id": f"public-{int(document_id):03d}-pg{page_number:03d}-p{procedure_index:02d}",
                    "step_id": identifier,
                    "source_page": page_number,
                    "step_order": order,
                    "source_index": step_index,
                    "title": title,
                    "action": action,
                    "step_type": _nullable(step.get("step_type")),
                }
            )
        normalized.append({"title": title, "page_number": page_number, "steps": steps})
    return normalized


def extract_instructions(parsed: dict[str, Any], generator: TextGenerator) -> dict[str, Any]:
    document_id = str(parsed["document_id"])
    pages = parsed.get("pages", [])
    procedures: list[dict[str, Any]] = []
    traces = []
    for page in pages:
        page_number = int(page["page_number"])
        text = str(page.get("text") or "").strip()
        if not text:
            traces.append({"page_number": page_number, "status": "no_extractable_text"})
            continue
        prompt = INSTRUCTION_PROMPT.format(document_id=document_id, page_number=page_number, text=text)
        raw = generator.generate(SYSTEM_PROMPT, prompt)
        try:
            parsed_answer = parse_json_response(raw)
            predictions = normalize_procedures(parsed_answer, document_id, page_number)
            procedures.extend(predictions)
            traces.append({"page_number": page_number, "status": "ok", "raw_response": raw})
        except (ValueError, json.JSONDecodeError) as exc:
            traces.append({"page_number": page_number, "status": "invalid_json_or_schema", "error": str(exc), "raw_response": raw})
    return {
        "document_id": document_id,
        "stage": "instruction_extraction",
        "backend": parsed.get("backend"),
        "prompt_version": INSTRUCTION_PROMPT_VERSION,
        "prompt_sha256": sha256_text(SYSTEM_PROMPT + INSTRUCTION_PROMPT),
        "procedures": procedures,
        "traces": traces,
    }


def extract_slots(instructions: dict[str, Any], generator: TextGenerator) -> dict[str, Any]:
    document_id = str(instructions["document_id"])
    slots: list[dict[str, Any]] = []
    traces = []
    for procedure in instructions.get("procedures", []):
        for step in procedure.get("steps", []):
            prompt = SLOT_PROMPT.format(
                document_id=document_id,
                title=step.get("title") or "unknown",
                action=step["action"],
            )
            raw = generator.generate(SYSTEM_PROMPT, prompt)
            try:
                result = parse_json_response(raw)
                extracted = {
                    name: _nullable(result.get(name))
                    for name in ("action", "target", "tool", "location", "verification", "description", "energy_type", "step_type")
                }
                slots.append({
                    **step,
                    "predicted_action": extracted.pop("action"),
                    **extracted,
                })
                traces.append({"step_id": step["step_id"], "status": "ok", "raw_response": raw})
            except (ValueError, json.JSONDecodeError) as exc:
                traces.append({"step_id": step["step_id"], "status": "invalid_json_or_schema", "error": str(exc), "raw_response": raw})
    return {
        "document_id": document_id,
        "stage": "slot_extraction",
        "prompt_version": SLOT_PROMPT_VERSION,
        "prompt_sha256": sha256_text(SYSTEM_PROMPT + SLOT_PROMPT),
        "steps": slots,
        "traces": traces,
    }


def decompose_action(action: str, generator: TextGenerator) -> list[dict[str, str | None]]:
    raw = generator.generate(SYSTEM_PROMPT, ATOM_PROMPT.format(action=action))
    result = parse_json_response(raw)
    atoms = result.get("atoms")
    if not isinstance(atoms, list):
        raise ValueError("Atomic decomposition must contain an atoms list")
    output = []
    for atom in atoms:
        if not isinstance(atom, dict) or not _nullable(atom.get("verb")):
            raise ValueError("Each atom requires a verb")
        output.append({name: _nullable(atom.get(name)) for name in ("verb", "target", "tool", "location")})
    return output
