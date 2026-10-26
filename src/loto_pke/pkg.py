"""Typed, document-explicit procedural graph construction.

Relations are a versioned PKO-inspired working projection, not a claim that
the original expert ontology or its fine-grained reasoning was recovered.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from loto_pke.schema import PublicDocument


GRAPH_SCHEMA_VERSION = "pko-projection-1"
ROLE_RELATIONS = {
    "target": ("Target", "acts_on"),
    "tool": ("Tool", "uses"),
    "location": ("Location", "occurs_at"),
    "verification": ("Verification", "has_verification"),
}


def gold_steps(document: PublicDocument) -> list[dict[str, Any]]:
    """Convert structured TXT annotations to explicit reference slots."""

    return [
        step.to_dict()
        for procedure in document.procedures
        for step in procedure.steps
    ]


def build_graph(document_id: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Construct action/entity nodes, role edges, and per-group step order."""

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    triplets: list[dict[str, str]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ids: set[str] = set()

    for index, step in enumerate(steps, start=1):
        if str(step.get("document_id")) != str(document_id):
            raise ValueError("Cannot mix document IDs in one graph")
        step_id = str(step["step_id"])
        if step_id in ids:
            raise ValueError(f"Duplicate step ID: {step_id}")
        ids.add(step_id)
        procedure_id = str(step.get("procedure_id") or f"document-{document_id}")
        grouped[procedure_id].append(step)

        source_step_text = str(step.get("action") or "").strip()
        if not source_step_text:
            continue
        action_text = str(step.get("predicted_action") or source_step_text).strip()

        action_id = f"{step_id}:action"
        nodes.extend(
            [
                {"id": step_id, "type": "Step", "label": source_step_text, "procedure_id": procedure_id},
                {
                    "id": action_id,
                    "type": "Action",
                    "label": action_text,
                    "source_step_id": step_id,
                    "step_type": step.get("step_type"),
                    "energy_type": step.get("energy_type"),
                    "description": step.get("description"),
                },
            ]
        )
        edges.append({"source": step_id, "relation": "has_action", "target": action_id})

        for role, (node_type, relation) in ROLE_RELATIONS.items():
            text = step.get(role)
            if not isinstance(text, str) or not text.strip():
                continue
            role_id = f"{step_id}:{role}"
            label = text.strip()
            nodes.append({"id": role_id, "type": node_type, "label": label, "source_step_id": step_id})
            edges.append({"source": action_id, "relation": relation, "target": role_id})
            triplets.append({"subject": action_text, "relation": relation, "object": label})

        # Optional second-stage atomic decomposition. The unmodified source
        # instruction remains the parent Action for provenance.
        atoms = step.get("atoms")
        if isinstance(atoms, list):
            previous_atom: tuple[str, str] | None = None
            for atom_index, atom in enumerate(atoms, start=1):
                if not isinstance(atom, dict) or not isinstance(atom.get("verb"), str):
                    continue
                atom_label = atom["verb"].strip()
                if not atom_label:
                    continue
                atom_id = f"{step_id}:atom{atom_index}"
                nodes.append({"id": atom_id, "type": "AtomicAction", "label": atom_label, "source_step_id": step_id})
                edges.append({"source": action_id, "relation": "decomposes_to", "target": atom_id})
                for role, (node_type, relation) in ROLE_RELATIONS.items():
                    if role == "verification":
                        continue
                    argument = atom.get(role)
                    if not isinstance(argument, str) or not argument.strip():
                        continue
                    argument_id = f"{atom_id}:{role}"
                    argument_text = argument.strip()
                    nodes.append({"id": argument_id, "type": node_type, "label": argument_text, "source_step_id": step_id})
                    edges.append({"source": atom_id, "relation": relation, "target": argument_id})
                    triplets.append({"subject": atom_label, "relation": relation, "object": argument_text})
                if previous_atom:
                    edges.append({"source": previous_atom[0], "relation": "precedes", "target": atom_id})
                    triplets.append({"subject": previous_atom[1], "relation": "precedes", "object": atom_label})
                previous_atom = (atom_id, atom_label)

    for procedure_id, procedure_steps in grouped.items():
        # Input order is the annotation/extraction reading order. We must not
        # sort globally by step_order because numbering restarts at boundaries.
        nonempty = [step for step in procedure_steps if step.get("action")]
        for predecessor, successor in zip(nonempty, nonempty[1:]):
            edges.append({"source": predecessor["step_id"], "relation": "precedes", "target": successor["step_id"]})
            triplets.append(
                {
                    "subject": str(predecessor.get("predicted_action") or predecessor["action"]).strip(),
                    "relation": "precedes",
                    "object": str(successor.get("predicted_action") or successor["action"]).strip(),
                }
            )

        # PKO-inspired transitive precedence, separately identified from
        # the immediate precedence edges for ablation and fair comparison.
        for start_index, predecessor in enumerate(nonempty):
            for successor in nonempty[start_index + 2:]:
                edges.append({
                    "source": predecessor["step_id"],
                    "relation": "precedes_transitive",
                    "target": successor["step_id"],
                })

    return {
        "document_id": str(document_id),
        "schema_version": GRAPH_SCHEMA_VERSION,
        "nodes": nodes,
        "edges": edges,
        "triplets": triplets,
        "steps": steps,
        "grouping": "procedure_id_input_order",
        "transitive_precedence": "derived_edges_separate_from_explicit_triplets",
        "visual_evidence_status": "not_annotated",
    }


def graph_from_document(document: PublicDocument) -> dict[str, Any]:
    return build_graph(document.document_id, gold_steps(document))
