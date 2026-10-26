"""One-pass VLM graph baselines and optional second-pass visual linking.

Two output modes distinguish a direct graph-generation experiment from a
controlled single-pass slot prediction followed by the shared graph mapper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loto_pke.common import sha256_text
from loto_pke.llm import TextGenerator, parse_json_response
from loto_pke.pkg import build_graph


BASELINE_VERSION = "end-to-end-vlm-v1"
SYSTEM = (
    "You extract only document-explicit Lockout-Tagout knowledge from ordered "
    "page images. Preserve original identifiers and step order. No safety "
    "procedure may be completed or invented from background knowledge. "
    "Unknown fields must be null. Return one valid JSON object."
)
STRUCTURED_PROMPT = """Given {page_count} page image(s) in reading order, produce the
entire structured procedural representation in one pass. Return JSON:
{{"procedures":[{{"title":null,"steps":[{{"step_order":1,"action":"source text",
"target":null,"tool":null,"location":null,"verification":null,
"description":null,"energy_type":null,"step_type":"executable",
"source_page":1}}]}}]}}
step_type is executable, conditional, or other. Do not add steps absent from
the images. Use source_page 1-based, and null for unknown fields.
"""
DIRECT_PROMPT = """Given {page_count} page image(s) in reading order, produce the
entire procedural knowledge graph in ONE pass. Return JSON with exactly:
{{"nodes":[{{"id":"n1","type":"Step","label":"source text"}},
{{"id":"n2","type":"Action","label":"source text"}}],
"edges":[{{"source":"n1","relation":"has_action","target":"n2"}}]}}
Allowed node types: Step, Action, AtomicAction, Target, Tool, Location,
Verification, Image. Allowed relations: has_action, acts_on, uses, occurs_at,
has_verification, precedes, decomposes_to, has_visual_evidence.
Use unique node IDs. Do not invent implicit actions or objects. A graph may
be empty when there are no readable actions. No commentary outside JSON.
"""
VISUAL_LINK_PROMPT = """For the following extracted graph and ordered page images,
identify only visually supported action/entity references. Return JSON:
{{"links":[{{"node_id":"n1","page_number":1,"bbox":null}}]}}
Each link refers to an existing node and 1-based page number. A bbox, if
unambiguously identifiable, uses [x0,y0,x1,y1] normalized to [0,1] in image
coordinates. Otherwise use null. Do not link unsupported nodes.
GRAPH:
{graph_json}
"""


def _nullable(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected string/null, got {type(value).__name__}")
    return value.strip() or None


def normalize_structured_baseline(
    answer: dict[str, Any], document_id: str, page_count: int,
) -> dict[str, Any]:
    """Validate a one-pass VLM response and construct the common PKG schema."""

    procedures = answer.get("procedures")
    if not isinstance(procedures, list):
        raise ValueError("One-pass output must contain a procedures array")
    steps: list[dict[str, Any]] = []
    for p_index, procedure in enumerate(procedures, start=1):
        if not isinstance(procedure, dict) or not isinstance(procedure.get("steps"), list):
            raise ValueError("Each procedure must contain steps")
        title = _nullable(procedure.get("title"))
        procedure_id = f"public-{int(document_id):03d}-vlm-p{p_index:02d}"
        for s_index, step in enumerate(procedure["steps"], start=1):
            if not isinstance(step, dict) or not _nullable(step.get("action")):
                raise ValueError("Every predicted step requires an action")
            order = step.get("step_order")
            page = step.get("source_page")
            if order is not None and (type(order) is not int or order < 1):
                raise ValueError("Invalid step_order")
            if page is not None and (type(page) is not int or not 1 <= page <= page_count):
                raise ValueError("source_page outside the input page range")
            steps.append({
                "document_id": document_id,
                "procedure_id": procedure_id,
                "step_id": f"{procedure_id}-s{s_index:03d}",
                "step_order": order,
                "source_page": page,
                "title": title,
                **{
                    field: _nullable(step.get(field))
                    for field in (
                        "action", "target", "tool", "location", "verification",
                        "description", "energy_type", "step_type",
                    )
                },
            })
    result = build_graph(document_id, steps)
    result["output_mode"] = "structured_steps_shared_graph_projection"
    return result


def normalize_direct_graph(answer: dict[str, Any], document_id: str) -> dict[str, Any]:
    """Validate graph connectivity; keep exact model-generated node labels."""

    nodes, edges = answer.get("nodes"), answer.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Direct graph requires nodes and edges arrays")
    allowed_types = {
        "Step", "Action", "AtomicAction", "Target", "Tool", "Location", "Verification", "Image",
    }
    allowed_relations = {
        "has_action", "acts_on", "uses", "occurs_at", "has_verification", "precedes",
        "decomposes_to", "has_visual_evidence",
    }
    normalized_nodes: list[dict[str, str]] = []
    by_id: dict[str, dict[str, str]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("Invalid node record")
        node_id, label, kind = node.get("id"), node.get("label"), node.get("type")
        if not isinstance(node_id, str) or not node_id.strip() or node_id in by_id:
            raise ValueError(f"Missing/duplicate node ID: {node_id!r}")
        if not isinstance(label, str) or not label.strip() or kind not in allowed_types:
            raise ValueError(f"Invalid node label/type: {node!r}")
        normalized = {"id": node_id, "type": kind, "label": label.strip()}
        normalized_nodes.append(normalized)
        by_id[node_id] = normalized

    normalized_edges: list[dict[str, str]] = []
    triplets = []
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("Invalid edge record")
        source, target, relation = edge.get("source"), edge.get("target"), edge.get("relation")
        if source not in by_id or target not in by_id or relation not in allowed_relations:
            raise ValueError(f"Unknown edge endpoint/relation: {edge!r}")
        normalized_edges.append({"source": source, "relation": relation, "target": target})
        if relation in {"acts_on", "uses", "occurs_at", "has_verification", "precedes", "has_visual_evidence"}:
            triplets.append({
                "subject": by_id[source]["label"],
                "relation": relation,
                "object": by_id[target]["label"],
            })

    return {
        "document_id": document_id,
        "schema_version": "direct_vlm-1",
        "nodes": normalized_nodes,
        "edges": normalized_edges,
        "triplets": triplets,
        "steps": [],
        "output_mode": "direct_graph_from_images",
    }


def add_visual_links(
    graph: dict[str, Any], answer: dict[str, Any], page_count: int,
) -> dict[str, Any]:
    """Attach predicted image_of-like edges, without treating them as gold."""

    links = answer.get("links")
    if not isinstance(links, list):
        raise ValueError("Visual linking requires a links array")
    by_id = {node["id"]: node for node in graph["nodes"]}
    output = {**graph, "nodes": list(graph["nodes"]), "edges": list(graph["edges"]),
              "triplets": list(graph["triplets"])}
    seen = set()
    for entry in links:
        if not isinstance(entry, dict) or entry.get("node_id") not in by_id:
            raise ValueError("Visual link references a missing graph node")
        page = entry.get("page_number")
        if type(page) is not int or not 1 <= page <= page_count:
            raise ValueError("Visual link has an invalid page number")
        source = entry["node_id"]
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        image_id = f"visual:page{page}"
        if image_id not in by_id:
            image_node = {"id": image_id, "type": "Image", "label": f"page:{page}"}
            by_id[image_id] = image_node
            output["nodes"].append(image_node)
        bbox = entry.get("bbox")
        if bbox is not None and (
            not isinstance(bbox, list) or len(bbox) != 4
            or any(type(value) not in (int, float) or not 0 <= value <= 1 for value in bbox)
            or bbox[0] > bbox[2] or bbox[1] > bbox[3]
        ):
            raise ValueError("Bbox requires normalized ordered coordinates")
        output["edges"].append({
            "source": source, "target": image_id, "relation": "has_visual_evidence",
            "bbox": bbox, "bbox_coordinate_system": "normalized_image_pixels" if bbox is not None else None,
        })
        output["triplets"].append({
            "subject": by_id[source]["label"], "relation": "has_visual_evidence",
            "object": f"page:{page}",
        })
    output["visual_evidence_status"] = "vlm_predicted_not_gold"
    return output


def run_vlm_baseline(
    document_id: str, image_paths: list[Path], generator: TextGenerator,
    mode: str = "direct_graph", visual_links: bool = False,
) -> dict[str, Any]:
    if mode not in {"direct_graph", "structured_steps"}:
        raise ValueError(f"Unsupported baseline mode: {mode}")
    if not image_paths:
        raise ValueError("Cannot run a visual baseline without page images")

    prompt = (DIRECT_PROMPT if mode == "direct_graph" else STRUCTURED_PROMPT).format(
        page_count=len(image_paths),
    )
    raw = generator.generate(SYSTEM, prompt, images=image_paths)
    result: dict[str, Any] = {
        "document_id": document_id,
        "stage": "end_to_end_vlm",
        "baseline_version": BASELINE_VERSION,
        "mode": mode,
        "prompt_sha256": sha256_text(SYSTEM + prompt),
        "input_page_count": len(image_paths),
        "raw_response": raw,
    }
    try:
        answer = parse_json_response(raw)
        graph = (
            normalize_direct_graph(answer, document_id)
            if mode == "direct_graph"
            else normalize_structured_baseline(answer, document_id, len(image_paths))
        )
        result["status"] = "ok"
        result["graph"] = graph
    except (json.JSONDecodeError, ValueError) as exc:
        result["status"] = "invalid_response"
        result["error"] = str(exc)
        return result

    if visual_links:
        link_prompt = VISUAL_LINK_PROMPT.format(
            graph_json=json.dumps(graph, ensure_ascii=False),
        )
        raw_links = generator.generate(SYSTEM, link_prompt, images=image_paths)
        result["raw_visual_link_response"] = raw_links
        try:
            result["graph"] = add_visual_links(graph, parse_json_response(raw_links), len(image_paths))
            result["visual_link_status"] = "ok"
        except (json.JSONDecodeError, ValueError) as exc:
            result["visual_link_status"] = "invalid_response"
            result["visual_link_error"] = str(exc)
    else:
        result["visual_link_status"] = "not_requested"
    return result
