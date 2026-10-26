"""Transparent, unverified ontology-rule auditing and constrained LLM review.

The expert-authored original rule file and expert judgments are unavailable.
This module refuses to treat the provided starter rules as safety certification.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from loto_pke.llm import TextGenerator, parse_json_response


SAFETY_SYSTEM = (
    "You audit whether an extracted instruction representation is consistent "
    "with supplied research constraints. Do not infer absent steps, and do not "
    "declare a procedure safe. Return JSON only; use null when uncertain."
)


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _has_term(text: str, phrase: str) -> bool:
    normalized_text = _normalized(text)
    normalized_phrase = _normalized(phrase)
    # Boundary-aware phrase matching avoids 'heat' inside unrelated words.
    return re.search(r"(?<!\w)" + re.escape(normalized_phrase) + r"(?!\w)", normalized_text) is not None


def _categories(text: str, keywords: dict[str, list[str]]) -> list[str]:
    return sorted(
        category
        for category, terms in keywords.items()
        if any(_has_term(text, term) for term in terms)
    )


def deterministic_safety_audit(graph: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    """Surface *possible* inconsistencies for human review, not pass/fail."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in graph.get("steps", []):
        groups[str(step.get("procedure_id") or f"document-{graph['document_id']}")].append(step)

    results = []
    keyword_map = rules.get("action_categories", {})
    for procedure_id, steps in groups.items():
        all_text = " ".join(
            str(step.get(name) or "")
            for step in steps
            for name in ("description", "energy_type", "target", "action")
        )
        detected_energy = _categories(all_text, rules.get("energy_types", {}))
        mapped_steps = []
        by_category: dict[str, list[int]] = defaultdict(list)
        for index, step in enumerate(steps):
            action = str(step.get("action") or "")
            categories = _categories(action, keyword_map)
            for category in categories:
                by_category[category].append(index)
            mapped_steps.append({"step_id": step["step_id"], "categories": categories})

        flags: list[dict[str, Any]] = []
        for required in rules.get("required_categories", []):
            if required not in by_category:
                flags.append(
                    {
                        "criterion": "missing_action",
                        "severity": "possible",
                        "detail": f"No lexical evidence for required category {required!r}",
                        "step_ids": [],
                    }
                )

        for first, second in rules.get("ordered_pairs", []):
            if first in by_category and second in by_category and min(by_category[first]) > min(by_category[second]):
                flags.append(
                    {
                        "criterion": "action_order",
                        "severity": "possible",
                        "detail": f"Detected {second} before {first}",
                        "step_ids": [steps[min(by_category[second])]["step_id"], steps[min(by_category[first])]["step_id"]],
                    }
                )

        for check in rules.get("target_checks", []):
            energy = check.get("energy_type")
            if energy not in detected_energy:
                continue
            if check.get("incompatible_if_only_energy_type") and len(detected_energy) != 1:
                continue
            for step in steps:
                target = str(step.get("target") or "")
                for term in check.get("unexpected_target_terms", []):
                    if _has_term(target, term):
                        flags.append(
                            {
                                "criterion": "possible_energy_target_mismatch",
                                "severity": "possible",
                                "detail": f"Target contains {term!r} with inferred energy {energy!r}",
                                "step_ids": [step["step_id"]],
                            }
                        )

        missing_required = any(flag["criterion"] == "missing_action" for flag in flags)
        order_violations = any(flag["criterion"] == "action_order" for flag in flags)
        energy_target_mismatch = any(flag["criterion"] == "possible_energy_target_mismatch" for flag in flags)
        order_assessable = all(
            first in by_category and second in by_category
            for first, second in rules.get("ordered_pairs", [])
        )

        results.append(
            {
                "procedure_id": procedure_id,
                "detected_energy_types": detected_energy,
                "ordered_steps": [
                    {key: step.get(key) for key in (
                        "step_id", "action", "target", "tool", "location", "verification", "energy_type",
                    )}
                    for step in steps
                ],
                "mapped_steps": mapped_steps,
                "review_flags": flags,
                # Lexical research proxies, NOT claims of compliant operation.
                "energy_source_correct": not energy_target_mismatch if detected_energy else None,
                "action_order_correct": not order_violations if order_assessable else None,
                "missing_required_steps": missing_required,
                "judgment_provenance": "exploratory_keyword_heuristics",
                "requires_human_review": True,
            }
        )

    return {
        "document_id": graph["document_id"],
        "approach": "ontology_only_exploratory",
        "expert_verified_rules": bool(rules.get("expert_verified", False)),
        "safety_certification": "NOT_PERFORMED",
        "procedure_blocks": results,
    }


def llm_safety_audit(
    graph: dict[str, Any], rules: dict[str, Any], generator: TextGenerator,
) -> dict[str, Any]:
    """Prompt against fixed rules while retaining unedited model responses."""

    by_procedure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in graph.get("steps", []):
        by_procedure[str(step.get("procedure_id"))].append(step)

    blocks = []
    for procedure_id, steps in by_procedure.items():
        prompt = (
            "Evaluate this procedure block using ONLY the supplied rule JSON. "
            "The outcome is preliminary and must be checked by an expert. "
            "Return JSON with energy_source_correct, action_order_correct, "
            "missing_required_steps (each boolean or null), evidence_step_ids (list), "
            "and explanation (string).\n\n"
            "Rules:\n" + json.dumps(rules, ensure_ascii=False) +
            "\nProcedure:\n" + json.dumps(steps, ensure_ascii=False)
        )
        raw = generator.generate(SAFETY_SYSTEM, prompt)
        try:
            result = parse_json_response(raw)
            values = {}
            for key in ("energy_source_correct", "action_order_correct", "missing_required_steps"):
                value = result.get(key)
                if value is not None and not isinstance(value, bool):
                    raise ValueError(f"{key} must be true, false, or null")
                values[key] = value
            evidence = result.get("evidence_step_ids", [])
            if not isinstance(evidence, list) or any(not isinstance(item, str) for item in evidence):
                raise ValueError("evidence_step_ids must be a list of strings")
            blocks.append({
                "procedure_id": procedure_id,
                "status": "ok",
                **values,
                "ordered_steps": [
                    {key: step.get(key) for key in (
                        "step_id", "action", "target", "tool", "location", "verification", "energy_type",
                    )}
                    for step in steps
                ],
                "evidence_step_ids": evidence,
                "explanation": str(result.get("explanation") or ""),
                "judgment_provenance": "llm_not_certified",
                "raw_response": raw,
            })
        except (ValueError, json.JSONDecodeError) as exc:
            blocks.append({"procedure_id": procedure_id, "status": "invalid_response", "error": str(exc), "raw_response": raw})

    return {
        "document_id": graph["document_id"],
        "approach": "ontology_llm_exploratory",
        "expert_verified_rules": bool(rules.get("expert_verified", False)),
        "safety_certification": "NOT_PERFORMED",
        "procedure_blocks": blocks,
    }
