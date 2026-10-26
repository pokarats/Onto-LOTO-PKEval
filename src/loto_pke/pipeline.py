"""Public-only, resumable command-line experiments for LOTO PKE research.

Every model stage emits its raw trace and per-document status. Existing output
is never replaced unless the user explicitly supplies --overwrite or --resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from loto_pke.annotation_agreement import inter_annotator_agreement
from loto_pke.audit import summarize, write_audit
from loto_pke.baselines import run_vlm_baseline
from loto_pke.common import environment_info, read_json, read_jsonl, write_json, write_jsonl
from loto_pke.comparison import compare_pipelines
from loto_pke.evaluation import (
    evaluate_expert_agreement,
    evaluate_graphs,
    evaluate_instructions,
    evaluate_parsing,
    evaluate_slots,
    expert_review_template,
    extract_steps,
    layout_review_template,
)
from loto_pke.extraction import decompose_action, extract_instructions, extract_slots
from loto_pke.human_review import human_review_template, summarize_human_review
from loto_pke.llm import OllamaGenerator
from loto_pke.metrics import TextSimilarity
from loto_pke.parsing import parse_document, source_images
from loto_pke.pkg import build_graph, graph_from_document, gold_steps
from loto_pke.public_dataset import load_public_dataset, read_selection_file
from loto_pke.safety import deterministic_safety_audit, llm_safety_audit
from loto_pke.smart_optional import evaluate_smart_parsing
from loto_pke.vision import (
    ClipScorer,
    HFVisionGenerator,
    PaliGemmaScorer,
    build_image_candidates,
    grounding_accuracy,
    grounding_label_template,
    rank_visual_evidence,
)


RUNNER_VERSION = "public-loto-pke-0.2.0"


def _source_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if root.name != "public_data" and (root / "public_data").is_dir():
        root = root / "public_data"
    return root


def _selected_dataset(args: argparse.Namespace) -> tuple[Any, list[Any]]:
    selection = read_selection_file(args.selection_file) if args.selection_file else None
    dataset = load_public_dataset(args.data_root, selected_ids=selection)
    critical = [item for item in dataset.issues if item.severity == "error" and (
        item.document_id is None or any(doc.document_id == item.document_id and doc.selected for doc in dataset.documents)
    )]
    if critical:
        raise ValueError(f"Public-data validation found {len(critical)} errors; run audit first")
    documents = [document for document in dataset.documents if document.selected]
    if args.limit is not None:
        documents = documents[:args.limit]
    return dataset, documents


def _ollama(args: argparse.Namespace) -> OllamaGenerator:
    return OllamaGenerator(
        model=args.model,
        endpoint=args.ollama_endpoint,
        seed=args.seed,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        timeout_seconds=args.timeout,
    )


def _stage_status(record: dict[str, Any], tag: str) -> dict[str, Any]:
    result = dict(record)
    if "status" not in result:
        traces = result.get("traces", [])
        if traces and any(item.get("status") == "invalid_json_or_schema" for item in traces):
            result["status"] = "partial_invalid_outputs"
        elif tag == "parsed" and not result.get("text", "").strip():
            result["status"] = "no_extractable_text"
        elif tag == "instructions" and not result.get("procedures"):
            result["status"] = "empty_extraction"
        elif tag == "slots" and not result.get("steps"):
            result["status"] = "empty_extraction"
        else:
            result["status"] = "ok"
    return result


def _check_output_policy(path: Path, overwrite: bool, resume: bool) -> None:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume cannot be used together")
    if path.exists() and not (overwrite or resume):
        raise FileExistsError(f"Output exists: {path}; use --resume or --overwrite explicitly")


def _safe_invoke(document_id: str, process: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        record = process()
        if str(record.get("document_id")) != document_id:
            raise ValueError("Stage output document ID does not match its input")
        return record
    except Exception as exc:
        return {
            "document_id": document_id,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _run_items(
    args: argparse.Namespace,
    items: Iterable[Any],
    get_id: Callable[[Any], str],
    process: Callable[[Any], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Atomic per-record checkpoints allow a Slurm job to resume safely."""

    output: Path = args.output
    _check_output_policy(output, args.overwrite, args.resume)
    item_list = list(items)
    keys = [get_id(item) for item in item_list]
    if len(set(keys)) != len(keys):
        raise ValueError("Stage inputs must have unique document IDs")

    old = read_jsonl(output) if args.resume and output.is_file() else []
    old_index = {str(record["document_id"]): record for record in old}
    if len(old_index) != len(old):
        raise ValueError("Resumed output contains duplicate document IDs")
    if set(old_index) - set(keys):
        raise ValueError("Resumed output contains documents absent from the current selection")

    completed = dict(old_index)
    changed = False
    for index, item in enumerate(item_list, start=1):
        key = get_id(item)
        if key in completed and not (args.retry_errors and completed[key].get("status") == "error"):
            print(f"[{index}/{len(item_list)}] {key}: resumed", flush=True)
            continue
        result = _safe_invoke(key, lambda: process(item))
        completed[key] = result
        changed = True
        # Write only records in stable input order. Safe if killed between docs.
        write_jsonl(output, (completed[current] for current in keys if current in completed), overwrite=True)
        print(f"[{index}/{len(item_list)}] {key}: {result.get('status', 'ok')}", flush=True)

    if not output.exists() or not item_list:
        write_jsonl(output, (completed[key] for key in keys if key in completed), overwrite=True)
    if changed or not args.resume:
        _metadata(args, output, len(completed),
                  sum(value.get("status") == "error" for value in completed.values()))
    failed = [key for key in keys if key in completed and completed[key].get("status") == "error"]
    if failed:
        raise RuntimeError(
            f"{len(failed)} document(s) failed: {', '.join(failed)}. "
            f"Partial checkpoint saved at {output}; rerun with --resume --retry-errors."
        )
    return [completed[key] for key in keys if key in completed]


def _metadata(args: argparse.Namespace, output: Path, count: int, errors: int) -> None:
    source_hashes = {}
    for key in ("input", "gold", "rules", "selection_file", "normalization", "first", "second", "multistage", "baseline"):
        path = getattr(args, key, None)
        if isinstance(path, Path) and path.is_file():
            checksum = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(chunk)
            source_hashes[key] = checksum.hexdigest()
    metadata = {
        "runner_version": RUNNER_VERSION,
        "command": args.command,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": count,
        "error_record_count": errors,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"func", "ollama_endpoint"}
        },
        "environment": environment_info(),
        "input_file_sha256": source_hashes,
        "scientific_status": "independent_public_data_only_reimplementation",
    }
    write_json(Path(str(output) + ".meta.json"), metadata, overwrite=True)


def _write_report(args: argparse.Namespace, report: dict[str, Any]) -> None:
    _check_output_policy(args.output, args.overwrite, args.resume)
    write_json(args.output, report, overwrite=args.overwrite or args.resume)
    _metadata(args, args.output, 1, 0)
    print(f"Saved {args.output}")


def _read_stage(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def cmd_audit(args: argparse.Namespace) -> None:
    dataset = load_public_dataset(
        args.data_root,
        selected_ids=read_selection_file(args.selection_file) if args.selection_file else None,
    )
    if args.output_dir.exists() and not args.overwrite and any(args.output_dir.iterdir()):
        raise FileExistsError("Audit directory is nonempty; use --overwrite after reviewing its contents")
    summary = write_audit(dataset, args.output_dir)
    print(json.dumps({key: summary[key] for key in (
        "document_count", "selected_document_count", "inferred_procedure_group_count",
        "annotated_step_record_count", "issue_severity_counts",
    )}, indent=2))


def cmd_gold_graphs(args: argparse.Namespace) -> None:
    _, documents = _selected_dataset(args)
    _run_items(args, documents, lambda doc: doc.document_id, lambda doc: {
        **graph_from_document(doc), "status": "ok", "source": "public_structured_txt_annotation",
    })


def cmd_parse(args: argparse.Namespace) -> None:
    _, documents = _selected_dataset(args)
    root = _source_root(args.data_root)
    _run_items(args, documents, lambda doc: doc.document_id, lambda doc: _stage_status({
        **parse_document(doc, root, args.backend, args.ocr_language),
        "source_sha256": doc.source.sha256 if doc.source else None,
    }, "parsed"))


def cmd_instructions(args: argparse.Namespace) -> None:
    records = _read_stage(args.input)
    generator = _ollama(args)

    def process(record: dict[str, Any]) -> dict[str, Any]:
        if record.get("status") == "error":
            raise ValueError("Upstream document parsing failed")
        return _stage_status(extract_instructions(record, generator), "instructions")

    _run_items(args, records, lambda record: str(record["document_id"]), process)


def cmd_slots(args: argparse.Namespace) -> None:
    records = _read_stage(args.input)
    generator = _ollama(args)

    def process(record: dict[str, Any]) -> dict[str, Any]:
        if record.get("status") == "error":
            raise ValueError("Upstream instruction extraction failed")
        return _stage_status(extract_slots(record, generator), "slots")

    _run_items(args, records, lambda record: str(record["document_id"]), process)


def cmd_graphs(args: argparse.Namespace) -> None:
    records = _read_stage(args.input)
    atom_generator = _ollama(args) if args.atomize else None

    def process(record: dict[str, Any]) -> dict[str, Any]:
        if record.get("status") == "error":
            raise ValueError("Upstream model extraction failed")
        steps = [dict(step) for step in extract_steps(record)]
        traces = []
        if atom_generator:
            for step in steps:
                try:
                    step["atoms"] = decompose_action(str(step["action"]), atom_generator)
                    traces.append({"step_id": step["step_id"], "status": "ok"})
                except (ValueError, json.JSONDecodeError) as exc:
                    traces.append({"step_id": step["step_id"], "status": "invalid_response", "error": str(exc)})
        graph = build_graph(str(record["document_id"]), steps)
        graph["status"] = "ok" if steps else "empty_extraction"
        graph["source_stage"] = record.get("stage", "unknown")
        graph["atomization_traces"] = traces
        return graph

    _run_items(args, records, lambda record: str(record["document_id"]), process)


def cmd_safety(args: argparse.Namespace) -> None:
    records = _read_stage(args.input)
    rules = read_json(args.rules)
    if rules.get("expert_verified") is True:
        print("WARNING: external expert_verified flag was supplied; verify provenance independently", file=sys.stderr)
    generator = _ollama(args) if args.mode == "ontology_llm" else None

    def process(record: dict[str, Any]) -> dict[str, Any]:
        graph = record.get("graph", record)
        if graph.get("status") == "error":
            raise ValueError("Upstream graph construction failed")
        if not graph.get("steps"):
            return {
                "document_id": str(graph["document_id"]),
                "status": "not_evaluable",
                "reason": "No ordered steps are available for a procedure-level safety audit",
                "safety_certification": "NOT_PERFORMED",
                "procedure_blocks": [],
            }
        return (llm_safety_audit(graph, rules, generator) if generator is not None
                else deterministic_safety_audit(graph, rules))

    _run_items(args, records, lambda record: str(record["document_id"]), process)


def cmd_baseline(args: argparse.Namespace) -> None:
    _, documents = _selected_dataset(args)
    root = _source_root(args.data_root)
    if args.generator == "hf":
        generator = HFVisionGenerator(
            args.model, offline=not args.allow_download,
            max_new_tokens=args.max_new_tokens, seed=args.seed, dtype=args.dtype,
        )
    else:
        generator = _ollama(args)

    def process(doc: Any) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="loto_vlm_pages_") as directory:
            images = source_images(doc, root, tmpdir=Path(directory))
            result = run_vlm_baseline(doc.document_id, images, generator, args.mode, args.visual_links)
        result["model"] = args.model
        result["generator"] = args.generator
        return result

    _run_items(args, documents, lambda doc: doc.document_id, process)


def _gold_step_queries(documents: list[Any], field: str) -> list[dict[str, str]]:
    return [
        {"step_id": step["step_id"], "document_id": doc.document_id,
         "text": str(step.get(field) or "")}
        for doc in documents
        for step in gold_steps(doc)
        if step.get(field)
    ]


def _queries(args: argparse.Namespace, documents: list[Any]) -> list[dict[str, str]]:
    if args.query_input:
        records = _read_stage(args.query_input)
        allowed = {doc.document_id for doc in documents}
        return [
            {"step_id": str(step["step_id"]), "document_id": str(record["document_id"]),
             "text": str(step.get(args.query_field) or "")}
            for record in records if str(record["document_id"]) in allowed
            for step in extract_steps(record)
            if step.get(args.query_field)
        ]
    return _gold_step_queries(documents, args.query_field)


def cmd_ground(args: argparse.Namespace) -> None:
    if args.resume:
        raise ValueError("Use --overwrite, not --resume, for the grounding command")
    _check_output_policy(args.output, args.overwrite, args.resume)
    dataset, documents = _selected_dataset(args)
    root = _source_root(args.data_root)
    queries = _queries(args, documents)
    candidates = build_image_candidates(dataset, root)
    if args.method == "clip":
        scorer = ClipScorer(args.model or "openai/clip-vit-large-patch14-336", offline=not args.allow_download)
    else:
        scorer = PaliGemmaScorer(args.model or "google/paligemma2-3b-mix-448", offline=not args.allow_download)
    # Image retrieval is query-level, not a document record; --resume is
    # deliberately unsupported until per-query checkpointing is implemented.
    ranked = rank_visual_evidence(queries, candidates, scorer, args.top_k)
    write_jsonl(args.output, ranked, overwrite=args.overwrite)
    labels = read_jsonl(args.gold_labels) if args.gold_labels else None
    evaluation = grounding_accuracy(ranked, labels, tuple(range(1, args.top_k + 1)))
    evaluation.update({
        "query_count": len(queries),
        "candidate_count": len(candidates),
        "scorer": scorer.method,
        "retrieval_level": "page_image_not_annotated_subimage_region",
        "document_proxy_hit_at_1": sum(row["document_proxy_hit_at_1"] for row in ranked) / len(ranked) if ranked else None,
        "document_proxy_hit_at_2": sum(row["document_proxy_hit_at_2"] for row in ranked) / len(ranked) if ranked else None,
    })
    write_json(Path(str(args.output) + ".evaluation.json"), evaluation, overwrite=args.overwrite)
    _metadata(args, args.output, len(ranked), 0)
    print(f"Saved {len(ranked)} grounding rankings to {args.output}")


def cmd_ground_template(args: argparse.Namespace) -> None:
    dataset, documents = _selected_dataset(args)
    queries = _queries(args, documents)
    candidates = build_image_candidates(dataset, _source_root(args.data_root))
    _check_output_policy(args.output, args.overwrite, args.resume)
    write_jsonl(args.output, grounding_label_template(queries, candidates), overwrite=args.overwrite or args.resume)
    print(f"Saved {len(queries)} unlabeled grounding tasks to {args.output}")


def cmd_evaluate_graphs(args: argparse.Namespace) -> None:
    report = evaluate_graphs(
        _read_stage(args.input), _read_stage(args.gold),
        methods=args.methods,
        thresholds={"fuzzy": args.fuzzy_threshold, "embedding": args.embedding_threshold},
        granularity=args.granularity,
        ged=args.ged,
        ged_timeout=args.ged_timeout,
        embedding_model=args.embedding_model,
        offline=not args.allow_download,
    )
    _write_report(args, report)


def cmd_evaluate_instructions(args: argparse.Namespace) -> None:
    similarity = TextSimilarity(args.method, model_name=args.embedding_model, offline=not args.allow_download)
    _write_report(args, evaluate_instructions(_read_stage(args.input), _read_stage(args.gold), similarity))


def cmd_evaluate_slots(args: argparse.Namespace) -> None:
    similarity = TextSimilarity(args.method, model_name=args.embedding_model, offline=not args.allow_download)
    _write_report(args, evaluate_slots(_read_stage(args.input), _read_stage(args.gold), similarity, args.threshold))


def cmd_evaluate_parsing(args: argparse.Namespace) -> None:
    reference = _read_stage(args.gold) if args.gold else None
    _write_report(args, evaluate_parsing(_read_stage(args.input), reference))


def cmd_evaluate_smart(args: argparse.Namespace) -> None:
    _write_report(args, evaluate_smart_parsing(
        _read_stage(args.input), _read_stage(args.gold),
        matcher=args.matcher,
        bleurt_checkpoint=str(args.bleurt_checkpoint) if args.bleurt_checkpoint else None,
    ))


def cmd_layout_template(args: argparse.Namespace) -> None:
    _check_output_policy(args.output, args.overwrite, args.resume)
    tasks = layout_review_template(_read_stage(args.input))
    write_jsonl(args.output, tasks, overwrite=args.overwrite or args.resume)
    print(f"Saved {len(tasks)} pending layout annotation tasks")


def cmd_expert_template(args: argparse.Namespace) -> None:
    _check_output_policy(args.output, args.overwrite, args.resume)
    tasks = expert_review_template(_read_stage(args.input))
    write_jsonl(args.output, tasks, overwrite=args.overwrite or args.resume)
    print(f"Saved {len(tasks)} pending expert review tasks")


def cmd_expert_agreement(args: argparse.Namespace) -> None:
    gold = _read_stage(args.gold) if args.gold else None
    _write_report(args, evaluate_expert_agreement(_read_stage(args.input), gold))


def cmd_inter_annotator_agreement(args: argparse.Namespace) -> None:
    normalizer = read_json(args.normalization) if args.normalization else None
    _write_report(args, inter_annotator_agreement(
        _read_stage(args.first), _read_stage(args.second), normalizer,
    ))


def cmd_human_template(args: argparse.Namespace) -> None:
    _check_output_policy(args.output, args.overwrite, args.resume)
    tasks = human_review_template(_read_stage(args.input))
    write_jsonl(args.output, tasks, overwrite=args.overwrite or args.resume)
    print(f"Saved {len(tasks)} pending instruction/slot review tasks")


def cmd_human_report(args: argparse.Namespace) -> None:
    _write_report(args, summarize_human_review(_read_stage(args.input)))


def cmd_compare_pipelines(args: argparse.Namespace) -> None:
    _write_report(args, compare_pipelines(
        _read_stage(args.multistage), _read_stage(args.baseline), _read_stage(args.gold),
        method=args.method, threshold=args.threshold,
        bootstrap_samples=args.bootstrap_samples, seed=args.seed,
        embedding_model=args.embedding_model, offline=not args.allow_download,
    ))


def _add_output(parser: argparse.ArgumentParser, *, streaming: bool = False) -> None:
    parser.add_argument("--output", required=True, type=Path, help="JSONL for stages; JSON for reports")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace prior output")
    parser.add_argument("--resume", action="store_true", help="Resume a previous stage/checkpoint")
    if streaming:
        parser.add_argument("--retry-errors", action="store_true", help="On resume, retry failed records")


def _add_data(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", required=True, type=Path, help="Path to public_data/ or its parent")
    parser.add_argument("--selection-file", type=Path, help="Newline-separated public document IDs")
    parser.add_argument("--limit", type=int, help="First N selected documents for a smoke run")


def _add_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="qwen3:14b", help="Locally installed Ollama model identifier")
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434", help="Loopback service URL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=600, help="Request timeout in seconds")


def _add_similarity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=("exact", "fuzzy", "embedding"), default="fuzzy")
    parser.add_argument("--embedding-model", default="google/embeddinggemma-300m")
    parser.add_argument("--allow-download", action="store_true", help="Permit HF weights download; offline otherwise")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loto-pke",
        description="Reproducible public-only LOTO procedural knowledge extraction experiments",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Inventory inputs and audit TXT annotations")
    audit.add_argument("--data-root", required=True, type=Path)
    audit.add_argument("--output-dir", required=True, type=Path)
    audit.add_argument("--selection-file", type=Path)
    audit.add_argument("--overwrite", action="store_true")
    audit.set_defaults(func=cmd_audit)

    gold_graphs = sub.add_parser("gold-graphs", help="Construct gold PKGs from the public TXT slots")
    _add_data(gold_graphs)
    _add_output(gold_graphs, streaming=True)
    gold_graphs.set_defaults(func=cmd_gold_graphs)

    parse = sub.add_parser("parse", help="Parse public image/PDF pages")
    _add_data(parse)
    _add_output(parse, streaming=True)
    parse.add_argument("--backend", required=True, choices=("tesseract", "pymupdf", "docling"))
    parse.add_argument("--ocr-language", default="eng")
    parse.set_defaults(func=cmd_parse)

    instructions = sub.add_parser("instructions", help="Few-shot instruction extraction via local Ollama")
    instructions.add_argument("--input", required=True, type=Path)
    _add_output(instructions, streaming=True)
    _add_model(instructions)
    instructions.set_defaults(func=cmd_instructions)

    slots = sub.add_parser("slots", help="One instruction -> action/target/tool/location/etc.")
    slots.add_argument("--input", required=True, type=Path)
    _add_output(slots, streaming=True)
    _add_model(slots)
    slots.set_defaults(func=cmd_slots)

    graphs = sub.add_parser("graphs", help="Build typed PKGs; optionally atomize actions with an LLM")
    graphs.add_argument("--input", required=True, type=Path)
    _add_output(graphs, streaming=True)
    _add_model(graphs)
    graphs.add_argument("--atomize", action="store_true", help="Second-prompt atomic-action decomposition")
    graphs.set_defaults(func=cmd_graphs)

    safety = sub.add_parser("safety", help="Exploratory rule-only or ontology-LLM review")
    safety.add_argument("--input", required=True, type=Path)
    _add_output(safety, streaming=True)
    safety.add_argument("--rules", required=True, type=Path)
    safety.add_argument("--mode", choices=("ontology_only", "ontology_llm"), required=True)
    _add_model(safety)
    safety.set_defaults(func=cmd_safety)

    baseline = sub.add_parser("baseline", help="Qwen/Gemma image-to-PKG end-to-end baseline")
    _add_data(baseline)
    _add_output(baseline, streaming=True)
    _add_model(baseline)
    baseline.add_argument("--generator", choices=("hf", "ollama"), default="hf")
    baseline.add_argument("--mode", choices=("direct_graph", "structured_steps"), default="direct_graph")
    baseline.add_argument("--visual-links", action="store_true")
    baseline.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    baseline.add_argument("--allow-download", action="store_true")
    baseline.set_defaults(func=cmd_baseline, model="Qwen/Qwen3.5-4B")

    ground = sub.add_parser("ground", help="Rank page-image evidence using CLIP/PaliGemma")
    _add_data(ground)
    _add_output(ground)
    ground.add_argument("--method", choices=("clip", "paligemma"), required=True)
    ground.add_argument("--model", help="Local Hugging Face checkpoint")
    ground.add_argument("--query-input", type=Path, help="Optional model output; otherwise gold TXT slots are queries")
    ground.add_argument("--query-field", choices=("action", "target", "tool", "verification"), default="action")
    ground.add_argument("--gold-labels", type=Path, help="Verified image IDs for true Acc@k; otherwise only proxy")
    ground.add_argument("--top-k", type=int, default=2)
    ground.add_argument("--allow-download", action="store_true")
    ground.set_defaults(func=cmd_ground)

    ground_template = sub.add_parser("grounding-template", help="Blank fine-grounding annotation tasks")
    _add_data(ground_template)
    _add_output(ground_template)
    ground_template.add_argument("--query-input", type=Path)
    ground_template.add_argument("--query-field", choices=("action", "target", "tool", "verification"), default="action")
    ground_template.set_defaults(func=cmd_ground_template)

    eval_graphs = sub.add_parser("evaluate-graphs", help="Triplets, node types, Jaccard, optional GED")
    eval_graphs.add_argument("--input", required=True, type=Path)
    eval_graphs.add_argument("--gold", required=True, type=Path)
    _add_output(eval_graphs)
    eval_graphs.add_argument("--methods", nargs="+", choices=("exact", "fuzzy", "embedding"), default=["exact", "fuzzy"])
    eval_graphs.add_argument("--fuzzy-threshold", type=float, default=0.85)
    eval_graphs.add_argument("--embedding-threshold", type=float, default=0.90)
    eval_graphs.add_argument("--embedding-model", default="google/embeddinggemma-300m")
    eval_graphs.add_argument("--granularity", choices=("document", "procedure_index"), default="document")
    eval_graphs.add_argument("--ged", action="store_true")
    eval_graphs.add_argument("--ged-timeout", type=float, default=5.0)
    eval_graphs.add_argument("--allow-download", action="store_true")
    eval_graphs.set_defaults(func=cmd_evaluate_graphs)

    for name, func in (("evaluate-instructions", cmd_evaluate_instructions), ("evaluate-slots", cmd_evaluate_slots)):
        command = sub.add_parser(name)
        command.add_argument("--input", required=True, type=Path)
        command.add_argument("--gold", required=True, type=Path)
        _add_output(command)
        _add_similarity(command)
        if name == "evaluate-slots":
            command.add_argument("--threshold", type=float, default=0.85)
        command.set_defaults(func=func)

    parse_eval = sub.add_parser("evaluate-parsing", help="Evaluate only against verified page text/layout gold")
    parse_eval.add_argument("--input", required=True, type=Path)
    parse_eval.add_argument("--gold", type=Path)
    _add_output(parse_eval)
    parse_eval.set_defaults(func=cmd_evaluate_parsing)

    smart = sub.add_parser("evaluate-smart", help="Official SMART adapter (requires verified transcribed pages)")
    smart.add_argument("--input", required=True, type=Path)
    smart.add_argument("--gold", required=True, type=Path)
    smart.add_argument("--matcher", choices=("chrf", "bleurt20"), default="chrf")
    smart.add_argument("--bleurt-checkpoint", type=Path)
    _add_output(smart)
    smart.set_defaults(func=cmd_evaluate_smart)

    layout_tasks = sub.add_parser("layout-template", help="Blank page layout annotations (no pseudo-gold)")
    layout_tasks.add_argument("--input", required=True, type=Path)
    _add_output(layout_tasks)
    layout_tasks.set_defaults(func=cmd_layout_template)

    expert_tasks = sub.add_parser("expert-template", help="Blank certified-expert review tasks")
    expert_tasks.add_argument("--input", required=True, type=Path)
    _add_output(expert_tasks)
    expert_tasks.set_defaults(func=cmd_expert_template)

    expert_eval = sub.add_parser("expert-agreement", help="Pearson/Spearman/accuracy on verified expert judgments")
    expert_eval.add_argument("--input", required=True, type=Path)
    expert_eval.add_argument("--gold", type=Path)
    _add_output(expert_eval)
    expert_eval.set_defaults(func=cmd_expert_agreement)

    agreement = sub.add_parser("agreement-annotators", help="IAA from two independent annotated step files")
    agreement.add_argument("--first", required=True, type=Path)
    agreement.add_argument("--second", required=True, type=Path)
    agreement.add_argument("--normalization", type=Path, help="Approved equivalence mappings for conceptual scores")
    _add_output(agreement)
    agreement.set_defaults(func=cmd_inter_annotator_agreement)

    human_template = sub.add_parser("human-template", help="Blank instruction/slot quality review tasks")
    human_template.add_argument("--input", required=True, type=Path)
    _add_output(human_template)
    human_template.set_defaults(func=cmd_human_template)

    human_report = sub.add_parser("human-report", help="Summarize verified four-level human ratings")
    human_report.add_argument("--input", required=True, type=Path)
    _add_output(human_report)
    human_report.set_defaults(func=cmd_human_report)

    compare = sub.add_parser("compare-pipelines", help="Paired F1 and bootstrap CI for multistage versus VLM")
    compare.add_argument("--multistage", required=True, type=Path)
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--gold", required=True, type=Path)
    compare.add_argument("--method", choices=("exact", "fuzzy", "embedding"), default="fuzzy")
    compare.add_argument("--threshold", type=float, default=0.85)
    compare.add_argument("--embedding-model", default="google/embeddinggemma-300m")
    compare.add_argument("--bootstrap-samples", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--allow-download", action="store_true")
    _add_output(compare)
    compare.set_defaults(func=cmd_compare_pipelines)

    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = argv if argv is not None else sys.argv[1:]
    # Backward compatibility with the original v0.1 audit-only entrypoint.
    if arguments and arguments[0].startswith("--") and arguments[0] != "--help":
        arguments = ["audit", *arguments]
    parser = build_parser()
    args = parser.parse_args(arguments)
    if hasattr(args, "limit") and args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    if hasattr(args, "top_k") and args.top_k < 1:
        parser.error("--top-k must be positive")
    if hasattr(args, "retry_errors") and args.retry_errors and not args.resume:
        parser.error("--retry-errors requires --resume")
    if getattr(args, "resume", False) and args.command not in {
        "gold-graphs", "parse", "instructions", "slots", "graphs", "safety", "baseline",
    }:
        parser.error("--resume is supported only for per-document checkpointed stages")
    try:
        if hasattr(args, "output"):
            for source_name in (
                "input", "gold", "rules", "selection_file", "normalization",
                "first", "second", "multistage", "baseline", "query_input", "gold_labels",
            ):
                source = getattr(args, source_name, None)
                if isinstance(source, Path) and source.resolve() == args.output.resolve():
                    raise ValueError(f"Refusing to replace input {source_name} with its own output")
        args.func(args)
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, NotADirectoryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
