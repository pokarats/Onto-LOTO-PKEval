# Ontology-Guided Procedural Knowledge Extraction and Validation Case Study on the Lockout-Tagout Documents

This repository contains the research implementation (**public-data-only**) for the paper:
*Ontology-Guided Procedural Knowledge Extraction and Validation: Case Study on
the Lockout-Tagout Documents*.

**Release 0.2.0 — runnable stages, not a recovery of the original results.**
The original prompt set, expert-authored safety ontology, annotated figure
regions, visual grounding links, and exact 18-document evaluation subset were
not available. This package implements those stages but does not manufacture
missing expert judgments or reference labels. Read
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) before citing metrics.

The package contains **no datasets, web documents, proprietary PDFs, model
weights, or expert labels**. It consumes `data/public_data/` only. It never
loads `proprietary_*` data or `public_batches`.

## Install and locate data

Place `loto_pke_public/` beside the existing `data/` directory:

```text
/Onto-LOTO-PKEval/
├── data/public_data/
│   ├── 1/        1.webp, 1.txt
│   ├── ...
│   └── 20/       20.pdf, 20.txt, page_1.png, page_2.png
└───
    ├── configs/  
    ├── docs/  
    ├── scripts/
    ├── src/loto_pke/  
    ├── tests/  
    └── pyproject.toml
    └── README.md
    
```

From the codebase directory, install in a new Python 3.10+ environment:

```bash
python3 -m pip install -e .
python3 -m pip install -e '.[ocr]'         # Pillow, PyMuPDF
python3 -m pip install -e '.[parsing]'     # optional Docling
python3 -m pip install -e '.[evaluation]'  # optional embedding, NLTK, NetworkX
python3 -m pip install -e '.[vision]'      # optional HF Transformers, torch
```

The separate `tesseract` binary and English trained data must be installed
for OCR. Optional dependencies are loaded lazily. Hugging Face weights are
local-only by default; `--allow-download` is an explicit opt-in. The two
invocation styles are equivalent: `loto-pke` and
`PYTHONPATH=src python3 -m loto_pke`.

For command examples below, set:

```bash
DATA_ROOT=~/Onto-LOTO-PKEva/data/public_data
OUT_ROOT=outputs/public_v0_2
mkdir -p "$OUT_ROOT"
```

## A. Dataset audit, gold graph, document parsing

```bash
PYTHONPATH=src python3 -m loto_pke audit \
  --data-root "$DATA_ROOT" --output-dir "$OUT_ROOT/audit"

PYTHONPATH=src python3 -m loto_pke gold-graphs \
  --data-root "$DATA_ROOT" --output "$OUT_ROOT/gold_graphs.jsonl"

PYTHONPATH=src python3 -m loto_pke parse \
  --data-root "$DATA_ROOT" --backend tesseract \
  --output "$OUT_ROOT/parsed_tesseract.jsonl"

# Run separately if Docling is installed:
PYTHONPATH=src python3 -m loto_pke parse \
  --data-root "$DATA_ROOT" --backend docling \
  --output "$OUT_ROOT/parsed_docling.jsonl"

# Native PDF text only; raster sources explicitly show no native text:
PYTHONPATH=src python3 -m loto_pke parse \
  --data-root "$DATA_ROOT" --backend pymupdf \
  --output "$OUT_ROOT/parsed_pymupdf.jsonl"
```

`gold_graphs.jsonl` is an **annotation-derived reference**, not a model
prediction. Typed Step/Action/Target/Tool/Location/Verification nodes,
semantic triplets, immediate precedence, and separately labeled transitive
precedence retain the input's original action wording.

## B. Two-stage Qwen3:14B extraction and PKG construction

The draft paper used `qwen3:14b` through a local Ollama installation. Start
your authorized Ollama service on the compute node and install the checkpoint
before running these commands. Only loopback Ollama URLs are accepted; use a
local SSH tunnel if the service is on another permitted host.

```bash
PYTHONPATH=src python3 -m loto_pke instructions \
  --input "$OUT_ROOT/parsed_docling.jsonl" --model qwen3:14b \
  --output "$OUT_ROOT/instructions_qwen.jsonl"

PYTHONPATH=src python3 -m loto_pke slots \
  --input "$OUT_ROOT/instructions_qwen.jsonl" --model qwen3:14b \
  --output "$OUT_ROOT/slots_qwen.jsonl"

PYTHONPATH=src python3 -m loto_pke graphs \
  --input "$OUT_ROOT/slots_qwen.jsonl" --atomize --model qwen3:14b \
  --output "$OUT_ROOT/graphs_multistage.jsonl"
```

Prompts live in `src/loto_pke/extraction.py`, versioned and hashed, with
concise examples. They are **independent reconstructions** of the described
tasks, not the missing original prompts. `--atomize` re-prompts each action
for ordered atomic action/argument relations; omit for a controlled ablation.
The extraction model uses temperature 0, seed 42, JSON mode, and 1,024 maximum
tokens by default; every response and validation failure is retained.

Runs atomically checkpoint after each document. Restart with `--resume`;
failed records can be retried with `--resume --retry-errors`. Existing output
is never silently overwritten (`--overwrite` is explicit). Jobs with failed
document records exit nonzero so Slurm can detect them.

## C. Reference and parsing evaluation

```bash
PYTHONPATH=src python3 -m loto_pke evaluate-graphs \
  --input "$OUT_ROOT/graphs_multistage.jsonl" \
  --gold "$OUT_ROOT/gold_graphs.jsonl" --methods exact fuzzy \
  --output "$OUT_ROOT/pkg_metrics.json"

# Local embedding model and NetworkX must already be installed:
PYTHONPATH=src python3 -m loto_pke evaluate-graphs \
  --input "$OUT_ROOT/graphs_multistage.jsonl" \
  --gold "$OUT_ROOT/gold_graphs.jsonl" --methods exact fuzzy embedding \
  --embedding-model google/embeddinggemma-300m \
  --granularity procedure_index --ged \
  --output "$OUT_ROOT/pkg_metrics_3strategies.json"

PYTHONPATH=src python3 -m loto_pke evaluate-instructions \
  --input "$OUT_ROOT/instructions_qwen.jsonl" \
  --gold "$OUT_ROOT/gold_graphs.jsonl" --method fuzzy \
  --output "$OUT_ROOT/instruction_metrics.json"

# Cosine-based semantic alignment, closer to the draft's intended metric:
PYTHONPATH=src python3 -m loto_pke evaluate-instructions \
  --input "$OUT_ROOT/instructions_qwen.jsonl" \
  --gold "$OUT_ROOT/gold_graphs.jsonl" --method embedding \
  --embedding-model google/embeddinggemma-300m \
  --output "$OUT_ROOT/instruction_semantic_alignment.json"

PYTHONPATH=src python3 -m loto_pke evaluate-slots \
  --input "$OUT_ROOT/slots_qwen.jsonl" \
  --gold "$OUT_ROOT/gold_graphs.jsonl" --method fuzzy --threshold 0.85 \
  --output "$OUT_ROOT/slot_metrics.json"
```

Available scores include exact/fuzzy/embedding one-to-one triplet matching,
typed node P/R/F1, node/edge Jaccard, optional normalized graph edit distance
with matching-strategy-dependent substitutions, ROUGE-1/L, BLEU-4, optional
METEOR, and step alignment cosine/accuracy/off-diagonal/margin. `--granularity
document` evaluates documents; `procedure_index` is **ordinal pairing of
inferred groups**. The latter does not recover the paper's manually selected
36 procedure boundaries and is not applicable to direct VLM graphs with no
step groups.

The structured TXT files are *not* human transcriptions of every PDF page,
and contain no visual-region boxes. For parsing IoU and SMART:

```bash
PYTHONPATH=src python3 -m loto_pke layout-template \
  --input "$OUT_ROOT/parsed_docling.jsonl" \
  --output "$OUT_ROOT/layout_needs_annotation.jsonl"

PYTHONPATH=src python3 -m loto_pke evaluate-parsing \
  --input "$OUT_ROOT/parsed_docling.jsonl" \
  --output "$OUT_ROOT/layout_unavailable_report.json"

# Once a human annotator provides verified page boxes/text:
PYTHONPATH=src python3 -m loto_pke evaluate-parsing \
  --input "$OUT_ROOT/parsed_docling.jsonl" \
  --gold PATH_TO_VERIFIED_LAYOUT.jsonl \
  --output "$OUT_ROOT/layout_metrics.json"

# Optional official SMART adapter, using a CHRF matcher (NOT paper's BLEURT):
PYTHONPATH=src python3 -m loto_pke evaluate-smart \
  --input "$OUT_ROOT/parsed_docling.jsonl" \
  --gold PATH_TO_VERIFIED_LAYOUT.jsonl --matcher chrf \
  --output "$OUT_ROOT/smart_chrf.json"
```

SMART with BLEURT-20 also has a `--matcher bleurt20
--bleurt-checkpoint CHECKPOINT_PATH` adapter. It requires Google's SMART and
BLEURT installation and must be integration-tested before claiming parity.

For SMART Evaluation, first follow the installation in
https://github.com/google-research/google-research/tree/master/smart_eval.

## D. Exploratory safety compliance research

```bash
PYTHONPATH=src python3 -m loto_pke safety \
  --input "$OUT_ROOT/graphs_multistage.jsonl" \
  --mode ontology_only --rules configs/safety_rules_exploratory.json \
  --output "$OUT_ROOT/safety_rules.jsonl"

PYTHONPATH=src python3 -m loto_pke safety \
  --input "$OUT_ROOT/graphs_multistage.jsonl" \
  --mode ontology_llm --model qwen3:14b \
  --rules configs/safety_rules_exploratory.json \
  --output "$OUT_ROOT/safety_ollama.jsonl"

PYTHONPATH=src python3 -m loto_pke expert-template \
  --input "$OUT_ROOT/safety_rules.jsonl" \
  --output "$OUT_ROOT/safety_expert_needs_review.jsonl"

PYTHONPATH=src python3 -m loto_pke expert-agreement \
  --input "$OUT_ROOT/safety_rules.jsonl" \
  --output "$OUT_ROOT/safety_without_expert_labels.json"

# After independent expert verification:
PYTHONPATH=src python3 -m loto_pke expert-agreement \
  --input "$OUT_ROOT/safety_rules.jsonl" \
  --gold PATH_TO_VERIFIED_EXPERT_LABELS.jsonl \
  --output "$OUT_ROOT/safety_expert_agreement.json"
```

**Safety-critical limitation:** The example rules are not expert-verified. The expert-verified annotations used in
published experiment results overlap with proprietary documents, the contents of which we do not have permission to
disclose. Keyword judgments flag possible errors for manual assessment, and the
Ontology-LLM is not a certified safety checker. No output authorizes work or
certifies regulatory compliance. Expert metrics use only verified annotations
and treat `Partially` as `No` on the energy-source question as in the appendix.

## E. CLIP/PaliGemma grounding and Qwen3.5/Gemma 4 visual baselines

```bash
PYTHONPATH=src python3 -m loto_pke grounding-template \
  --data-root "$DATA_ROOT" \
  --output "$OUT_ROOT/grounding_needs_annotation.jsonl"

PYTHONPATH=src python3 -m loto_pke ground \
  --data-root "$DATA_ROOT" --method clip --top-k 2 \
  --output "$OUT_ROOT/clip_rankings.jsonl"

PYTHONPATH=src python3 -m loto_pke ground \
  --data-root "$DATA_ROOT" --method paligemma --top-k 2 \
  --output "$OUT_ROOT/paligemma_rankings.jsonl"

PYTHONPATH=src python3 -m loto_pke baseline \
  --data-root "$DATA_ROOT" --generator hf \
  --model Qwen/Qwen3.5-4B --dtype float16 \
  --mode direct_graph --visual-links --max-new-tokens 2048 \
  --output "$OUT_ROOT/qwen35_direct.jsonl"

PYTHONPATH=src python3 -m loto_pke baseline \
  --data-root "$DATA_ROOT" --generator hf \
  --model google/gemma-4-E4B-it --dtype float16 \
  --mode direct_graph --visual-links --max-new-tokens 2048 \
  --output "$OUT_ROOT/gemma4_direct.jsonl"
```

Models are local, with HF `device_map="auto"`; use `--dtype float16` on older
GPUs without BF16 support. `direct_graph` asks the VLM for nodes and edges in
one pass. `structured_steps` instead generates complete slots in one VLM
call, then uses the same graph mapper as the multi-step system. These are
distinct baselines. Fine-grained cross-modal Acc@1/2 requires a verified
`--gold-labels` file; absent this, reports contain a **document-level proxy**.
PaliGemma's implemented **yes/no logit margin** is not CLIP cosine nor the
paper's unspecified PaliGemma embedding score.

For a **paired** comparison on the intersection of successfully evaluated
documents, with bootstrap confidence intervals (multi-step minus VLM):

```bash
PYTHONPATH=src python3 -m loto_pke compare-pipelines \
  --multistage "$OUT_ROOT/graphs_multistage.jsonl" \
  --baseline "$OUT_ROOT/qwen35_direct.jsonl" \
  --gold "$OUT_ROOT/gold_graphs.jsonl" \
  --method fuzzy --threshold 0.85 \
  --output "$OUT_ROOT/qwen35_vs_multistage.json"
```

This comparison reports triplet and Action-node F1 differences. The original
conditional-step and visual-evidence F1 remain `null` until independently
verified conditional labels and fine-grained visual grounding gold exist.

## F. Human inspection and inter-annotator agreement

```bash
PYTHONPATH=src python3 -m loto_pke human-template \
  --input "$OUT_ROOT/slots_qwen.jsonl" \
  --output "$OUT_ROOT/human_needs_review.jsonl"

PYTHONPATH=src python3 -m loto_pke human-report \
  --input PATH_TO_VERIFIED_HUMAN_REVIEW.jsonl \
  --output "$OUT_ROOT/human_review_summary.json"

PYTHONPATH=src python3 -m loto_pke agreement-annotators \
  --first PATH_TO_ANNOTATOR_A_STEPS.jsonl \
  --second PATH_TO_ANNOTATOR_B_STEPS.jsonl \
  --output "$OUT_ROOT/iaa_exact.json"
```

Human labels are `CORRECT / PARTIAL / WRONG / N-SURE`. Concept-normalized
inter-annotator agreement requires an explicit approved synonym/ontology map.
`configs/conceptual_normalization_template.json` is intentionally **unfilled**.

## Tests, known input counts, and reproducibility

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

In the supplied archive, 20 public document folders yielded 21 page images,
195 structured action records, and 42 **inferred** title/order groups. One
record in document 14 has no title. All 20 documents passed CPU reference
graph construction and all 21 pages passed OCR. These counts are **not model
quality results**.

The paper reports 18 public documents, 36 procedures, 62 instructions, and 21
visual evidence items. Neither its excluded document IDs nor its instruction
filtering rule can be recovered from the supplied ZIP. If the experimental
subset is later confirmed, pass `--selection-file FILE` (one public document
ID per line) rather than guessing which examples to drop.

Web accessibility does not imply permission to redistribute each source image
or document. Keep the dataset separate from this source-code release until
individual licenses and usage rights are verified.


