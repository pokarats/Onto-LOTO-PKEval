# Reproducibility and validity audit

## Provenance

This package and its descriptions is an independent reconstruction and sanitized version from the paper
draft and the supplied `public_data` directory. It **does not** claim to
recover the missing original code, model output files, expert judgments,
private experiment data, or ontology. The draft references a future GitHub
release; this archive is a new implementation with its own package version.

**Dataset checked:** the user's public ZIP supplied in October 2026. Do not
publish the source images in this code repository until their individual
license terms are checked. Code uses only public data, not any `proprietary_*` path.

## What was tested in this environment

| Check | Result |
| --- | --- |
| Public document folders | 20 |
| Source documents | 19 raster images + 1 PDF |
| Available visual page files | 21 (including 2 PDF renders) |
| Annotation action records | 195 |
| Groups inferred from title/order resets | 42 |
| Missing title annotations | 1 (document 14) |
| OCR run | All 21 pages completed with some noisy text |
| Reference graph conversion | All 20 documents completed |
| Reference graph facts (derived, not authored) | 521 triplets |
| PyMuPDF native text | 19 image-only documents explicitly marked no text; PDF processed |
| Deterministic keyword audit | All 20 documents completed; flags are not certifications |
| Unit tests | Passed (see `python3 -m unittest ...`) |
| Local Ollama/Qwen3 inference | **Not executed**; model/service unavailable here |
| HF Qwen3.5/Gemma4/CLIP/PaliGemma inference | **Not executed**; weights/GPU unavailable here |
| Docling | **Not executed**; optional dependency unavailable here |
| BLEURT-20 SMART | **Not executed**; checkpoint and optional packages unavailable here |

The draft's public-only Table 2 instead gives 18 documents, 36 procedures,
62 instructions, and 21 annotated visual evidence items. The source material
does not specify the two excluded document IDs or a defensible filter mapping
195 action records to 62 scored instructions. **Do not optimize a filter to
match these published counts**. Recover the authors' evaluation manifest or
state that the subset differs. The 21 available page images are not equivalent
to 21 verified figure-region annotations.

## Stage-by-stage scientific parity

| Draft experiment | Runnable code | What is missing / differs |
| --- | --- | --- |
| Public data inventory | `audit`, `gold-graphs` | Inferred groups; unknown manuscript subset |
| Docling/PyMuPDF parsing | `parse --backend ...` | Most inputs are raster, not the original paper's PDF representation |
| Layout IoU | `layout-template`, `evaluate-parsing` | Annotated region boxes and coordinate transforms |
| SMART-1/L | `evaluate-smart` optional adapter | Gold page transcriptions; original BLEURT-20 checkpoint, verified integration |
| Qwen3:14B extraction | `instructions`, `slots` | Reconstructed prompt examples; unavailable original exact prompt and runs |
| PKO graph construction | `graphs --atomize` | PKO-inspired typed projection, not original expert ontology mapping |
| Node/edge/triplet alignment | `evaluate-graphs`, `evaluate-slots` | Approximate procedure boundaries; matching rules may differ from original |
| GED | `evaluate-graphs --ged` | NetworkX with stated costs/timeout; original graph implementation missing |
| Step text quality | `evaluate-instructions` | Text pairing by document order; no paper reference grouping |
| Ontology-only rule audit | `safety --mode ontology_only` | Starter keyword constraints **not expert verified** |
| Ontology-LLM audit | `safety --mode ontology_llm` | Same exploratory rule set; local Qwen service required |
| Expert accuracy / Pearson / Spearman | `expert-template`, `expert-agreement` | Independent certified LOTO assessor responses |
| CLIP visual grounding | `ground --method clip` | Gold subimage regions/links absent; retrieval is page-level |
| PaliGemma visual grounding | `ground --method paligemma` | Uses generative logit margin, **not** publication's unspecified embedding cosine |
| Qwen3.5/Gemma4 baselines | `baseline --model ...` | HF weights/GPU; prompt reconstruction; original decoding unknown |
| Cross-modal Acc@k | `ground --gold-labels ...` | Verified image/region evidence mappings |
| Multi-stage vs end-to-end | `compare-pipelines` | Aligned model outputs missing until re-run; unknown paper subset |
| Human CORRECT/PARTIAL/WRONG/N-SURE review | `human-template`, `human-report` | Human decisions |
| Exact and conceptual IAA | `agreement-annotators` | Second independent annotations and approved conceptual equivalence map |

## Scientific design differences to disclose

1. **Baseline input domain:** the 19 image-only public files do not contain
   native PDF text. PyMuPDF's native extractor should not be judged for OCR on
   those inputs. A rasterized-PDF experiment could be run as a separate
   control, but it would not reproduce the source document representation.
2. **Procedure pairing:** `--granularity procedure_index` pairs the first
   inferred group to the first inferred group, etc. It is not an expert-verified
   correspondence. Default document-level scoring avoids claiming otherwise.
3. **Triplet matching:** our one-to-one maximum-cardinality matching accepts
   a triplet if the relation matches and the mean subject/object similarity is
   at least the specified threshold. The augmenting path method maximizes
   *count*, not summed similarity. Fuzzy threshold defaults to 0.85,
   embedding to 0.90. All details are explicit in JSON reports.
4. **Graph Edit Distance:** computed by NetworkX (if installed), with unit
   insertions/deletions, same-type node substitutions accepted if similarity
   exceeds the requested method threshold, and exact edge-relation costs.
   Timeouts and normalization are logged. This is not necessarily the
   published custom normalization.
5. **Safety:** ontology constraints are exploratory lexical research rules.
   They must not be used to decide whether any machine is safely isolated.
   The output metadata says `NOT_PERFORMED` for safety certification.
6. **PaliGemma:** unlike contrastive CLIP embeddings, a generative checkpoint
   does not expose the same simple image/text cosine. Our explicit yes/no
   first-token logit margin is a different ranking baseline. Never label it
   "PaliGemma embedding cosine".
7. **Grounding:** where an input image is a full scanned page, document-ID
   matching is only a rough information-retrieval proxy. Accuracy against
   cropped visual regions requires separately annotated candidates and
   links, plus a faithful image-region candidate generator.
8. **Human evaluation:** a `verified` status is assigned only by the human
   annotator/export workflow. Blank templates default to
   `needs_human_annotation`, `needs_human_review`, or
   `needs_certified_expert_review`; no code promotes them automatically.

## Minimum requirements for a defensible new paper result

1. Freeze the evaluation document IDs and write a transparent selection file.
2. Have an annotator confirm procedure segmentation and which annotations
   are executable vs. conditional rather than filtering by desired count.
3. Preserve image/page/region references and annotate fine-grained grounding
   for each evaluated step/entity.
4. Obtain independent expert rule approval and safety judgments before
   computing compliance agreement; explicitly document jurisdiction and
   applicable industrial safety standard.
5. Pin the exact model checkpoint revisions, package versions, hardware,
   prompts, and decoding options in a run manifest. Our sidecars record source
   SHA-256 checksums, versions, and arguments, but are not a lockfile.
6. Run the controlled comparisons on the *same eligible documents* and
   report both absolute metric values and coverage/failed inference rates.
7. Treat public template limitations (generic language, missing equipment
   identifiers, scans, source rights) as part of the threat-to-validity analysis.

## How to interpret checkpoint files

Every long stage writes `<stage>.jsonl` atomically after each document and
`<stage>.jsonl.meta.json` at completion. `status: error` rows preserve the
document ID and error message; the command exits nonzero. Use
`--resume --retry-errors` after fixing the underlying cause. `status: ok`
means the stage executed and its output schema validated; it does **not**
imply the predicted actions are correct or safe.

Input file hashes and the environment's package versions are recorded in
sidecars. The original sources remain unchanged, and `audit/documents.jsonl`
has source-relative names and SHA-256 checksums for every source asset.
