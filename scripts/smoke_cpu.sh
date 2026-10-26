#!/usr/bin/env bash
# Usage: bash scripts/smoke_cpu.sh /absolute/path/data/public_data /absolute/path/new-smoke-output
# Never publishes or modifies input documents. No model weights are needed.
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: bash scripts/smoke_cpu.sh DATA_ROOT NEW_OUTPUT_DIR" >&2
  exit 2
fi

DATA_ROOT="$1"
OUTPUT_ROOT="$2"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Output target already exists; choose a new directory: $OUTPUT_ROOT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
cd "$PROJECT_ROOT"

PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m loto_pke audit \
  --data-root "$DATA_ROOT" --output-dir "$OUTPUT_ROOT/audit"
PYTHONPATH=src python3 -m loto_pke gold-graphs \
  --data-root "$DATA_ROOT" --limit 2 --output "$OUTPUT_ROOT/gold_2.jsonl"
PYTHONPATH=src python3 -m loto_pke parse \
  --data-root "$DATA_ROOT" --limit 2 --backend tesseract \
  --output "$OUTPUT_ROOT/ocr_2.jsonl"
PYTHONPATH=src python3 -m loto_pke safety \
  --input "$OUTPUT_ROOT/gold_2.jsonl" \
  --mode ontology_only --rules configs/safety_rules_exploratory.json \
  --output "$OUTPUT_ROOT/safety_2.jsonl"
PYTHONPATH=src python3 -m loto_pke evaluate-parsing \
  --input "$OUTPUT_ROOT/ocr_2.jsonl" \
  --output "$OUTPUT_ROOT/parsing_missing_gold.json"

echo "Smoke pipeline completed (no model inference, no certification): $OUTPUT_ROOT"
