#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(dirname "$ROOT_DIR")}"
SGLANG_DIR="${SGLANG_DIR:-$WORKSPACE_ROOT/sglang}"
SPARSE_GEMM_DIR="${SPARSE_GEMM_DIR:-$WORKSPACE_ROOT/SparseGEMM}"
PYTHON="${PYTHON:-/tmp/mosaicmoe-accuracy-venv/bin/python}"
MODEL="${MODEL:-/ossfs/workspace/MosaicMoE/models/Qwen1.5-MoE-A2.7B}"
DATASET_ROOT="${DATASET_ROOT:-/tmp/mosaicmoe-eval-datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/sparse_accuracy_results}"
NSAMPLES="${NSAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LIMIT="${LIMIT:-}"
STAGGER_SECONDS="${STAGGER_SECONDS:-20}"
CONFIGS="${CONFIGS:-all}"
PPL_ONLY="${PPL_ONLY:-0}"
SKIP_PPL="${SKIP_PPL:-0}"
SKIP_HUMANEVAL="${SKIP_HUMANEVAL:-1}"
PPL_MAX_TOKENS="${PPL_MAX_TOKENS:-}"
SAVE_SAMPLES="${SAVE_SAMPLES:-0}"

export PYTHONPATH="${SGLANG_DIR}/python:${SPARSE_GEMM_DIR}:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export WIKITEXT2_DATA_DIR="${DATASET_ROOT}/wikitext_salesforce/wikitext-2-raw-v1"
mkdir -p "$OUTPUT_DIR"

common=(
  "$ROOT_DIR/evaluation/run_sparse_accuracy.py"
  --model "$MODEL"
  --dataset-root "$DATASET_ROOT"
  --nsamples "$NSAMPLES"
  --batch-size "$BATCH_SIZE"
  --seed 0
)
if [[ -n "$LIMIT" ]]; then
  common+=(--limit "$LIMIT")
fi
if [[ "$PPL_ONLY" == "1" ]]; then
  common+=(--ppl-only)
fi
if [[ "$SKIP_PPL" == "1" ]]; then
  common+=(--skip-ppl)
fi
if [[ "$SKIP_HUMANEVAL" == "1" ]]; then
  common+=(--skip-humaneval)
fi
if [[ -n "$PPL_MAX_TOKENS" ]]; then
  common+=(--ppl-max-tokens "$PPL_MAX_TOKENS")
fi
if [[ "$SAVE_SAMPLES" == "1" ]]; then
  common+=(--save-samples)
fi

run_one() {
  local gpu="$1" label="$2"
  shift 2
  if [[ "$CONFIGS" != "all" && ",$CONFIGS," != *",$label,"* ]]; then
    echo "Skipping $label"
    return
  fi
  echo "Launching $label on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "${common[@]}" \
    --label "$label" --output "$OUTPUT_DIR/$label.json" "$@" \
    >"$OUTPUT_DIR/$label.log" 2>&1 &
  pids+=("$!")
  labels+=("$label")
  sleep "$STAGGER_SECONDS"
}

pids=()
labels=()
run_one 0 dense --sparsity-type dense
run_one 1 unstructured_25 --sparsity-type unstructured --sparsity-ratio 0.25
run_one 2 unstructured_50 --sparsity-type unstructured --sparsity-ratio 0.50
run_one 3 unstructured_75 --sparsity-type unstructured --sparsity-ratio 0.75
run_one 4 nm_2_4 --sparsity-type 2:4 --sparsity-ratio 0.50 --prune-n 2 --prune-m 4 --kept-pattern 2:4
run_one 5 nm_1_4 --sparsity-type 1:4 --sparsity-ratio 0.75 --prune-n 3 --prune-m 4 --kept-pattern 1:4

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "Completed ${labels[$i]}"
  else
    echo "Failed ${labels[$i]}; see $OUTPUT_DIR/${labels[$i]}.log" >&2
    status=1
  fi
done
complete_results=1
for label in dense unstructured_25 unstructured_50 unstructured_75 nm_2_4 nm_1_4; do
  if [[ ! -f "$OUTPUT_DIR/$label.json" ]]; then
    complete_results=0
  fi
done
if [[ "$status" -eq 0 && "$complete_results" -eq 1 ]]; then
  "$PYTHON" "$ROOT_DIR/evaluation/summarize_sparse_accuracy.py" \
    --input-dir "$OUTPUT_DIR" \
    --markdown "$OUTPUT_DIR/summary.md" \
    --csv "$OUTPUT_DIR/summary.csv"
fi
exit "$status"
