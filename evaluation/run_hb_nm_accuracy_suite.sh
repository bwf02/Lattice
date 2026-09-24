#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(dirname "$ROOT_DIR")}"
SGLANG_DIR="${SGLANG_DIR:-$WORKSPACE_ROOT/sglang}"
SPARSE_GEMM_DIR="${SPARSE_GEMM_DIR:-$WORKSPACE_ROOT/SparseGEMM}"
PYTHON="${PYTHON:-/tmp/mosaicmoe-accuracy-venv/bin/python}"
MODEL="${MODEL:-/ossfs/workspace/MosaicMoE/models/Qwen1.5-MoE-A2.7B}"
DATASET_ROOT="${DATASET_ROOT:-/tmp/mosaicmoe-eval-datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/hb_nm_accuracy_results}"
NSAMPLES="${NSAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LIMIT="${LIMIT:-}"
PPL_MAX_TOKENS="${PPL_MAX_TOKENS:-}"
STAGGER_SECONDS="${STAGGER_SECONDS:-20}"
SKIP_PPL="${SKIP_PPL:-0}"
SAVE_SAMPLES="${SAVE_SAMPLES:-1}"

export PYTHONPATH="${SGLANG_DIR}/python:${SPARSE_GEMM_DIR}:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export WIKITEXT2_DATA_DIR="${DATASET_ROOT}/wikitext_salesforce/wikitext-2-raw-v1"
mkdir -p "$OUTPUT_DIR"

common=(
  "$ROOT_DIR/evaluation/run_sparse_accuracy_legacy.py"
  --model "$MODEL"
  --dataset-root "$DATASET_ROOT"
  --nsamples "$NSAMPLES"
  --batch-size "$BATCH_SIZE"
  --seed 0
  --skip-humaneval
  --block-h 16
  --block-w 16
  --hybrid-block-score sum
)
if [[ "$SKIP_PPL" == "1" ]]; then
  common+=(--skip-ppl)
fi
if [[ -n "$LIMIT" ]]; then
  common+=(--limit "$LIMIT")
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
run_one 0 hb_nm_25 --sparsity-type hybrid_block_sparse --sparsity-ratio 0.25 --block-n 1 --block-m 2 --kept-pattern 'HB 25%: 1/2 blocks use 2:4'
run_one 1 hb_nm_50 --sparsity-type hb_nm --sparsity-ratio 0.50 --block-n 1 --block-m 1 --kept-pattern 'HB 50%: all blocks use 2:4'
run_one 2 hb_nm_75 --sparsity-type hb_nm --sparsity-ratio 0.75 --block-n 1 --block-m 2 --kept-pattern 'HB 75%: 1/2 blocks kept with 2:4'

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "Completed ${labels[$i]}"
  else
    echo "Failed ${labels[$i]}; see $OUTPUT_DIR/${labels[$i]}.log" >&2
    status=1
  fi
done

if [[ "$status" -eq 0 ]]; then
  "$PYTHON" "$ROOT_DIR/evaluation/summarize_sparse_accuracy.py" \
    --input-dir "$OUTPUT_DIR" \
    --labels hb_nm_25,hb_nm_50,hb_nm_75 \
    --markdown "$OUTPUT_DIR/summary.md" \
    --csv "$OUTPUT_DIR/summary.csv"
fi
exit "$status"
