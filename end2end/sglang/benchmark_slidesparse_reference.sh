#!/usr/bin/env bash
# Match the cases in the archived 20260901 paper figure, without baseline reruns.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${RESULT_ROOT:?Set RESULT_ROOT to a new collection directory}
MODELS=${MODEL_NAMES:-qwen15 deepseek_v2_lite qwen3 llama4_scout}
export BACKENDS=slidesparse
export BENCH_REPEATS=${BENCH_REPEATS:-1}
export WARMUP_REPEATS=${WARMUP_REPEATS:-1}
export PREFILL_M_VALUES=${PREFILL_M_VALUES:-32768 16384 8192 4096}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

for model in $MODELS; do
  if [[ "$model" == qwen3 ]]; then
    # Prefill and BS8/16 were recollected at TP4 with fusion disabled.
    RESULT_ROOT="$ROOT/qwen3-tp4" MODEL_NAMES=qwen3 PROFILES='prefill decode' \
      QWEN3_TP_SIZE=4 QWEN3_GPU_SET=0,1,2,3 QWEN3_MEM_FRACTION=0.68 \
      QWEN3_DISABLE_ALLREDUCE_FUSION=1 DECODE_CONCURRENCIES='8 16' \
      SLIDESPARSE_ACTIVATION_CHUNK_M=1024 \
      bash "$SCRIPT_DIR/benchmark_paper_e2e.sh"
    # BS32/64 retain the earlier TP2 measurements in the paper figure.
    RESULT_ROOT="$ROOT/qwen3-tp2" MODEL_NAMES=qwen3 PROFILES=decode \
      QWEN3_TP_SIZE=2 QWEN3_GPU_SET=0,1 QWEN3_MEM_FRACTION=0.60 \
      QWEN3_DISABLE_ALLREDUCE_FUSION=0 DECODE_CONCURRENCIES='32 64' \
      SLIDESPARSE_ACTIVATION_CHUNK_M=1024 \
      bash "$SCRIPT_DIR/benchmark_paper_e2e.sh"
  else
    RESULT_ROOT="$ROOT/$model" MODEL_NAMES="$model" PROFILES='prefill decode' \
      DECODE_CONCURRENCIES='8 16 32 64' \
      bash "$SCRIPT_DIR/benchmark_paper_e2e.sh"
  fi
done
