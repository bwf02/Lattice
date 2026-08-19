#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$(dirname "$REPO_ROOT")}
SGLANG_DIR=${SGLANG_DIR:-$WORKSPACE_ROOT/sglang}
SPARSE_GEMM_DIR=${SPARSE_GEMM_DIR:-$WORKSPACE_ROOT/SparseGEMM}
PYTHON_BIN=${PYTHON_BIN:-/tmp/sglang-venv/bin/python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
SERVER_PORT=${SERVER_PORT:-32200}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}
RESULT_ROOT=${RESULT_ROOT:-/tmp/e2e-sparse-vs-deepgemm-${RUN_TAG}}
MODEL_NAMES=${MODEL_NAMES:-qwen15}
BACKENDS=${BACKENDS:-deep_gemm sparse_gemm}
CHUNK_SIZES=${CHUNK_SIZES:-8192 16384 32768}
CONCURRENCIES=${CONCURRENCIES:-32 64 128}
BENCH_REPEATS=${BENCH_REPEATS:-3}
WARMUP_REPEATS=${WARMUP_REPEATS:-1}
PROMPT_WAVES=${PROMPT_WAVES:-1}
INPUT_LEN=${INPUT_LEN:-4096}
OUTPUT_LEN=${OUTPUT_LEN:-32}
SEED=${SEED:-42}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-1200}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.68}
CUDA_GRAPH_BS_DECODE=${CUDA_GRAPH_BS_DECODE:-32 64 128}
SGLANG_DUAL_STREAM_BASE=${SGLANG_DUAL_STREAM_BASE:-5a1c399975ffb890e157a20e821825285d6d3887}
SPARSE_PREFILL_BASE=${SPARSE_PREFILL_BASE:-0dea676ebc739f03381bc3b803ef68bffea27112}
SPARSE_PREBIND_BASE=${SPARSE_PREBIND_BASE:-1ed94387742044e7cc77dd6ffbcc2d871ff24652}

require_ancestor() {
  local repo=$1
  local required_commit=$2
  local feature=$3
  if ! git -C "$repo" merge-base --is-ancestor "$required_commit" HEAD; then
    echo "$repo HEAD does not contain required $feature commit $required_commit" >&2
    exit 1
  fi
}

require_ancestor "$SGLANG_DIR" "$SGLANG_DUAL_STREAM_BASE" "prefill dual-stream"
require_ancestor "$SPARSE_GEMM_DIR" "$SPARSE_PREFILL_BASE" "TP2 prefill tuning"
require_ancestor "$SPARSE_GEMM_DIR" "$SPARSE_PREBIND_BASE" "active-expert prebind"
sglang_commit=$(git -C "$SGLANG_DIR" rev-parse HEAD)
sparse_gemm_commit=$(git -C "$SPARSE_GEMM_DIR" rev-parse HEAD)

declare -A MODEL_DIRS=(
  [qwen15]=/ossfs/workspace/MosaicMoE/models/Qwen1.5-MoE-A2.7B
  [deepseek_v2_lite]=/tmp/models/DeepSeek-V2-Lite
  [qwen3]=/tmp/models/Qwen3-30B-A3B
  [ernie45]=/tmp/models/ERNIE-4.5-21B-A3B-PT
)
declare -A EXPORT_DIRS=(
  [qwen15]=/tmp/sparse_gemm_export_nm12_shared/Qwen1.5-MoE-A2.7B
  [deepseek_v2_lite]=/tmp/sparse_gemm_exports/DeepSeek-V2-Lite
  [qwen3]=/tmp/sparse_gemm_exports/Qwen3-30B-A3B
  [ernie45]=/tmp/sparse_gemm_exports/ERNIE-4.5-21B-A3B-PT
)

mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/results"
printf '%s\n' \
  "models=$MODEL_NAMES" \
  "backends=$BACKENDS" \
  "chunks=$CHUNK_SIZES" \
  "concurrencies=$CONCURRENCIES" \
  "repeats=$BENCH_REPEATS" \
  "warmup_repeats=$WARMUP_REPEATS" \
  "prompt_waves=$PROMPT_WAVES" \
  "input_len=$INPUT_LEN" \
  "output_len=$OUTPUT_LEN" \
  "seed=$SEED" \
  "cuda_graph_bs_decode=$CUDA_GRAPH_BS_DECODE" \
  "prefill_cuda_graph=disabled" \
  "prefill_dual_stream=enabled" \
  "active_expert_prebind=enabled" \
  "shared_experts_fusion=enabled" \
  "sparse_layout_policy=auto_for_8k_masked_for_16k_32k" \
  "sglang_commit=$sglang_commit" \
  "sparse_gemm_commit=$sparse_gemm_commit" >"$RESULT_ROOT/config.txt"

server_pid=
cleanup_server() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=
  fi
}
trap cleanup_server EXIT INT TERM

for model_name in $MODEL_NAMES; do
  model_dir=${MODEL_DIRS[$model_name]:-}
  export_dir=${EXPORT_DIRS[$model_name]:-}
  if [[ ! -f "$model_dir/config.json" ]]; then
    echo "Missing model config: $model_dir/config.json" >&2
    exit 1
  fi

  for chunk_size in $CHUNK_SIZES; do
    for backend in $BACKENDS; do
    if [[ "$backend" != deep_gemm && "$backend" != sparse_gemm ]]; then
      echo "Unsupported backend: $backend" >&2
      exit 2
    fi
    if [[ "$backend" == sparse_gemm && ! -f "$export_dir/manifest.json" ]]; then
      echo "Missing SparseGEMM manifest: $export_dir/manifest.json" >&2
      exit 1
    fi

      run_name=${model_name}-${backend}-chunk${chunk_size}
      server_log=$RESULT_ROOT/logs/${run_name}.server.log
      sparse_layout=auto
      if (( chunk_size >= 16384 )); then
        sparse_layout=masked
      fi
      server_args=(
        -m sglang.launch_server
        --model-path "$model_dir"
        --host 127.0.0.1
        --port "$SERVER_PORT"
        --tp-size 2
        --ep-size 1
        --moe-a2a-backend none
        --moe-runner-backend "$backend"
        --attention-backend triton
        --trust-remote-code
        --cuda-graph-bs-decode $CUDA_GRAPH_BS_DECODE
        --disable-prefill-cuda-graph
        --context-length 8192
        --max-running-requests 128
        --chunked-prefill-size "$chunk_size"
        --max-prefill-tokens "$chunk_size"
        --mem-fraction-static "$MEM_FRACTION_STATIC"
        --random-seed "$SEED"
      )

      echo "[$(date '+%F %T')] starting $run_name"
      cleanup_server
      if [[ "$backend" == sparse_gemm ]]; then
        env -u SGLANG_DEEPEP_BF16_DISPATCH \
          SGLANG_SPARSE_GEMM_MOE_PATH="$export_dir" \
          SGLANG_SPARSE_GEMM_SHARED_MIN_M=0 \
          SGLANG_SPARSE_GEMM_KERNEL=wgmma_tma \
          SGLANG_SPARSE_GEMM_LAYOUT="$sparse_layout" \
          SGLANG_SPARSE_GEMM_CONTIGUOUS_MIN_M=4096 \
          SGLANG_SPARSE_GEMM_M_ALIGNMENT=64 \
          SGLANG_SPARSE_GEMM_MASKED_M_ALIGNMENT=64 \
          SPARSE_GEMM_ACTIVE_EXPERT_PREBIND=1 \
          SGLANG_MOE_PREFILL_DUAL_STREAM=1 \
          PYTHONPATH="$SGLANG_DIR/python:$SPARSE_GEMM_DIR" \
          CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
          setsid "$PYTHON_BIN" "${server_args[@]}" >"$server_log" 2>&1 &
      else
        env -u SGLANG_SPARSE_GEMM_MOE_PATH \
          -u SGLANG_SPARSE_GEMM_SHARED_MIN_M \
          -u SGLANG_SPARSE_GEMM_KERNEL \
          -u SGLANG_SPARSE_GEMM_LAYOUT \
          -u SGLANG_SPARSE_GEMM_CONTIGUOUS_MIN_M \
          -u SGLANG_SPARSE_GEMM_M_ALIGNMENT \
          -u SGLANG_SPARSE_GEMM_MASKED_M_ALIGNMENT \
          -u SPARSE_GEMM_ACTIVE_EXPERT_PREBIND \
          SGLANG_MOE_PREFILL_DUAL_STREAM=1 \
          PYTHONPATH="$SGLANG_DIR/python:$SPARSE_GEMM_DIR" \
          CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
          setsid "$PYTHON_BIN" "${server_args[@]}" >"$server_log" 2>&1 &
      fi
      server_pid=$!

      deadline=$((SECONDS + STARTUP_TIMEOUT))
      until curl -fsS "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
          echo "Server exited while starting $run_name" >&2
          tail -n 160 "$server_log" >&2
          exit 1
        fi
        if (( SECONDS >= deadline )); then
          echo "Server startup timed out for $run_name" >&2
          tail -n 160 "$server_log" >&2
          exit 1
        fi
        sleep 2
      done

      for concurrency in $CONCURRENCIES; do
        num_prompts=$((PROMPT_WAVES * concurrency))
        for warmup in $(seq 1 "$WARMUP_REPEATS"); do
          echo "[$(date '+%F %T')] warming up $run_name concurrency=$concurrency repeat=$warmup"
          PYTHONPATH="$SGLANG_DIR/python" "$PYTHON_BIN" \
            -m sglang.benchmark.serving \
            --backend sglang \
            --host 127.0.0.1 \
            --port "$SERVER_PORT" \
            --model "$model_dir" \
            --dataset-name random-ids \
            --random-input-len "$INPUT_LEN" \
            --random-output-len "$OUTPUT_LEN" \
            --random-range-ratio 0 \
            --num-prompts "$num_prompts" \
            --request-rate inf \
            --max-concurrency "$concurrency" \
            --flush-cache \
            --seed "$SEED" \
            --temperature 0 \
            --disable-tqdm >/dev/null
        done
        for repeat in $(seq 1 "$BENCH_REPEATS"); do
          output_file=$RESULT_ROOT/results/${run_name}-c${concurrency}-r${repeat}.jsonl
          bench_log=$RESULT_ROOT/logs/${run_name}-c${concurrency}-r${repeat}.bench.log
          if [[ -s "$output_file" ]]; then
            echo "[$(date '+%F %T')] reusing $output_file"
            continue
          fi
          echo "[$(date '+%F %T')] benchmarking $run_name concurrency=$concurrency repeat=$repeat"
          PYTHONPATH="$SGLANG_DIR/python" "$PYTHON_BIN" \
            -m sglang.benchmark.serving \
            --backend sglang \
            --host 127.0.0.1 \
            --port "$SERVER_PORT" \
            --model "$model_dir" \
            --tokenizer "$model_dir" \
            --dataset-name random-ids \
            --random-input-len "$INPUT_LEN" \
            --random-output-len "$OUTPUT_LEN" \
            --random-range-ratio 0 \
            --num-prompts "$num_prompts" \
            --request-rate inf \
            --max-concurrency "$concurrency" \
            --flush-cache \
            --seed "$SEED" \
            --temperature 0 \
            --disable-tqdm \
            --output-details \
            --output-file "$output_file" >"$bench_log" 2>&1
        done
      done
      cleanup_server
    done
  done
done

echo "Benchmark completed: $RESULT_ROOT"
