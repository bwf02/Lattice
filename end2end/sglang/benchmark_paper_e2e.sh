#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$(dirname "$REPO_ROOT")}
SGLANG_DIR=${SGLANG_DIR:-$WORKSPACE_ROOT/sglang}
SPARSE_GEMM_DIR=${SPARSE_GEMM_DIR:-$WORKSPACE_ROOT/SparseGEMM}
PYTHON_BIN=${PYTHON_BIN:-/tmp/sglang-venv/bin/python}
RESULT_ROOT=${RESULT_ROOT:-/tmp/losparse-paper-e2e-$(date +%Y%m%d-%H%M%S)}
MODEL_NAMES=${MODEL_NAMES:-qwen15 deepseek_v2_lite qwen3 llama4_scout}
BACKENDS=${BACKENDS:-deep_gemm sparse_gemm}
PROFILES=${PROFILES:-prefill decode mixed}
BENCH_REPEATS=${BENCH_REPEATS:-5}
WARMUP_REPEATS=${WARMUP_REPEATS:-1}
SEED=${SEED:-42}
DATASET_NAME=${DATASET_NAME:-random}
DATASET_PATH=${DATASET_PATH:-/tmp/datasets/sharegpt-modelscope/ShareGPT_V3_unfiltered_cleaned_split.json}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-1.0}
SERVER_PORT=${SERVER_PORT:-32200}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-1800}
CUDA_GRAPH_BS_DECODE=${CUDA_GRAPH_BS_DECODE:-1 2 4 8 16 32 64 128 256 512}
DECODE_CONCURRENCIES=${DECODE_CONCURRENCIES:-32 64 128 256 512}
SGLANG_REVISION=${SGLANG_REVISION:-unknown}
SPARSE_GEMM_REVISION=${SPARSE_GEMM_REVISION:-unknown}
MOSAIC_MOE_REVISION=${MOSAIC_MOE_REVISION:-unknown}
INVOCATION_ID=$(date +%Y%m%d-%H%M%S)

declare -A MODEL_DIRS=(
  [qwen15]=/ossfs/workspace/MosaicMoE/models/Qwen1.5-MoE-A2.7B
  [deepseek_v2_lite]=/tmp/models/DeepSeek-V2-Lite
  [qwen3]=/tmp/models/Qwen3-30B-A3B
  [llama4_scout]=/tmp/models/Llama-4-Scout-17B-16E-Instruct
)
declare -A EXPORT_DIRS=(
  [qwen15]=/tmp/sparse_gemm_export_nm12_shared/Qwen1.5-MoE-A2.7B
  [deepseek_v2_lite]=/tmp/sparse_gemm_exports/DeepSeek-V2-Lite
  [qwen3]=/tmp/sparse_gemm_exports/Qwen3-30B-A3B
  [llama4_scout]=/tmp/sparse_gemm_exports/Llama-4-Scout-17B-16E-Instruct
)
declare -A TP_SIZES=(
  [qwen15]=2
  [deepseek_v2_lite]=2
  [qwen3]=2
  [llama4_scout]=4
)
declare -A GPU_SETS=(
  [qwen15]=0,1
  [deepseek_v2_lite]=0,1
  [qwen3]=0,1
  [llama4_scout]=0,1,2,3
)
declare -A MEM_FRACTIONS=(
  [qwen15]=${QWEN15_MEM_FRACTION:-0.68}
  [deepseek_v2_lite]=${DEEPSEEK_V2_LITE_MEM_FRACTION:-0.68}
  [qwen3]=${QWEN3_MEM_FRACTION:-0.72}
  [llama4_scout]=${LLAMA4_SCOUT_MEM_FRACTION:-0.76}
)

for cutlass_namespace in cute cutlass; do
  include_link=$SPARSE_GEMM_DIR/deep_gemm/include/$cutlass_namespace
  include_target=$SPARSE_GEMM_DIR/third-party/cutlass/include/$cutlass_namespace
  if [[ ! -e "$include_link" ]]; then
    [[ -d "$include_target" ]] || {
      echo "Missing CUTLASS include directory: $include_target" >&2
      exit 1
    }
    ln -sfn "$include_target" "$include_link"
  fi
done

mkdir -p "$RESULT_ROOT"/{logs,raw,warmup,metadata}
nvidia-smi -q >"$RESULT_ROOT/metadata/nvidia-smi.txt"
cp "$0" "$RESULT_ROOT/metadata/benchmark_paper_e2e.sh"
PYTHONPATH="$SGLANG_DIR/python" "$PYTHON_BIN" -c \
  'import deep_gemm; print(deep_gemm.__version__, deep_gemm.__file__)' \
  >"$RESULT_ROOT/metadata/deep_gemm.txt"
printf '%s\n' \
  "models=$MODEL_NAMES" \
  "backends=$BACKENDS" \
  "profiles=$PROFILES" \
  "bench_repeats=$BENCH_REPEATS" \
  "warmup_repeats=$WARMUP_REPEATS" \
  "seed=$SEED" \
  "dataset_name=$DATASET_NAME" \
  "dataset_path=$DATASET_PATH" \
  "random_range_ratio=$RANDOM_RANGE_RATIO" \
  "pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF:-unset}" \
  "qwen15_mem_fraction=${MEM_FRACTIONS[qwen15]}" \
  "deepseek_v2_lite_mem_fraction=${MEM_FRACTIONS[deepseek_v2_lite]}" \
  "qwen3_mem_fraction=${MEM_FRACTIONS[qwen3]}" \
  "llama4_scout_mem_fraction=${MEM_FRACTIONS[llama4_scout]}" \
  "decode_cuda_graph_bs=$CUDA_GRAPH_BS_DECODE" \
  "decode_concurrencies=$DECODE_CONCURRENCIES" \
  "prefill_cuda_graph=disabled" \
  "sglang_revision=$SGLANG_REVISION" \
  "sparse_gemm_revision=$SPARSE_GEMM_REVISION" \
  "mosaic_moe_revision=$MOSAIC_MOE_REVISION" \
  >"$RESULT_ROOT/metadata/invocation-${INVOCATION_ID}.txt"

server_pid=
cleanup_server() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=
  fi
}
trap cleanup_server EXIT INT TERM

contains_word() {
  [[ " $1 " == *" $2 "* ]]
}

start_server() {
  local model_name=$1 backend=$2 profile=$3
  local model_dir=${MODEL_DIRS[$model_name]}
  local export_dir=${EXPORT_DIRS[$model_name]}
  local tp_size=${TP_SIZES[$model_name]}
  local gpu_set=${GPU_SETS[$model_name]}
  local mem_fraction=${MEM_FRACTIONS[$model_name]}
  local chunk_size=32768
  if [[ "$profile" == prefill ]]; then
    chunk_size=-1
  fi

  local run_name=${model_name}-${backend}-${profile}
  local server_log=$RESULT_ROOT/logs/${run_name}.server.log
  local -a graph_args=(--cuda-graph-bs-decode $CUDA_GRAPH_BS_DECODE)
  local -a model_args=()
  if [[ "$model_name" == qwen3 ]]; then
    model_args+=(--enforce-disable-flashinfer-allreduce-fusion)
  fi
  if [[ "$profile" == prefill ]]; then
    graph_args=(--cuda-graph-backend-decode disabled)
  fi
  local -a server_args=(
    -m sglang.launch_server
    --model-path "$model_dir"
    --host 127.0.0.1
    --port "$SERVER_PORT"
    --tp-size "$tp_size"
    --ep-size 1
    --moe-a2a-backend none
    --moe-runner-backend "$backend"
    --attention-backend triton
    --trust-remote-code
    "${model_args[@]}"
    "${graph_args[@]}"
    --disable-prefill-cuda-graph
    --context-length 8192
    --max-running-requests 512
    --chunked-prefill-size "$chunk_size"
    --max-prefill-tokens 32768
    --mem-fraction-static "$mem_fraction"
    --random-seed "$SEED"
  )

  cleanup_server
  echo "[$(date '+%F %T')] start $run_name on GPU $gpu_set"
  if [[ "$backend" == sparse_gemm ]]; then
    [[ -f "$export_dir/manifest.json" ]] || {
      echo "Missing SparseGEMM manifest: $export_dir/manifest.json" >&2
      return 1
    }
    env -u SGLANG_DEEPEP_BF16_DISPATCH \
      SGLANG_SPARSE_GEMM_MOE_PATH="$export_dir" \
      SGLANG_SPARSE_GEMM_SHARED_MIN_M=0 \
      SGLANG_SPARSE_GEMM_KERNEL=wgmma_tma \
      SGLANG_SPARSE_GEMM_LAYOUT=auto \
      SGLANG_SPARSE_GEMM_CONTIGUOUS_MIN_M=4096 \
      SGLANG_SPARSE_GEMM_M_ALIGNMENT=64 \
      SGLANG_SPARSE_GEMM_MASKED_M_ALIGNMENT=64 \
      SPARSE_GEMM_ACTIVE_EXPERT_PREBIND=1 \
      SGLANG_MOE_PREFILL_DUAL_STREAM=1 \
      PYTHONPATH="$SGLANG_DIR/python:$SPARSE_GEMM_DIR" \
      CUDA_VISIBLE_DEVICES="$gpu_set" \
      setsid "$PYTHON_BIN" "${server_args[@]}" >"$server_log" 2>&1 &
  else
    env -u SGLANG_SPARSE_GEMM_MOE_PATH \
      -u SGLANG_SPARSE_GEMM_SHARED_MIN_M \
      -u SGLANG_SPARSE_GEMM_KERNEL \
      -u SGLANG_SPARSE_GEMM_LAYOUT \
      -u SPARSE_GEMM_ACTIVE_EXPERT_PREBIND \
      SGLANG_MOE_PREFILL_DUAL_STREAM=1 \
      PYTHONPATH="$SGLANG_DIR/python" \
      CUDA_VISIBLE_DEVICES="$gpu_set" \
      setsid "$PYTHON_BIN" "${server_args[@]}" >"$server_log" 2>&1 &
  fi
  server_pid=$!

  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  until curl -fs "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Server exited while starting $run_name" >&2
      tail -n 160 "$server_log" >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Server startup timed out for $run_name" >&2
      tail -n 160 "$server_log" >&2
      return 1
    fi
    sleep 2
  done
}

run_case() {
  local model_name=$1 backend=$2 profile=$3 case_name=$4
  local input_len=$5 output_len=$6 concurrency=$7
  local model_dir=${MODEL_DIRS[$model_name]}
  local run_name=${model_name}-${backend}-${profile}-${case_name}

  for warmup in $(seq 1 "$WARMUP_REPEATS"); do
    local warmup_file=$RESULT_ROOT/warmup/${run_name}-r${warmup}.jsonl
    echo "[$(date '+%F %T')] warmup $run_name r$warmup"
    PYTHONPATH="$SGLANG_DIR/python" "$PYTHON_BIN" -m sglang.benchmark.serving \
      --backend sglang --host 127.0.0.1 --port "$SERVER_PORT" \
      --model "$model_dir" --tokenizer "$model_dir" \
      --dataset-name "$DATASET_NAME" --dataset-path "$DATASET_PATH" \
      --random-input-len "$input_len" --random-output-len "$output_len" \
      --random-range-ratio "$RANDOM_RANGE_RATIO" \
      --num-prompts "$concurrency" --request-rate inf \
      --max-concurrency "$concurrency" --flush-cache --seed "$SEED" \
      --temperature 0 --tokenize-prompt --disable-tqdm --output-details \
      --output-file "$warmup_file" >/dev/null
  done

  for repeat in $(seq 1 "$BENCH_REPEATS"); do
    local output_file=$RESULT_ROOT/raw/${run_name}-r${repeat}.jsonl
    local bench_log=$RESULT_ROOT/logs/${run_name}-r${repeat}.bench.log
    if [[ -s "$output_file" ]]; then
      echo "[$(date '+%F %T')] reuse $output_file"
      continue
    fi
    echo "[$(date '+%F %T')] measure $run_name r$repeat"
    PYTHONPATH="$SGLANG_DIR/python" "$PYTHON_BIN" -m sglang.benchmark.serving \
      --backend sglang --host 127.0.0.1 --port "$SERVER_PORT" \
      --model "$model_dir" --tokenizer "$model_dir" \
      --dataset-name "$DATASET_NAME" --dataset-path "$DATASET_PATH" \
      --random-input-len "$input_len" --random-output-len "$output_len" \
      --random-range-ratio "$RANDOM_RANGE_RATIO" \
      --num-prompts "$concurrency" --request-rate inf \
      --max-concurrency "$concurrency" --flush-cache --seed "$SEED" \
      --temperature 0 --tokenize-prompt --disable-tqdm --output-details \
      --output-file "$output_file" >"$bench_log" 2>&1
  done
}

run_profile() {
  local model_name=$1 backend=$2 profile=$3
  start_server "$model_name" "$backend" "$profile"
  case "$profile" in
    prefill)
      run_case "$model_name" "$backend" "$profile" m4096 1024 1 4
      run_case "$model_name" "$backend" "$profile" m8192 1024 1 8
      run_case "$model_name" "$backend" "$profile" m16384 1024 1 16
      run_case "$model_name" "$backend" "$profile" m32768 1024 1 32
      ;;
    decode)
      for concurrency in $DECODE_CONCURRENCIES; do
        run_case "$model_name" "$backend" "$profile" \
          "c${concurrency}" 16 256 "$concurrency"
      done
      ;;
    mixed)
      run_case "$model_name" "$backend" "$profile" c32 4096 1024 32
      run_case "$model_name" "$backend" "$profile" c64 4096 1024 64
      run_case "$model_name" "$backend" "$profile" c128 4096 1024 128
      ;;
    *)
      echo "Unknown profile: $profile" >&2
      return 2
      ;;
  esac
  cleanup_server
}

for model_name in $MODEL_NAMES; do
  [[ -f "${MODEL_DIRS[$model_name]}/config.json" ]] || {
    echo "Missing model config: ${MODEL_DIRS[$model_name]}/config.json" >&2
    exit 1
  }
  find "${MODEL_DIRS[$model_name]}" -maxdepth 1 -type f \
    \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit \
    | grep -q . || {
      echo "Missing model weight shards under ${MODEL_DIRS[$model_name]}" >&2
      exit 1
    }
  cp "${MODEL_DIRS[$model_name]}/config.json" \
    "$RESULT_ROOT/metadata/${model_name}-model-config.json"
  if contains_word "$BACKENDS" sparse_gemm; then
    [[ -f "${EXPORT_DIRS[$model_name]}/manifest.json" ]] || {
      echo "Missing SparseGEMM manifest: ${EXPORT_DIRS[$model_name]}/manifest.json" >&2
      exit 1
    }
    cp "${EXPORT_DIRS[$model_name]}/manifest.json" \
      "$RESULT_ROOT/metadata/${model_name}-sparse-manifest.json"
  fi
done

for model_name in $MODEL_NAMES; do
  for backend in $BACKENDS; do
    [[ "$backend" == deep_gemm || "$backend" == sparse_gemm ]] || {
      echo "Unsupported E2E backend: $backend" >&2
      exit 2
    }
    for profile in $PROFILES; do
      contains_word "prefill decode mixed" "$profile" || {
        echo "Unsupported profile: $profile" >&2
        exit 2
      }
      run_profile "$model_name" "$backend" "$profile"
    done
  done
done

"$PYTHON_BIN" "$REPO_ROOT/end2end/sglang/summarize_paper_e2e.py" "$RESULT_ROOT"
echo "Benchmark completed: $RESULT_ROOT"
