#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 MODEL_DIR EXPORT_DIR" >&2
  exit 2
fi

MODEL_DIR=$(cd "$1" && pwd)
EXPORT_DIR=$2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MOSAIC_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
WORKSPACE_DIR=$(cd "$MOSAIC_DIR/.." && pwd)
SGLANG_DIR=${SGLANG_DIR:-$WORKSPACE_DIR/sglang}
SPARSE_GEMM_DIR=${SPARSE_GEMM_DIR:-$WORKSPACE_DIR/SparseGEMM}
PYTHON_BIN=${PYTHON_BIN:-/tmp/sglang-venv/bin/python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
TP_SIZE=${TP_SIZE:-2}
SERVER_PORT=${SERVER_PORT:-32000}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-2048}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-4}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-900}
NUM_WORKERS=${NUM_WORKERS:-1}
INCLUDE_SHARED_EXPERT=${INCLUDE_SHARED_EXPERT:-auto}
LOG_FILE=${LOG_FILE:-/tmp/sglang-sparse-smoke-${SERVER_PORT}.log}

for path in "$SGLANG_DIR" "$SPARSE_GEMM_DIR"; do
  if [[ ! -d "$path" ]]; then
    echo "Required repository does not exist: $path" >&2
    exit 1
  fi
done
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment does not exist: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="$SGLANG_DIR/python:$SPARSE_GEMM_DIR:$MOSAIC_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$INCLUDE_SHARED_EXPERT" == auto ]]; then
  model_type=$(
    "$PYTHON_BIN" - "$MODEL_DIR/config.json" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1]))["model_type"])
PY
  )
  if [[ "$model_type" == qwen2_moe || "$model_type" == deepseek_v2 || \
        "$model_type" == ernie4_5_moe || "$model_type" == llama4 ]]; then
    INCLUDE_SHARED_EXPERT=1
  else
    INCLUDE_SHARED_EXPERT=0
  fi
fi
if [[ "$INCLUDE_SHARED_EXPERT" != 0 && "$INCLUDE_SHARED_EXPERT" != 1 ]]; then
  echo "INCLUDE_SHARED_EXPERT must be auto, 0, or 1" >&2
  exit 2
fi

export_args=(
  "$MOSAIC_DIR/scripts/export_moe_sparse_gemm.py"
  --model-dir "$MODEL_DIR"
  --output-dir "$EXPORT_DIR"
  --no-download
  --num-workers "$NUM_WORKERS"
  --skip-existing
)
if [[ "$INCLUDE_SHARED_EXPERT" == 1 ]]; then
  export_args+=(--include-shared-expert)
fi
"$PYTHON_BIN" "${export_args[@]}"

export SGLANG_SPARSE_GEMM_MOE_PATH="$EXPORT_DIR"
export SGLANG_SPARSE_GEMM_KERNEL=${SGLANG_SPARSE_GEMM_KERNEL:-wgmma_tma}
export SGLANG_SPARSE_GEMM_LAYOUT=${SGLANG_SPARSE_GEMM_LAYOUT:-auto}
export SGLANG_SPARSE_GEMM_CONTIGUOUS_MIN_M=${SGLANG_SPARSE_GEMM_CONTIGUOUS_MIN_M:-4096}
export SGLANG_SPARSE_GEMM_M_ALIGNMENT=${SGLANG_SPARSE_GEMM_M_ALIGNMENT:-64}
export SGLANG_SPARSE_GEMM_MASKED_M_ALIGNMENT=${SGLANG_SPARSE_GEMM_MASKED_M_ALIGNMENT:-64}

server_args=(
  -m sglang.launch_server
  --model-path "$MODEL_DIR"
  --host 127.0.0.1
  --port "$SERVER_PORT"
  --tp-size "$TP_SIZE"
  --ep-size 1
  --moe-a2a-backend none
  --moe-runner-backend sparse_gemm
  --attention-backend triton
  --trust-remote-code
  --disable-cuda-graph
  --context-length "$CONTEXT_LENGTH"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --chunked-prefill-size "$CONTEXT_LENGTH"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
)
if [[ ${CPU_OFFLOAD_GB:-0} != 0 ]]; then
  server_args+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  "$PYTHON_BIN" "${server_args[@]}" >"$LOG_FILE" 2>&1 &
server_pid=$!

cleanup() {
  kill "$server_pid" 2>/dev/null || true
  pkill -TERM -P "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl -fsS "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "SGLang exited before becoming healthy. Log: $LOG_FILE" >&2
    tail -n 120 "$LOG_FILE" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "SGLang startup timed out after ${STARTUP_TIMEOUT}s. Log: $LOG_FILE" >&2
    tail -n 120 "$LOG_FILE" >&2
    exit 1
  fi
  sleep 2
done

curl -fsS "http://127.0.0.1:${SERVER_PORT}/v1/models" >"/tmp/sglang-sparse-models-${SERVER_PORT}.json"
curl -fsS "http://127.0.0.1:${SERVER_PORT}/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":8}}' \
  >"/tmp/sglang-sparse-generate-${SERVER_PORT}.json"

"$PYTHON_BIN" - "$SERVER_PORT" <<'PY'
import json
import sys
from pathlib import Path

port = sys.argv[1]
models = json.loads(Path(f"/tmp/sglang-sparse-models-{port}.json").read_text())
generation = json.loads(Path(f"/tmp/sglang-sparse-generate-{port}.json").read_text())
if not models.get("data"):
    raise SystemExit("/v1/models returned no models")
if not generation.get("text"):
    raise SystemExit("/generate returned no text")
print(json.dumps({"model": models["data"][0]["id"], "text": generation["text"]}))
PY
