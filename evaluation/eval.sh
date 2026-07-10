#!/usr/bin/env bash

set -euo pipefail

# Stage 0: Fixed evaluation configuration.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="/tmp/mosaicmoe-eval-venv"
MODEL_ID="Qwen/Qwen1.5-MoE-A2.7B"
MODEL_DIR="${ROOT_DIR}/models/Qwen1.5-MoE-A2.7B"
OUTPUT_DIR="${ROOT_DIR}/results/Qwen1.5-MoE-A2.7B"
MODE="${1:-smoke}"

usage() {
  cat <<'EOF'
Usage: ./evaluation/eval.sh [smoke|full]
EOF
}

case "${MODE}" in
  smoke|full)
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

# Stage 1: Create an isolated environment under /tmp and install dependencies.
echo "[1/3] Preparing evaluation environment: ${VENV_DIR}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/requirements.txt"

# Stage 2: Download the model once and reuse the local checkpoint.
echo "[2/3] Preparing model: ${MODEL_ID}"
if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  mkdir -p "${MODEL_DIR}"
  python -c \
    "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_ID}', local_dir='${MODEL_DIR}')"
fi

# Stage 3: Run a short smoke test or the complete benchmark suite.
echo "[3/3] Running ${MODE} evaluation"
mkdir -p "${OUTPUT_DIR}"

if [[ "${MODE}" == "full" ]]; then
  "${ROOT_DIR}/evaluation/run_eval.sh" \
    "${MODEL_DIR}" "${OUTPUT_DIR}" 0 1
else
  lm_eval --model hf \
    --model_args "pretrained=${MODEL_DIR},dtype=float16,trust_remote_code=True" \
    --tasks piqa \
    --limit 16 \
    --batch_size 1 \
    --output_path "${OUTPUT_DIR}/smoke"
fi
