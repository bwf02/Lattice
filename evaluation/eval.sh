#!/usr/bin/env bash

set -euo pipefail

# Stage 0: Fixed evaluation configuration.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Inherit the host environment (torch / CUDA etc.), then add extra deps on top.
#   - `uv venv --system-site-packages`  : fast venv that reuses system torch (not reinstalled)
#   - `pip install -r requirements.txt` : add only eval extras (lm-eval, modelscope, ...)
#     (uv pip resolves very slowly against the internal PyPI mirror here, so plain pip
#      is used for the dependency step; it hits the same index and installs in seconds.)
VENV_DIR="/tmp/lattice-eval-venv"
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

# Fall back to a HuggingFace mirror for the `datasets` library (used by lm_eval to
# fetch benchmark data) when huggingface.co is unreachable. Does not override an
# explicit user setting. Model download itself uses ModelScope (Stage 2).
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Stage 1: Create a venv that inherits the host environment (torch etc.),
# then install only the extra deps from requirements.txt on top of it.
echo "[1/3] Preparing evaluation environment: ${VENV_DIR}"
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install uv
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  uv venv "${VENV_DIR}" --system-site-packages --python /usr/bin/python3
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/requirements.txt"

# Stage 2: Download the model once and reuse the local checkpoint.
# Use ModelScope instead of HuggingFace Hub so it works in environments
# where huggingface.co is unreachable.
echo "[2/3] Preparing model (via ModelScope): ${MODEL_ID}"
if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  mkdir -p "${MODEL_DIR}"
  python -c \
    "from modelscope import snapshot_download; snapshot_download(model_id='${MODEL_ID}', local_dir='${MODEL_DIR}')"
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
    --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande,boolq,openbookqa \
    --limit 16 \
    --batch_size 1 \
    --output_path "${OUTPUT_DIR}/smoke_commonsense"
fi
