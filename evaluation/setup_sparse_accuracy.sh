#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/ossfs/workspace}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen1.5-MoE-A2.7B}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE_ROOT}/MosaicMoE/models/Qwen1.5-MoE-A2.7B}"
VENV_DIR="${VENV_DIR:-/tmp/mosaicmoe-accuracy-venv}"
DATASET_ROOT="${DATASET_ROOT:-/tmp/mosaicmoe-eval-datasets}"
UV_BOOTSTRAP_DIR="${UV_BOOTSTRAP_DIR:-/tmp/uv-bootstrap}"
UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.antfin-inc.com/simple/}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-0}"

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif [[ -x "$UV_BOOTSTRAP_DIR/bin/uv" ]]; then
  UV_BIN="$UV_BOOTSTRAP_DIR/bin/uv"
else
  python3 -m venv "$UV_BOOTSTRAP_DIR"
  "$UV_BOOTSTRAP_DIR/bin/python" -m pip install --index-url "$UV_INDEX_URL" uv
  UV_BIN="$UV_BOOTSTRAP_DIR/bin/uv"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$UV_BIN" venv "$VENV_DIR" --system-site-packages --python /usr/bin/python3
fi

# The inherited NVIDIA torch build has a prerelease version. Install packages
# that declare a torch dependency without dependency resolution so uv does not
# replace the working CUDA stack with a second PyTorch distribution.
UV_INDEX_URL="$UV_INDEX_URL" "$UV_BIN" pip install --python "$VENV_DIR/bin/python" \
  --no-deps 'lm-eval==0.4.11' 'modelscope==1.39.1' 'accelerate==1.14.0'
UV_INDEX_URL="$UV_INDEX_URL" "$UV_BIN" pip install --python "$VENV_DIR/bin/python" \
  'numpy==1.26.4' \
  'transformers==4.48.3' \
  'datasets==3.6.0' \
  'modelscope-hub==0.2.0' \
  'evaluate==0.4.6' \
  'jsonlines==4.0.0' \
  'pytablewriter==1.2.1' \
  'rouge-score==0.1.2' \
  'sacrebleu==2.6.0' \
  'scikit-learn==1.7.2' \
  'sqlitedict==2.1.0' \
  'zstandard==0.25.0' \
  'word2number==1.1' \
  'more-itertools==10.8.0' \
  'sentencepiece==0.2.1'
UV_INDEX_URL="$UV_INDEX_URL" "$UV_BIN" pip install --python "$VENV_DIR/bin/python" \
  'math-verify==0.9.0' \
  'antlr4-python3-runtime==4.11.1'

if [[ ! -s "$MODEL_DIR/config.json" ]] || ! compgen -G "$MODEL_DIR/*.safetensors" >/dev/null; then
  if [[ "$DOWNLOAD_MODEL" != "1" ]]; then
    echo "Model is missing at $MODEL_DIR; rerun with DOWNLOAD_MODEL=1" >&2
    exit 1
  fi
  MODEL_ID="$MODEL_ID" MODEL_DIR="$MODEL_DIR" "$VENV_DIR/bin/python" - <<'PY'
import os
from modelscope import snapshot_download

snapshot_download(os.environ["MODEL_ID"], local_dir=os.environ["MODEL_DIR"], max_workers=8)
PY
fi

DATASET_ROOT="$DATASET_ROOT" "$VENV_DIR/bin/python" - <<'PY'
import os
from pathlib import Path

from modelscope.hub.snapshot_download import dataset_snapshot_download

root = Path(os.environ["DATASET_ROOT"])
root.mkdir(parents=True, exist_ok=True)
datasets = [
    (
        "openai-mirror/gsm8k",
        root / "gsm8k_data",
        ["main/*", "README.md", "eval.yaml", "dataset_infos.json", ".gitattributes"],
    ),
    ("HuggingFaceH4/MATH-500", root / "math_500", None),
    ("openai-mirror/openai_humaneval", root / "humaneval", None),
    (
        "Salesforce/wikitext",
        root / "wikitext_salesforce",
        [
            "wikitext-2-raw-v1/*",
            "README.md",
            "dataset_infos.json",
            ".gitattributes",
        ],
    ),
]
for repo_id, local_dir, allow_patterns in datasets:
    dataset_snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        max_workers=8,
    )

required = {
    "GSM8K train": root / "gsm8k_data/main/train-00000-of-00001.parquet",
    "GSM8K test": root / "gsm8k_data/main/test-00000-of-00001.parquet",
    "MATH-500": root / "math_500/test.jsonl",
    "HumanEval": root / "humaneval/openai_humaneval/test-00000-of-00001.parquet",
    "WikiText-2 train": root / "wikitext_salesforce/wikitext-2-raw-v1/train-00000-of-00001.parquet",
    "WikiText-2 test": root / "wikitext_salesforce/wikitext-2-raw-v1/test-00000-of-00001.parquet",
}
for name, path in required.items():
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path}")
    print(f"{name}: {path}")
PY

"$VENV_DIR/bin/python" - <<'PY'
import datasets
import lm_eval
import torch
import transformers

assert transformers.__version__ == "4.48.3"
assert torch.cuda.is_available()
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__} datasets={datasets.__version__}")
print(f"lm_eval={lm_eval.__file__}")
PY

printf 'VENV_DIR=%s\nMODEL_DIR=%s\nDATASET_ROOT=%s\n' \
  "$VENV_DIR" "$MODEL_DIR" "$DATASET_ROOT"
