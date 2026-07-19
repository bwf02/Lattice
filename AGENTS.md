# Global Codex Rules

## Git Operations

- `git add`, `git commit`, and `git push` require explicit human approval before running.
- Commit messages must use Conventional Commits, for example:
  - `fix: ...`
  - `feat: ...`
  - `docs: ...`
  - `test: ...`
- Commit logs must be written in English.
- Commit bodies should briefly explain why the change was made and what validation or tests were run, using one or two natural-language sentences without required labels.

## Repository Setup

```bash
git clone https://github.com/bwf02/MosaicMoE.git
cd MosaicMoE
git checkout mosaic_moe_dev_eval
```

## Project Structure

MosaicMoE is organized as an independent sparse MoE toolchain, not a fork of a
serving framework.

```text
MosaicMoE/
├── mosaic_moe/
│   ├── patterns/        # Sparse pattern semantics and validators
│   ├── pruning/         # Pruning frontends and HF checkpoint export
│   ├── weight_convert/  # Sparse checkpoint to packed kernel artifacts
│   ├── core/            # Framework-independent runtime abstractions
│   ├── kernels/         # Thin wrappers around third-party kernel backends
│   └── search/          # Pattern and kernel parameter search
├── evaluation/          # Accuracy evaluation pipeline
├── benchmarks/          # Kernel, layer, and serving benchmarks
├── end2end/
│   └── sglang/          # SGLang end-to-end integration experiments
├── third_party/
│   ├── SparseGEMM/      # External sparse GEMM kernel repository
│   └── sglang/          # Optional external SGLang checkout placeholder
├── patches/sglang/      # Temporary SGLang patches
├── scripts/             # Utility entry points
└── docs/                # Design and experiment notes
```

Do not place third-party source code directly under `mosaic_moe`. Kernel
implementations live in `third_party/SparseGEMM`; `mosaic_moe/kernels` should
only contain small Python wrappers, build/load helpers, dispatch code, and
numerical correctness adapters used by MosaicMoE.

The intended data flow is:

```text
pattern definition
  -> pruning and zero-weight HF checkpoint
  -> metadata and packed weight conversion
  -> SparseGEMM kernel wrapper
  -> end-to-end SGLang integration
```

## Remote Validation Workflow

Use this sequence for implementation and validation:

1. Modify and run static checks in the local repository.
2. Review the local diff, then commit and push after explicit approval.
3. Connect to the current SSH port provided by the user; the port may change.
4. Update `/ossfs/workspace/MosaicMoE` with `git pull --ff-only`.
5. Run unit tests and model smoke tests in the remote GPU environment.

```bash
ssh -p <current-port> 127.0.0.1
cd /ossfs/workspace/MosaicMoE
git pull --ff-only origin mosaic_moe_dev_eval
```

## Evaluation Usage

Use two evaluation levels for pruning experiments:

- Quick validation: WikiText2 perplexity and MMLU 5-shot only.
- Full evaluation: run the complete benchmark suite, including commonsense,
  MMLU 5-shot, and GSM8K 5-shot.

Use quick validation for routine pruning iterations. Run the full evaluation
only for selected configurations or when the user explicitly requests `full`.

Install the evaluation dependencies from the repository root:

```bash
uv venv /tmp/mosaicmoe-eval-venv \
  --clear --system-site-packages --python /usr/bin/python3
source /tmp/mosaicmoe-eval-venv/bin/activate
pip install -r requirements.txt
```

Run the complete setup, model download, and smoke evaluation pipeline from the
repository root:

```bash
./evaluation/eval.sh smoke
```

Run the full benchmark suite with the same setup:

```bash
./evaluation/eval.sh full
```

Run Wanda from its own directory. In this project, `N:M` denotes the number of
nonzero weights kept, while `--prune_n` denotes the number of weights pruned.

```bash
cd evaluation/wanda
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model Qwen/Qwen2.5-14B \
  --prune_method wanda \
  --sparsity_type 6:8 \
  --prune_n 2 --prune_m 8 \
  --save /path/to/results/wanda_6_8 \
  --save_model /path/to/checkpoints/wanda_6_8
```

Run HB-N:M Wanda pruning on Qwen1.5-MoE routed experts. `block_n:block_m`
controls inter-block retention, while each retained block uses intra-block 2:4:

```bash
cd evaluation/wanda
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --prune_method wanda \
  --sparsity_type hb_nm \
  --block_h 16 --block_w 16 \
  --block_n 1 --block_m 2 \
  --nsamples 16 \
  --save /path/to/results/hb_nm \
  --save_model /path/to/checkpoints/hb_nm
```

Run the HB-N:M unit tests from the repository root:

```bash
python -m unittest discover -s evaluation/tests -v
```

Run SparseGPT from its own directory:

```bash
cd evaluation/sparsegpt
CUDA_VISIBLE_DEVICES=0 python llama.py Qwen/Qwen2.5-7B wikitext2 \
  --prunen 2 --prunem 8 \
  --save /path/to/checkpoints/sparsegpt_6_8
```

Run the benchmark suite from the repository root after producing a checkpoint:

```bash
./evaluation/run_eval.sh /path/to/checkpoint /path/to/results 0 auto
```
