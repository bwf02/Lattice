# LATTICE: Sparse MoE Inference on GPUs

LATTICE is a sparse Mixture-of-Experts (MoE) inference system that co-designs
hierarchical weight sparsity and expert-aware grouped GEMM on NVIDIA Hopper GPUs.

[Overview](#overview) · [Repositories](#repositories) · [Setup](#setup) ·
[Accuracy](#model-accuracy) · [Kernels](#kernel-benchmarks) ·
[Serving](#end-to-end-serving) · [Validation](#validation)

## Overview

Structured weight sparsity reduces the computation and memory footprint of MoE
expert layers, but low sparsity offers limited benefits unless sparse-format and
runtime scheduling overheads are controlled. LATTICE introduces the Hierarchical
Block N:M (HiBNM) format, which combines dense blocks and hardware-supported 2:4
sparse blocks, together with Max-Row Sum-of-Squares (MRSS) pruning to protect
sensitive weight rows. Its sparse grouped-GEMM backend incorporates hierarchical
tiling, asynchronous pipelining, and an Active-Expert Prebind (AEP) Scheduler.
Integration with SGLang enables end-to-end prefill and decoding.

This repository provides pruning, accuracy evaluation, checkpoint export, and
experiment drivers. Kernel implementations and serving integration are maintained
in two companion repositories.

## Repositories

| Repository | Contents |
|---|---|
| [Lattice](https://github.com/bwf02/Lattice) | Pruning, accuracy evaluation, checkpoint export, experiment drivers |
| [SparseGEMM](https://github.com/bwf02/SparseGEMM) | HiBNM packing, sparse kernels, kernel benchmarks |
| [sglang](https://github.com/bwf02/sglang) | SGLang fork with LATTICE and baseline MoE backends |

Use the `main` branches and keep the repositories in sibling directories:

```bash
mkdir lattice-workspace
cd lattice-workspace
git clone --branch main https://github.com/bwf02/Lattice.git
git clone --branch main https://github.com/bwf02/SparseGEMM.git
git clone --branch main https://github.com/bwf02/sglang.git

# Companion revisions used by these instructions.
git -C SparseGEMM checkout 3b27d079d21e67732bce600776193c182d7de689
git -C sglang checkout b4150ea6d49c415a535baad0495acbf91901e980
git -C SparseGEMM submodule update --init --recursive
cd Lattice
```

The internal Python package retains the name `mosaic_moe` for compatibility.
Model weights and datasets are obtained separately under their respective
licenses.

## Setup

### Hardware and software

| Component | Requirement |
|---|---|
| Platform | Linux; Bash 4 or newer for serving scripts |
| Sparse kernels | NVIDIA Hopper GPU with SM90a support, e.g. H20/H100/H800 |
| Accuracy evaluation | GPU memory sufficient for one model replica per evaluation worker |
| Serving | Multiple GPUs for the selected tensor-parallel configuration |
| Python | Python 3.11 recommended |
| Build tools | Git, a C++ compiler, CUDA toolkit with `nvcc`, and CUDA-enabled PyTorch |

Use separate environments for accuracy and serving. The pinned SGLang fork
declares PyTorch 2.11.0 and CUDA-13-related dependencies; its driver/toolkit
requirements must be satisfied independently of the accuracy environment.
See its [dependency file](https://github.com/bwf02/sglang/blob/b4150ea6d49c415a535baad0495acbf91901e980/python/pyproject.toml)
before selecting a GPU container.

### Accuracy environment

From `Lattice/`:

```bash
python3.11 -m venv .venv-accuracy
source .venv-accuracy/bin/activate

# Install a CUDA-enabled PyTorch build appropriate for the host first.
python -m pip install -r evaluation/requirements-accuracy.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()'
```

### Kernel and serving environment

In a compatible Linux/CUDA environment, from `Lattice/`:

```bash
python3.11 -m venv .venv-serving
source .venv-serving/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install -e ../sglang/python

# Build the SparseGEMM fork rather than installing an upstream DeepGEMM wheel.
DG_FORCE_BUILD=1 python -m pip install --no-build-isolation --no-deps -e ../SparseGEMM
python -m pip install modelscope safetensors
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import torch, deep_gemm, sparse_gemm; print(deep_gemm.__file__); assert torch.cuda.is_available()'
```

Keep the CUDA toolkit and submodule headers available for runtime JIT
compilation. Additional baseline build instructions are provided in
[SparseGEMM/baselines/moe_batch](https://github.com/bwf02/SparseGEMM/tree/main/baselines/moe_batch).

## Model accuracy

Activate the accuracy environment. The following example prunes the routed
experts of a local Qwen3 checkpoint and saves the result:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/run_sparse_accuracy.py \
  --model /models/Qwen3-30B-A3B \
  --label hibnm25 --sparsity-type hibnm --sparsity-ratio 0.25 \
  --prune-method wanda --hybrid-block-score max_row_squared \
  --block-h 64 --block-w 64 --block-n 1 --block-m 2 \
  --nsamples 32 --calibration-length 2048 --seed 0 \
  --prune-only --save-model results/checkpoints/hibnm25 \
  --output results/accuracy/hibnm25-pruning.json
```

Evaluate the saved checkpoint using one replica per GPU:

```bash
python evaluation/run_accuracy_multigpu.py \
  --gpus 0,1,2,3 --output-dir results/accuracy/hibnm25 \
  -- --model results/checkpoints/hibnm25 --skip-pruning --label hibnm25 \
  --tasks gsm8k math500 mmlu asdiv --batch-size 1 --seed 0

python evaluation/summarize_accuracy.py \
  results/accuracy/hibnm25/merged.json \
  --output results/accuracy/summary.csv
```

Add `--limit 8` to the evaluation arguments for an initial smoke test.
Supported patterns include `dense`, `hibnm`, `unstructured`, `2:4`, and `2:8`.
See [Accuracy reproduction](evaluation/ACCURACY_REPRODUCTION.md) for calibration
data, baseline commands, single-GPU execution, local dataset configuration,
checkpoint verification, and MMLU-only summaries.

## Kernel benchmarks

Activate the serving environment and run from `SparseGEMM/`:

```bash
python benchmarks/bench_moe_model_shapes.py \
  --models qwen15 --projections gate_up down \
  --batch-sizes 128 256 512 1024 2048 4096 \
  --native-only --warmup 20 --iterations 100 \
  --output results/kernel-qwen15.csv
```

Use `--help` for the available models, layouts, and baseline options.
External baselines require the build described in the
[baseline README](https://github.com/bwf02/SparseGEMM/blob/main/baselines/moe_batch/README.md).

## End-to-end serving

Activate the serving environment and return to `Lattice/`.
First export packed weights from the pruned checkpoint:

```bash
python scripts/export_qwen15_moe_sparse_gemm.py \
  --model-dir results/checkpoints/hibnm25 --no-download \
  --output-dir results/packed/qwen3-hibnm25 \
  --mask-source zeros --block-h 64 --block-w 64 --block-n 1 --block-m 2
```

The exporter supports multiple MoE model families despite its historical
filename. Run prefill and decoding with the same request and TP settings:

```bash
MODEL_NAMES=qwen3 BACKENDS="deep_gemm sparse_gemm" PROFILES="prefill decode" \
PYTHON_BIN="$PWD/.venv-serving/bin/python" \
QWEN3_MODEL_DIR=/models/Qwen3-30B-A3B \
QWEN3_EXPORT_DIR="$PWD/results/packed/qwen3-hibnm25" \
QWEN3_TP_SIZE=4 QWEN3_GPU_SET=0,1,2,3 \
RESULT_ROOT="$PWD/results/serving-qwen3" \
SGLANG_REVISION=$(git -C ../sglang rev-parse HEAD) \
SPARSE_GEMM_REVISION=$(git -C ../SparseGEMM rev-parse HEAD) \
MOSAIC_MOE_REVISION=$(git rev-parse HEAD) \
bash end2end/sglang/benchmark_paper_e2e.sh

python end2end/sglang/summarize_paper_e2e.py results/serving-qwen3
```

Other model identifiers are `qwen15`, `deepseek_v2_lite`, and `llama4_scout`.
Set the corresponding model/export directories and GPU allocation before
running. Keep result directories separate for each run.

## Validation

```bash
python -m pytest evaluation/tests -q
python -m pytest end2end/sglang/test_summarize_paper_e2e.py -q
```

The reconstructed accuracy workflow has CPU tests covering masks, tiny Qwen3
models, checkpoint export, harness integration, and shard merging. Full-model
GPU accuracy validation and an end-to-end clean-environment reproduction remain
pending. The calibration and prompting protocol is documented in the accuracy
guide; historical paper measurements are not regenerated by these tests.

## Repository layout

```text
evaluation/        Pruning, accuracy runners, summaries, and tests
mosaic_moe/        Pattern definitions, checkpoint export, runtime adapters
scripts/           Checkpoint export utilities
end2end/sglang/    Serving experiment drivers and summaries
docs/              Design notes
```

## License and dependencies

A top-level project license has not yet been selected. Bundled third-party
components retain their respective licenses and notices. The implementation
builds on DeepGEMM, SGLang, Wanda, SparseGPT, and NVIDIA CUDA libraries.
