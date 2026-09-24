# LATTICE: Sparse MoE Inference on GPUs

LATTICE co-designs hierarchical weight sparsity and expert-aware grouped GEMM
for efficient Mixture-of-Experts (MoE) inference on NVIDIA Hopper GPUs.
This repository is the main entry point for pruning, model-quality evaluation,
checkpoint export, and end-to-end experiments. It was previously named MosaicMoE;
the Python package remains `mosaic_moe` for compatibility.

## Overview

Low weight sparsity can preserve MoE model quality, but converting reduced
arithmetic into GPU speedup requires controlling format and scheduling overheads.
LATTICE combines Hierarchical Block N:M (HiBNM) weights with Max-Row
Sum-of-Squares (MRSS) pruning to protect sensitive rows. A custom sparse grouped
GEMM backend uses hierarchical tiling, asynchronous pipelining, and the
Active-Expert Prebind (AEP) Scheduler. Integration with SGLang supports
evaluation of prefill and decoding.

## Repositories and checkout

| Repository | Responsibility |
|---|---|
| [Lattice](https://github.com/bwf02/Lattice/tree/paper-dev) | Pruning, accuracy, checkpoint export, experiment orchestration |
| [SparseGEMM](https://github.com/bwf02/SparseGEMM/tree/paper-dev) | Sparse formats, packing, CUDA kernels, kernel benchmarks |
| [sglang](https://github.com/bwf02/sglang/tree/paper-dev) | Serving integration and MoE backend selection |

The paper code is on **`paper-dev`**, not the older `main` branches.
Keep the repositories in sibling directories:

```bash
mkdir lattice-workspace
cd lattice-workspace
git clone --branch paper-dev https://github.com/bwf02/Lattice.git
git clone --branch paper-dev https://github.com/bwf02/SparseGEMM.git
git clone --branch paper-dev https://github.com/bwf02/sglang.git
git -C SparseGEMM checkout 3b27d079d21e67732bce600776193c182d7de689
git -C sglang checkout b4150ea6d49c415a535baad0495acbf91901e980
git -C SparseGEMM submodule update --init --recursive
cd Lattice
```

Access to this repository is required while it remains private. Record the
Lattice commit and both companion revisions with every experiment.

## Accuracy experiments

See **[Accuracy reproduction](evaluation/ACCURACY_REPRODUCTION.md)** for
environment setup, Qwen3/HiBNM/MRSS pruning, 2:4 and 2:8 baselines, multi-GPU
evaluation, MMLU and other tasks, and result summarization.

```bash
python3.11 -m venv .venv-accuracy
source .venv-accuracy/bin/activate
# Install a suitable CUDA-enabled PyTorch build on a GPU host first.
python -m pip install -r evaluation/requirements-accuracy.txt
python -m pytest evaluation/tests -q
python evaluation/run_sparse_accuracy.py --help
```

The new accuracy pipeline is reconstructed from the experiment requirements,
not recovered historical source. Its calibration and prompting policies are
explicitly documented. CPU tests are not a full-model GPU accuracy validation,
and no historical paper results are silently replaced.

## Kernel and serving experiments

These paths require Linux, Hopper GPUs, a compatible CUDA toolkit/driver,
CUDA-enabled PyTorch, and the companion repositories. Follow the pinned
[SparseGEMM build instructions](https://github.com/bwf02/SparseGEMM/blob/paper-dev/README.md)
and [SGLang dependency declarations](https://github.com/bwf02/sglang/blob/paper-dev/python/pyproject.toml).
Use separate environments for serving and accuracy; their dependency versions differ.

- Kernel benchmark: `SparseGEMM/benchmarks/bench_moe_model_shapes.py`.
- External baseline build: `SparseGEMM/baselines/moe_batch/README.md`.
- Packed checkpoint export: `scripts/export_qwen15_moe_sparse_gemm.py --help`.
  Use `--mask-source zeros` to preserve an already-pruned mask.
- End-to-end experiments: `end2end/sglang/benchmark_paper_e2e.sh`.
  Override `PYTHON_BIN`, model/export paths, TP sizes and GPU sets before running.
- Serving summaries: `end2end/sglang/summarize_paper_e2e.py`.

Do not apply archived diagnostic or ablation patches indiscriminately. The
pipeline-ablation prerequisite working tree has not been recovered; that
patch is not part of this accuracy implementation.

## Layout

```text
evaluation/        Pruning, accuracy runners, merge/summarize tools, tests
mosaic_moe/        Sparse patterns, checkpoint export, runtime adapters
scripts/           Checkpoint export utilities
end2end/sglang/    Serving experiment drivers and summaries
docs/              Design notes
```

## Licensing and acknowledgments

The repository does not yet declare a top-level project license; select one
before distributing it as an open-source release. Bundled third-party components
retain their own licenses. Model weights and datasets must be obtained separately
under their terms. LATTICE builds on DeepGEMM, SGLang, Wanda, SparseGPT, and NVIDIA
CUDA libraries; preserve their notices and cite their work.
