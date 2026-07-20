# Project Structure

MosaicMoE is organized as an independent sparse MoE toolchain rather than a
serving-framework fork.

```text
MosaicMoE/
├── mosaic_moe/
│   ├── patterns/        # Sparse pattern semantics and validators
│   ├── pruning/         # Pruning frontends and HF checkpoint export
│   ├── weight_convert/  # Compatibility exports for SparseGEMM conversion
│   ├── core/            # Framework-independent runtime abstractions
│   ├── kernels/         # Wrappers around third-party kernel backends
│   └── search/          # Pattern and kernel parameter search
├── evaluation/          # Accuracy evaluation pipeline
├── benchmarks/          # Kernel, layer, and serving benchmarks
├── end2end/
│   └── sglang/          # SGLang integration experiments
├── third_party/
│   ├── SparseGEMM/      # Format, conversion, reference, and CUDA kernels
│   └── sglang/          # Optional external SGLang checkout placeholder
├── patches/sglang/      # Temporary SGLang patches
├── scripts/             # Utility entry points
└── docs/                # Design and experiment notes
```

The intended data flow is:

```text
pattern definition
  -> pruning and zero-weight HF checkpoint
  -> SparseGEMM metadata and packed weight conversion
  -> SparseGEMM reference or CUDA kernel
  -> end-to-end SGLang integration
```

`third_party/SparseGEMM` is the single owner of the kernel-facing hybrid sparse
format. MosaicMoE owns pruning policy and evaluation, and imports conversion
and computation APIs from the lightweight `sparse_gemm` Python package.
