# Project Structure

MosaicMoE is organized as an independent sparse MoE toolchain rather than a
serving-framework fork.

```text
MosaicMoE/
├── mosaic_moe/
│   ├── patterns/        # Sparse pattern semantics and validators
│   ├── pruning/         # Pruning frontends and HF checkpoint export
│   ├── weight_convert/  # Sparse checkpoint to packed kernel artifacts
│   ├── core/            # Framework-independent runtime abstractions
│   ├── kernels/         # Wrappers around third-party kernel backends
│   └── search/          # Pattern and kernel parameter search
├── evaluation/          # Accuracy evaluation pipeline
├── benchmarks/          # Kernel, layer, and serving benchmarks
├── end2end/
│   └── sglang/          # SGLang integration experiments
├── third_party/
│   ├── SparseGEMM/      # Sparse kernel backend
│   └── sglang/          # Optional external SGLang checkout placeholder
├── patches/sglang/      # Temporary SGLang patches
├── scripts/             # Utility entry points
└── docs/                # Design and experiment notes
```

The intended data flow is:

```text
pattern definition
  -> pruning and zero-weight HF checkpoint
  -> metadata and packed weight conversion
  -> SparseGEMM kernel wrapper
  -> end-to-end SGLang integration
```

