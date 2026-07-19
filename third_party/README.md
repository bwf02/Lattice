# Third-Party Dependencies

Third-party projects are kept outside the MosaicMoE Python package.

- `SparseGEMM/`: custom sparse GEMM kernel backend.
- `sglang/`: reserved for an external SGLang checkout or submodule if needed.

MosaicMoE should depend on these projects through narrow wrappers under
`mosaic_moe/kernels` and `end2end/sglang`.

