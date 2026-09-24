# Third-Party Dependencies

Third-party projects are kept outside the LATTICE Python package.

- `SparseGEMM/`: custom sparse GEMM kernel backend.
- `sglang/`: reserved for an external SGLang checkout or submodule if needed.

LATTICE should depend on these projects through narrow wrappers under
`lattice/kernels` and `end2end/sglang`.
