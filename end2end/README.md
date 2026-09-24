# End-to-End Integration

This directory contains framework integration experiments that connect LATTICE
artifacts to serving systems.

- `sglang/`: SGLang integration work, including loader hooks, runtime adapters,
  and end-to-end serving benchmarks.

Serving frameworks remain external third-party projects. LATTICE keeps only
the adapter code, patches, and experiment entry points needed to reproduce the
integration.

Use `sglang/validate_sparse_gemm_model.sh MODEL_DIR EXPORT_DIR` for one-model
validation. It exports or resumes the packed checkpoint, launches SGLang with
the SparseGEMM MoE backend, and verifies health, model discovery, and a
deterministic generation request. The script never downloads or deletes model
weights; exports and logs should be placed under `/tmp` on remote GPU hosts.
