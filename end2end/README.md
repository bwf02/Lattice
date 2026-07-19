# End-to-End Integration

This directory contains framework integration experiments that connect MosaicMoE
artifacts to serving systems.

- `sglang/`: SGLang integration work, including loader hooks, runtime adapters,
  and end-to-end serving benchmarks.

Serving frameworks remain external third-party projects. MosaicMoE keeps only
the adapter code, patches, and experiment entry points needed to reproduce the
integration.

