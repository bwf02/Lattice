# MosaicMoE

MosaicMoE is being refactored toward a single-GPU sparse MoE kernel design that focuses on block-structured / HB-N:M sparse expert computation. The revised direction removes the old padding-free motivation and emphasizes sparse format + kernel co-design for routed MoE workloads.

## Revision Focus

- [ ] Reposition HB-N:M as a sparse format for MoE routed token layouts, not a general sparse format.
- [ ] Focus on single-GPU grouped GEMM based sparse expert computation.
- [ ] Remove the assumption that modern MoE systems materialize padded activation buffers.
- [ ] Optimize sparse weight and routed activation paths separately with hybrid warp specialization.
- [ ] Validate accuracy on Qwen 16B / 30B before committing to kernel-level acceleration.

## Code Roadmap

### 0. Project Structure

- [ ] Keep sparse pattern definitions under `mosaic_moe/patterns`.
- [ ] Keep pruning and accuracy evaluation under `evaluation` until the pipeline is stable.
- [x] Add packed weight and metadata conversion under `third_party/SparseGEMM`.
- [ ] Use `third_party/SparseGEMM` as the external sparse kernel backend.
- [ ] Keep `third_party/sglang` as an optional external SGLang checkout placeholder.
- [ ] Put serving-framework integration work under `end2end/sglang`.
- [ ] Keep temporary SGLang patches under `patches/sglang`.

### 1. Pruning

- [ ] Build MosaicMoE pruning and accuracy scripts under `evaluation`.
- [ ] Add routed expert layer filtering, excluding attention, router, and shared expert layers.
- [ ] Decouple Wanda / SparseGPT importance computation from mask generation.
- [ ] Implement dense, 2:4, 4:6, 6:8, V:N:M, and HB-N:M masks.

### 2. Evaluation

- [ ] Fix calibration set, random seed, sample count, and sequence length.
- [ ] Add WikiText2 perplexity and `lm-eval` scripts.
- [ ] Record actual sparsity, per-layer sparsity, and task accuracy automatically.
- [ ] Run the full pipeline on the smallest model first before scaling to larger MoE models.
- [ ] Evaluate Qwen1.5-MoE-A2.7B, DeepSeek-V2-Lite, Qwen3-30B-A3B-Instruct-2507, and Mixtral-8x7B.

### 3. Checkpoint

- [ ] Export standard Hugging Face checkpoints with zeroed sparse weights.
- [ ] Implement an HB-N:M format validator.
- [x] Implement a SparseGEMM packer for nonzero weights and metadata.
- [x] Add pack / unpack numerical consistency tests.

### 4. Kernel

- [x] Implement dense, contiguous-grouped, and masked-grouped Torch references through `third_party/SparseGEMM`.
- [ ] Implement index-driven routed activation loading.
- [ ] Implement gate-up, activation, and down computation stages.
- [ ] Add hybrid warp specialization, TMA / `cp.async` pipelines, and autotuning.
- [ ] Benchmark correctness, token distribution sensitivity, and expert load imbalance.

### 5. End-to-End Integration

- [ ] Define an independent MosaicMoE runtime API.
- [ ] Integrate with the SGLang fused MoE backend under `end2end/sglang`.
- [ ] Implement a packed checkpoint loader and backend selection option.
- [ ] Compare against native Triton, DeepGEMM, CUTLASS, FlashInfer, and SGLang MoE backends.
