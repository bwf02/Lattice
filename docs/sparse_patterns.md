# Sparse Patterns

This document briefly defines the sparse patterns currently implemented in
MosaicMoE evaluation.

## HB-N:M

`hb_nm` uses a two-level sparse mask on routed MoE expert weights.

1. Split the weight matrix into `block_h x block_w` blocks.
2. Inside every block, apply contiguous 2:4 sparsity along the K dimension.
3. For every group of `block_m` consecutive blocks along K, keep the best
   `block_n` blocks.
4. The final mask is the intersection of the block-level N:M mask and the
   intra-block 2:4 mask.

Density:

```text
block_n / block_m * 1/2
```

Default `block_n=1, block_m=2` gives 25% density and 75% sparsity.

## Hybrid Block Sparse

`hybrid_block_sparse` mixes dense blocks and sparse blocks in the same routed
expert weight matrix.

1. Split the weight matrix into `block_h x block_w` blocks.
2. Compute the pruning loss of applying the inner sparse pattern to each block.
3. For every group of `block_m` consecutive blocks along K, select the lowest
   loss `block_n` blocks to become sparse blocks.
4. Selected sparse blocks use the current inner pattern, contiguous 2:4 along K.
5. Non-selected blocks remain dense.

Overall sparsity:

```text
block_n / block_m * 1/2
```

Default `block_n=1, block_m=2` gives 25% sparsity. The overall format is not
2:4; 2:4 is only the current inner pattern used inside selected sparse blocks.

### Canonical weight storage

Hybrid weights use one shared block selector and two compact value streams.
For each block row and group, bit `j` of `block_selector` is one when local
block `j` uses 2:4 sparsity and zero when it remains dense. Dense blocks are
stored in `dense_values`; sparse blocks are stored as `sparse_values` plus a
logical 2:4 pair code in `sparse_metadata`. Both streams follow increasing
local block order. The selector uses `torch.int64` in the canonical format;
the kernel backend may repack it and the 2:4 metadata for a target GPU.

Example:

```bash
python evaluation/wanda/main.py \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --prune_method wanda \
  --sparsity_type hybrid_block_sparse \
  --block_h 16 \
  --block_w 16 \
  --block_n 1 \
  --block_m 2 \
  --hybrid_block_score squared_sum
```

## Scope

Both patterns currently prune only Qwen1.5-MoE routed expert projections:

```text
model.layers.*.mlp.experts.*.{gate_proj,up_proj,down_proj}
```

Attention, router, dense MLP, and shared expert layers are not pruned.
