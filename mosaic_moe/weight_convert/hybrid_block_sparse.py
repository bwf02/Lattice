"""Canonical storage for hybrid dense-block and 2:4-block sparse weights."""

from dataclasses import dataclass
from typing import Tuple

import torch

from mosaic_moe.patterns import HybridBlockSparseConfig, validate_hb_nm_shape


_CODE_TO_PAIR = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
)


@dataclass(frozen=True)
class HybridBlockSparseWeight:
    """A shared block topology with separate dense and compressed 2:4 streams."""

    original_shape: Tuple[int, ...]
    config: HybridBlockSparseConfig
    block_selector: torch.Tensor
    dense_values: torch.Tensor
    sparse_values: torch.Tensor
    sparse_metadata: torch.Tensor

    def to_dense(self) -> torch.Tensor:
        return hybrid_block_sparse_to_dense(self)


def _normalize_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    if weight.dim() not in (2, 3):
        raise ValueError(
            "hybrid block sparse weights must have shape [N, K] or [E, N, K], "
            f"got {tuple(weight.shape)}"
        )
    leading_shape = tuple(weight.shape[:-2])
    return weight.reshape(-1, weight.shape[-2], weight.shape[-1]), leading_shape


def _validate_selector_width(config: HybridBlockSparseConfig) -> None:
    # int64 keeps bit operations portable across CPU and CUDA PyTorch builds.
    if config.block_m > 63:
        raise ValueError("block_m must be at most 63 for the int64 block selector")


def _as_blocks(
    tensor: torch.Tensor, config: HybridBlockSparseConfig
) -> torch.Tensor:
    batch, rows, columns = tensor.shape
    block_rows = rows // config.block_h
    block_groups = columns // (config.block_w * config.block_m)
    return (
        tensor.reshape(
            batch,
            block_rows,
            config.block_h,
            block_groups,
            config.block_m,
            config.block_w,
        )
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
    )


def _restore_leading_shape(tensor: torch.Tensor, leading_shape: Tuple[int, ...]):
    return tensor.reshape(leading_shape + tuple(tensor.shape[1:]))


def dense_to_hybrid_block_sparse(
    weight: torch.Tensor,
    prune_mask: torch.Tensor,
    config: HybridBlockSparseConfig,
) -> HybridBlockSparseWeight:
    """Compress a weight using an explicit hybrid block sparse prune mask.

    ``prune_mask`` is boolean with True entries denoting removed weights. Each
    outer block group must contain exactly ``block_n`` blocks whose every
    contiguous group of four K elements prunes exactly two entries. All other
    blocks must have no structurally pruned entries.
    """
    _validate_selector_width(config)
    flat_weight, leading_shape = _normalize_weight(weight)
    if prune_mask.shape != weight.shape:
        raise ValueError(
            "prune_mask shape must match weight shape, got "
            f"{tuple(prune_mask.shape)} and {tuple(weight.shape)}"
        )
    if prune_mask.dtype != torch.bool:
        raise TypeError("prune_mask must have dtype torch.bool")
    if prune_mask.device != weight.device:
        raise ValueError("prune_mask and weight must be on the same device")

    validate_hb_nm_shape(weight.shape[-2:], config)
    flat_mask = prune_mask.reshape_as(flat_weight)
    weight_blocks = _as_blocks(flat_weight, config)
    mask_blocks = _as_blocks(flat_mask, config)

    batch, block_rows, block_groups, _, _, _ = weight_blocks.shape
    quartets = mask_blocks.reshape(
        batch,
        block_rows,
        block_groups,
        config.block_m,
        config.block_h,
        config.block_w // 4,
        4,
    )
    quartet_pruned = quartets.sum(dim=-1)
    sparse_blocks = mask_blocks.any(dim=(-1, -2))
    valid_sparse = (quartet_pruned == 2).all(dim=(-1, -2))
    valid_dense = (quartet_pruned == 0).all(dim=(-1, -2))
    if not torch.all(torch.where(sparse_blocks, valid_sparse, valid_dense)).item():
        raise ValueError(
            "each selected sparse block must be exactly 2:4 and each dense "
            "block must contain no structurally pruned entries"
        )
    if not torch.all(sparse_blocks.sum(dim=-1) == config.block_n).item():
        raise ValueError(
            f"each group of {config.block_m} blocks must select exactly "
            f"{config.block_n} sparse blocks"
        )

    local_bits = 1 << torch.arange(
        config.block_m, dtype=torch.int64, device=weight.device
    )
    selector = (sparse_blocks.to(torch.int64) * local_bits).sum(dim=-1)

    dense_count = config.block_m - config.block_n
    dense_blocks = weight_blocks[~sparse_blocks].reshape(
        batch,
        block_rows,
        block_groups,
        dense_count,
        config.block_h,
        config.block_w,
    )
    selected_sparse_blocks = weight_blocks[sparse_blocks].reshape(
        batch,
        block_rows,
        block_groups,
        config.block_n,
        config.block_h,
        config.block_w // 4,
        4,
    )
    selected_sparse_mask = quartets[sparse_blocks].reshape_as(
        selected_sparse_blocks
    )

    keep_indices = torch.argsort(
        selected_sparse_mask.to(torch.int8), dim=-1, stable=True
    )[..., :2]
    sparse_values = torch.gather(
        selected_sparse_blocks, dim=-1, index=keep_indices
    ).reshape(
        batch,
        block_rows,
        block_groups,
        config.block_n,
        config.block_h,
        config.block_w // 2,
    )

    pair_to_code = torch.full(
        (4, 4), -1, dtype=torch.int16, device=weight.device
    )
    for code, (first, second) in enumerate(_CODE_TO_PAIR):
        pair_to_code[first, second] = code
    sparse_metadata = pair_to_code[
        keep_indices[..., 0], keep_indices[..., 1]
    ].to(torch.uint8)

    return HybridBlockSparseWeight(
        original_shape=tuple(weight.shape),
        config=config,
        block_selector=_restore_leading_shape(selector, leading_shape),
        dense_values=_restore_leading_shape(dense_blocks, leading_shape),
        sparse_values=_restore_leading_shape(sparse_values, leading_shape),
        sparse_metadata=_restore_leading_shape(sparse_metadata, leading_shape),
    )


def hybrid_block_sparse_to_dense(packed: HybridBlockSparseWeight) -> torch.Tensor:
    """Reconstruct the zero-filled dense-shaped weight from canonical storage."""
    config = packed.config
    _validate_selector_width(config)
    if len(packed.original_shape) not in (2, 3):
        raise ValueError("original_shape must describe [N, K] or [E, N, K]")
    validate_hb_nm_shape(packed.original_shape[-2:], config)

    leading_shape = packed.original_shape[:-2]
    rows, columns = packed.original_shape[-2:]
    batch = 1 if not leading_shape else leading_shape[0]
    block_rows = rows // config.block_h
    block_groups = columns // (config.block_w * config.block_m)
    dense_count = config.block_m - config.block_n

    expected_selector = leading_shape + (block_rows, block_groups)
    expected_dense = leading_shape + (
        block_rows,
        block_groups,
        dense_count,
        config.block_h,
        config.block_w,
    )
    expected_sparse = leading_shape + (
        block_rows,
        block_groups,
        config.block_n,
        config.block_h,
        config.block_w // 2,
    )
    expected_metadata = leading_shape + (
        block_rows,
        block_groups,
        config.block_n,
        config.block_h,
        config.block_w // 4,
    )
    actual_shapes = (
        tuple(packed.block_selector.shape),
        tuple(packed.dense_values.shape),
        tuple(packed.sparse_values.shape),
        tuple(packed.sparse_metadata.shape),
    )
    expected_shapes = (
        expected_selector,
        expected_dense,
        expected_sparse,
        expected_metadata,
    )
    if actual_shapes != expected_shapes:
        raise ValueError(
            "packed tensor shapes do not match original_shape and config: "
            f"expected {expected_shapes}, got {actual_shapes}"
        )
    if packed.block_selector.dtype != torch.int64:
        raise TypeError("block_selector must have dtype torch.int64")
    if packed.sparse_metadata.dtype != torch.uint8:
        raise TypeError("sparse_metadata must have dtype torch.uint8")
    if packed.dense_values.dtype != packed.sparse_values.dtype:
        raise TypeError("dense_values and sparse_values must use the same dtype")

    device = packed.dense_values.device
    tensors = (
        packed.block_selector,
        packed.sparse_values,
        packed.sparse_metadata,
    )
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("all packed tensors must be on the same device")

    selector = packed.block_selector.reshape(batch, block_rows, block_groups)
    if torch.any(selector < 0).item() or torch.any(
        (selector >> config.block_m) != 0
    ).item():
        raise ValueError("block_selector contains bits outside the block group")
    local_ids = torch.arange(config.block_m, dtype=torch.int64, device=device)
    sparse_blocks = ((selector.unsqueeze(-1) >> local_ids) & 1).bool()
    if not torch.all(sparse_blocks.sum(dim=-1) == config.block_n).item():
        raise ValueError("block_selector does not contain block_n sparse blocks per group")

    blocks = torch.zeros(
        (
            batch,
            block_rows,
            block_groups,
            config.block_m,
            config.block_h,
            config.block_w,
        ),
        dtype=packed.dense_values.dtype,
        device=device,
    )
    blocks[~sparse_blocks] = packed.dense_values.reshape(
        -1, config.block_h, config.block_w
    )

    metadata = packed.sparse_metadata.reshape(
        batch,
        block_rows,
        block_groups,
        config.block_n,
        config.block_h,
        config.block_w // 4,
    )
    if metadata.numel() and metadata.max().item() >= len(_CODE_TO_PAIR):
        raise ValueError("sparse_metadata contains an invalid 2:4 pair code")
    code_to_pair = torch.tensor(
        _CODE_TO_PAIR, dtype=torch.int64, device=device
    )
    pair_indices = code_to_pair[metadata.to(torch.int64)]
    sparse_quartets = torch.zeros(
        metadata.shape + (4,), dtype=packed.sparse_values.dtype, device=device
    )
    sparse_value_pairs = packed.sparse_values.reshape(metadata.shape + (2,))
    sparse_quartets.scatter_(-1, pair_indices, sparse_value_pairs)
    blocks[sparse_blocks] = sparse_quartets.reshape(
        -1, config.block_h, config.block_w
    )

    dense = (
        blocks.permute(0, 1, 4, 2, 3, 5)
        .contiguous()
        .reshape(batch, rows, columns)
    )
    return dense.reshape(packed.original_shape)
