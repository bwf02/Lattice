import re
from dataclasses import dataclass

import torch
import torch.nn as nn

from sparse_gemm.hybrid_sparse import HybridBlockSparseLayout


_ROUTED_EXPERT_PATTERN = re.compile(
    r"(?:^|\.)mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"
)


@dataclass(frozen=True)
class HBNMConfig:
    block_h: int = 16
    block_w: int = 16
    block_n: int = 1
    block_m: int = 2

    @property
    def density(self):
        return self.block_n / self.block_m * 0.5

    @property
    def sparsity(self):
        return 1.0 - self.density


@dataclass(frozen=True)
class HybridBlockSparseConfig(HybridBlockSparseLayout):
    """Pruning options layered on SparseGEMM's kernel-facing layout."""

    score_mode: str = "sum"


def validate_hb_nm_config(config):
    if config.block_h <= 0:
        raise ValueError("block_h must be greater than zero")
    if config.block_w <= 0:
        raise ValueError("block_w must be greater than zero")
    if config.block_w % 4 != 0:
        raise ValueError("block_w must be divisible by 4 for intra-block 2:4")
    if not 0 < config.block_n <= config.block_m:
        raise ValueError("block_n and block_m must satisfy 0 < block_n <= block_m")


def validate_hb_nm_options(config, prune_method, prune_n=0, prune_m=0):
    validate_hb_nm_config(config)
    if prune_n != 0 or prune_m != 0:
        raise ValueError("HB-N:M cannot be combined with --prune_n or --prune_m")
    if prune_method not in {"magnitude", "wanda"}:
        raise ValueError("HB-N:M currently supports only magnitude and wanda pruning")


def validate_hybrid_block_sparse_options(config, prune_method, prune_n=0, prune_m=0):
    validate_hb_nm_config(config)
    if config.score_mode not in {"sum", "squared_sum", "max_row_squared"}:
        raise ValueError(
            "hybrid_block_sparse score_mode must be sum, squared_sum, or "
            "max_row_squared"
        )
    if prune_n != 0 or prune_m != 0:
        raise ValueError("hybrid_block_sparse cannot be combined with --prune_n or --prune_m")
    if prune_method not in {"magnitude", "wanda"}:
        raise ValueError(
            "hybrid_block_sparse currently supports only magnitude and wanda pruning"
        )


def validate_hb_nm_shape(shape, config):
    validate_hb_nm_config(config)
    if len(shape) != 2:
        raise ValueError(f"HB-N:M requires a 2D weight matrix, got shape {tuple(shape)}")

    rows, columns = shape
    if rows % config.block_h != 0:
        raise ValueError(
            f"weight rows ({rows}) must be divisible by block_h ({config.block_h})"
        )
    if columns % config.block_w != 0:
        raise ValueError(
            f"weight columns ({columns}) must be divisible by block_w ({config.block_w})"
        )

    block_columns = columns // config.block_w
    if block_columns % config.block_m != 0:
        raise ValueError(
            f"block columns ({block_columns}) must be divisible by block_m "
            f"({config.block_m})"
        )


def magnitude_importance(weight):
    return weight.detach().abs().float()


def wanda_importance(weight, activation_scale):
    if activation_scale.numel() != weight.shape[1]:
        raise ValueError(
            "activation_scale length must match the weight input dimension"
        )
    scale = torch.sqrt(activation_scale.detach().float()).reshape(1, -1)
    return weight.detach().abs().float() * scale


def build_hb_nm_prune_mask(importance, config):
    """Return a boolean mask where True entries are pruned."""
    validate_hb_nm_shape(importance.shape, config)

    rows, columns = importance.shape
    block_rows = rows // config.block_h
    block_columns = columns // config.block_w

    blocks = (
        importance.reshape(
            block_rows, config.block_h, block_columns, config.block_w
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    groups_of_four = blocks.reshape(
        block_rows,
        block_columns,
        config.block_h,
        config.block_w // 4,
        4,
    )
    inner_order = torch.argsort(
        groups_of_four, dim=-1, descending=True, stable=True
    )
    inner_keep = torch.zeros_like(groups_of_four, dtype=torch.bool)
    inner_keep.scatter_(-1, inner_order[..., :2], True)

    block_scores = (groups_of_four * inner_keep).sum(dim=(-1, -2, -3))
    block_groups = block_scores.reshape(
        block_rows, block_columns // config.block_m, config.block_m
    )
    block_order = torch.argsort(
        block_groups, dim=-1, descending=True, stable=True
    )
    block_keep = torch.zeros_like(block_groups, dtype=torch.bool)
    block_keep.scatter_(-1, block_order[..., : config.block_n], True)
    block_keep = block_keep.reshape(block_rows, block_columns)

    keep_blocks = inner_keep.reshape(
        block_rows, block_columns, config.block_h, config.block_w
    ) & block_keep[..., None, None]
    keep_mask = (
        keep_blocks.permute(0, 2, 1, 3).contiguous().reshape(rows, columns)
    )
    return ~keep_mask


def build_hybrid_block_sparse_prune_mask(importance, config):
    """Prune 2:4 in the lowest-loss N blocks per group; leave other blocks dense."""
    validate_hb_nm_shape(importance.shape, config)

    rows, columns = importance.shape
    block_rows = rows // config.block_h
    block_columns = columns // config.block_w
    blocks = (
        importance.reshape(
            block_rows, config.block_h, block_columns, config.block_w
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    groups_of_four = blocks.reshape(
        block_rows,
        block_columns,
        config.block_h,
        config.block_w // 4,
        4,
    )
    inner_order = torch.argsort(
        groups_of_four, dim=-1, descending=True, stable=True
    )
    inner_keep = torch.zeros_like(groups_of_four, dtype=torch.bool)
    inner_keep.scatter_(-1, inner_order[..., :2], True)
    inner_prune = ~inner_keep

    pruned_importance = groups_of_four * inner_prune
    if config.score_mode == "sum":
        block_losses = pruned_importance.sum(dim=(-1, -2, -3))
    elif config.score_mode == "squared_sum":
        block_losses = pruned_importance.square().sum(dim=(-1, -2, -3))
    else:
        block_losses = pruned_importance.square().sum(dim=(-1, -2)).amax(dim=-1)
    block_groups = block_losses.reshape(
        block_rows, block_columns // config.block_m, config.block_m
    )
    block_order = torch.argsort(
        block_groups, dim=-1, descending=False, stable=True
    )
    sparse_blocks = torch.zeros_like(block_groups, dtype=torch.bool)
    sparse_blocks.scatter_(-1, block_order[..., : config.block_n], True)
    sparse_blocks = sparse_blocks.reshape(block_rows, block_columns)

    prune_blocks = inner_prune.reshape(
        block_rows, block_columns, config.block_h, config.block_w
    ) & sparse_blocks[..., None, None]
    return prune_blocks.permute(0, 2, 1, 3).contiguous().reshape(rows, columns)


def is_routed_expert_linear(name, module):
    return isinstance(module, nn.Linear) and _ROUTED_EXPERT_PATTERN.search(name) is not None


def find_routed_expert_linears(module):
    return {
        name: child
        for name, child in module.named_modules()
        if is_routed_expert_linear(name, child)
    }


def validate_qwen2_moe_layout(model):
    if getattr(model.config, "model_type", None) != "qwen2_moe":
        raise ValueError(
            "HB-N:M currently supports only Qwen1.5-MoE models "
            "with model_type='qwen2_moe'"
        )

    if not any(find_routed_expert_linears(layer) for layer in model.model.layers):
        raise ValueError(
            "The loaded Qwen2-MoE model does not expose routed experts as "
            "mlp.experts.<id>.{gate_proj,up_proj,down_proj} nn.Linear modules. "
            "Use the Transformers 4.57.3 model layout."
        )
