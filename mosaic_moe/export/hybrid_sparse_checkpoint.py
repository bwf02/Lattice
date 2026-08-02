"""Export Qwen1.5-MoE routed experts to SparseGEMM hybrid sparse format."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch

from mosaic_moe.patterns import (
    HybridBlockSparseConfig,
    build_hybrid_block_sparse_prune_mask,
)
from sparse_gemm.hybrid_sparse import dense_to_hybrid_block_sparse


_EXPERT_WEIGHT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)


@dataclass(frozen=True)
class ExportOptions:
    block_h: int = 64
    block_w: int = 64
    block_n: int = 1
    block_m: int = 2
    score_mode: str = "sum"
    mask_source: str = "magnitude"
    dtype: str = "bfloat16"
    max_layers: Optional[int] = None
    keep_dense: bool = False


def maybe_download_model(model_id: str, model_dir: Path) -> None:
    """Download a model with ModelScope when ``model_dir`` is not ready."""
    if (model_dir / "config.json").is_file():
        return

    from modelscope import snapshot_download

    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(model_id=model_id, local_dir=str(model_dir))


def export_qwen15_moe_hybrid_sparse(
    checkpoint_dir: Path,
    output_dir: Path,
    options: ExportOptions,
) -> Path:
    """Pack Qwen1.5-MoE routed expert weights into SparseGEMM format.

    The exported weights keep the original routed expert projections separate:

    - ``gate_proj``: ``[num_experts, intermediate_size, hidden_size]``
    - ``up_proj``: ``[num_experts, intermediate_size, hidden_size]``
    - ``down_proj``: ``[num_experts, hidden_size, intermediate_size]``
    """
    checkpoint_dir = checkpoint_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "weights").mkdir(exist_ok=True)

    config = _load_json(checkpoint_dir / "config.json")
    if config.get("model_type") != "qwen2_moe":
        raise ValueError(
            "expected a Qwen1.5-MoE checkpoint with model_type='qwen2_moe', "
            f"got {config.get('model_type')!r}"
        )

    key_to_file = _build_tensor_index(checkpoint_dir)
    grouped_keys = _collect_qwen_moe_keys(key_to_file)
    layers = sorted(grouped_keys)
    if options.max_layers is not None:
        layers = layers[: options.max_layers]

    layout = HybridBlockSparseConfig(
        block_h=options.block_h,
        block_w=options.block_w,
        block_n=options.block_n,
        block_m=options.block_m,
        score_mode=options.score_mode,
    )
    dtype = _parse_dtype(options.dtype)

    manifest = {
        "format_version": 1,
        "format": "sparse_gemm.hybrid_block_sparse",
        "source_checkpoint": str(checkpoint_dir),
        "model_type": config.get("model_type"),
        "model_config": {
            key: config.get(key)
            for key in (
                "hidden_size",
                "intermediate_size",
                "moe_intermediate_size",
                "num_experts",
                "num_experts_per_tok",
                "num_hidden_layers",
            )
            if key in config
        },
        "projection_layout": {
            "gate_proj": "[E, I, H]",
            "up_proj": "[E, I, H]",
            "down_proj": "[E, H, I]",
        },
        "export_options": asdict(options),
        "weights": [],
    }

    for layer_id in layers:
        layer_keys = grouped_keys[layer_id]
        experts = _validate_complete_layer(layer_id, layer_keys)
        layer_entries = []
        for projection in ("gate_proj", "up_proj", "down_proj"):
            weight = _stack_expert_projection(
                key_to_file, layer_keys, experts, projection
            )
            weight = weight.to(dtype=dtype).contiguous()
            entry = _pack_and_save(
                weight=weight,
                logical_name=f"model.layers.{layer_id}.mlp.experts.{projection}.weight",
                output_path=(
                    output_dir / "weights" / f"layer_{layer_id:03d}_{projection}.pt"
                ),
                layout=layout,
                options=options,
                source_keys=[
                    layer_keys[(expert_id, projection)] for expert_id in experts
                ],
            )
            layer_entries.append(entry)
            del weight

        manifest["weights"].extend(layer_entries)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def _pack_and_save(
    *,
    weight: torch.Tensor,
    logical_name: str,
    output_path: Path,
    layout: HybridBlockSparseConfig,
    options: ExportOptions,
    source_keys: List[str],
) -> dict:
    prune_mask = _build_prune_mask(weight, layout, options.mask_source)
    packed = dense_to_hybrid_block_sparse(weight, prune_mask, layout)

    payload = {
        "format_version": 1,
        "format": "sparse_gemm.hybrid_block_sparse",
        "logical_name": logical_name,
        "original_shape": tuple(packed.original_shape),
        "layout": {
            "block_h": packed.layout.block_h,
            "block_w": packed.layout.block_w,
            "block_n": packed.layout.block_n,
            "block_m": packed.layout.block_m,
        },
        "block_selector": packed.block_selector.cpu(),
        "dense_values": packed.dense_values.cpu(),
        "sparse_values": packed.sparse_values.cpu(),
        "sparse_metadata": packed.sparse_metadata.cpu(),
        "hardware_metadata": (
            None if packed.hardware_metadata is None else packed.hardware_metadata.cpu()
        ),
    }
    if options.keep_dense:
        payload["dense_zero_weight"] = weight.masked_fill(prune_mask, 0).cpu()
    torch.save(payload, output_path)

    total = prune_mask.numel()
    pruned = int(prune_mask.sum().item())
    entry = {
        "logical_name": logical_name,
        "file": str(output_path.relative_to(output_path.parent.parent)),
        "source_keys": source_keys,
        "original_shape": list(weight.shape),
        "dtype": str(weight.dtype).replace("torch.", ""),
        "sparsity": pruned / total,
        "nonzero_density": 1.0 - pruned / total,
        "has_hardware_metadata": packed.hardware_metadata is not None,
        "tensor_shapes": {
            "block_selector": list(packed.block_selector.shape),
            "dense_values": list(packed.dense_values.shape),
            "sparse_values": list(packed.sparse_values.shape),
            "sparse_metadata": list(packed.sparse_metadata.shape),
            "hardware_metadata": (
                None
                if packed.hardware_metadata is None
                else list(packed.hardware_metadata.shape)
            ),
        },
    }
    return entry


def _build_prune_mask(
    weight: torch.Tensor, layout: HybridBlockSparseConfig, mask_source: str
) -> torch.Tensor:
    if mask_source == "magnitude":
        masks = [
            build_hybrid_block_sparse_prune_mask(expert.abs().float(), layout)
            for expert in weight
        ]
        return torch.stack(masks, dim=0)
    if mask_source == "zeros":
        return weight == 0
    raise ValueError("mask_source must be 'magnitude' or 'zeros'")


def _collect_qwen_moe_keys(
    key_to_file: Mapping[str, Path]
) -> Dict[int, Dict[Tuple[int, str], str]]:
    grouped: Dict[int, Dict[Tuple[int, str], str]] = {}
    for key in key_to_file:
        match = _EXPERT_WEIGHT_RE.match(key)
        if match is None:
            continue
        layer_id = int(match.group("layer"))
        expert_id = int(match.group("expert"))
        proj = match.group("proj")
        grouped.setdefault(layer_id, {})[(expert_id, proj)] = key
    if not grouped:
        raise ValueError(
            "no Qwen1.5-MoE routed expert weights were found in the checkpoint"
        )
    return grouped


def _validate_complete_layer(
    layer_id: int, layer_keys: Mapping[Tuple[int, str], str]
) -> List[int]:
    experts = sorted({expert for expert, _ in layer_keys})
    missing = [
        (expert, proj)
        for expert in experts
        for proj in ("gate_proj", "up_proj", "down_proj")
        if (expert, proj) not in layer_keys
    ]
    if missing:
        raise ValueError(f"layer {layer_id} is missing routed expert weights: {missing}")
    return experts


def _stack_expert_projection(
    key_to_file: Mapping[str, Path],
    layer_keys: Mapping[Tuple[int, str], str],
    experts: Iterable[int],
    projection: str,
) -> torch.Tensor:
    return torch.stack(
        [
            _load_tensor(key_to_file, layer_keys[(expert_id, projection)])
            for expert_id in experts
        ],
        dim=0,
    )


def _load_tensor(key_to_file: Mapping[str, Path], key: str) -> torch.Tensor:
    path = key_to_file[key]
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)
    state = torch.load(path, map_location="cpu", weights_only=True)
    return state[key]


def _build_tensor_index(checkpoint_dir: Path) -> Dict[str, Path]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = _load_json(index_path)
        return {
            key: checkpoint_dir / filename
            for key, filename in index["weight_map"].items()
        }

    safetensor_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if safetensor_files:
        from safetensors import safe_open

        key_to_file: Dict[str, Path] = {}
        for path in safetensor_files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                key_to_file.update({key: path for key in handle.keys()})
        return key_to_file

    bin_index_path = checkpoint_dir / "pytorch_model.bin.index.json"
    if bin_index_path.is_file():
        index = _load_json(bin_index_path)
        return {
            key: checkpoint_dir / filename
            for key, filename in index["weight_map"].items()
        }

    bin_files = sorted(checkpoint_dir.glob("*.bin"))
    if bin_files:
        key_to_file = {}
        for path in bin_files:
            state = torch.load(path, map_location="cpu", weights_only=True)
            key_to_file.update({key: path for key in state.keys()})
            del state
        return key_to_file

    raise FileNotFoundError(f"no safetensors or pytorch_model.bin found in {checkpoint_dir}")


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _parse_dtype(dtype: str) -> torch.dtype:
    choices = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return choices[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {dtype!r}") from exc
