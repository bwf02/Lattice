"""Export Qwen1.5-MoE routed experts to SparseGEMM hybrid sparse format."""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    num_workers: int = 1
    skip_existing: bool = False


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

    The exported weights match SGLang fused MoE layout:

    - ``w13_weight``: ``[num_experts, 2 * intermediate_size, hidden_size]``,
      with ``gate_proj`` followed by ``up_proj``.
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
            "w13_weight": "[E, 2I, H] = concat(gate_proj, up_proj, dim=1)",
            "down_proj": "[E, H, I]",
        },
        "export_options": asdict(options),
        "weights": [],
    }

    # Filter out layers that already have both .pt files when skip_existing
    layers_to_export = list(layers)
    if options.skip_existing:
        layers_to_export = [
            lid for lid in layers
            if not _layer_files_exist(output_dir, lid)
        ]
        skipped = len(layers) - len(layers_to_export)
        if skipped:
            print(f"Skipping {skipped} layer(s) with existing .pt files")

    results: Dict[int, List[dict]] = {}
    if options.num_workers > 1:
        # Parallel: dispatch layers to worker processes
        workers = min(options.num_workers, len(layers_to_export))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _export_single_layer_worker,
                    checkpoint_dir,
                    layer_id,
                    output_dir,
                    layout,
                    options,
                ): layer_id
                for layer_id in layers_to_export
            }
            for future in as_completed(futures):
                layer_id = futures[future]
                results[layer_id] = future.result()
                print(f"  layer {layer_id} done")
    else:
        # Serial: process layers in order
        for layer_id in layers_to_export:
            results[layer_id] = _export_single_layer_serial(
                checkpoint_dir, layer_id, key_to_file, grouped_keys,
                output_dir, layout, options, dtype,
            )

    # Flatten manifest entries in layer order
    for layer_id in layers:
        if layer_id in results:
            manifest["weights"].extend(results[layer_id])

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def _layer_files_exist(output_dir: Path, layer_id: int) -> bool:
    w13 = output_dir / "weights" / f"layer_{layer_id:03d}_w13_weight.pt"
    down = output_dir / "weights" / f"layer_{layer_id:03d}_down_proj.pt"
    return w13.is_file() and down.is_file()


def _export_single_layer_worker(
    checkpoint_dir: Path,
    layer_id: int,
    output_dir: Path,
    layout: HybridBlockSparseConfig,
    options: ExportOptions,
) -> List[dict]:
    """Worker entry point for parallel export. Rebuilds key index locally."""
    threads = max(1, (os.cpu_count() or 1) // max(1, options.num_workers))
    torch.set_num_threads(threads)
    key_to_file = _build_tensor_index(checkpoint_dir)
    grouped_keys = _collect_qwen_moe_keys(key_to_file)
    dtype = _parse_dtype(options.dtype)
    return _export_single_layer_serial(
        checkpoint_dir, layer_id, key_to_file, grouped_keys,
        output_dir, layout, options, dtype,
    )


def _export_single_layer_serial(
    checkpoint_dir: Path,
    layer_id: int,
    key_to_file: Mapping[str, Path],
    grouped_keys: Mapping[int, Mapping[Tuple[int, str], str]],
    output_dir: Path,
    layout: HybridBlockSparseConfig,
    options: ExportOptions,
    dtype: torch.dtype,
) -> List[dict]:
    """Process one MoE layer: load, pack and save w13 + down weights."""
    layer_keys = grouped_keys[layer_id]
    experts = _validate_complete_layer(layer_id, layer_keys)
    layer_entries = []

    gate_weight = _stack_expert_projection(
        key_to_file, layer_keys, experts, "gate_proj"
    )
    up_weight = _stack_expert_projection(key_to_file, layer_keys, experts, "up_proj")
    w13_weight = (
        torch.cat([gate_weight, up_weight], dim=1).to(dtype=dtype).contiguous()
    )
    layer_entries.append(
        _pack_and_save(
            weight=w13_weight,
            logical_name=f"model.layers.{layer_id}.mlp.experts.w13_weight",
            output_path=(
                output_dir / "weights" / f"layer_{layer_id:03d}_w13_weight.pt"
            ),
            layout=layout,
            options=options,
            source_keys=[
                key
                for expert_id in experts
                for key in (
                    layer_keys[(expert_id, "gate_proj")],
                    layer_keys[(expert_id, "up_proj")],
                )
            ],
        )
    )
    del gate_weight, up_weight, w13_weight

    down_weight = _stack_expert_projection(
        key_to_file, layer_keys, experts, "down_proj"
    )
    down_weight = down_weight.to(dtype=dtype).contiguous()
    layer_entries.append(
        _pack_and_save(
            weight=down_weight,
            logical_name=f"model.layers.{layer_id}.mlp.experts.down_proj.weight",
            output_path=(
                output_dir / "weights" / f"layer_{layer_id:03d}_down_proj.pt"
            ),
            layout=layout,
            options=options,
            source_keys=[
                layer_keys[(expert_id, "down_proj")] for expert_id in experts
            ],
        )
    )
    del down_weight
    return layer_entries


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
