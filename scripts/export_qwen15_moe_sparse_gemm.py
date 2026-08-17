#!/usr/bin/env python3
"""Export supported MoE experts to SparseGEMM hybrid sparse weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosaic_moe.export.hybrid_sparse_checkpoint import (
    ExportOptions,
    export_moe_hybrid_sparse,
    maybe_download_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download or read a supported MoE model and export SGLang-ready SparseGEMM "
            "hybrid block sparse routed expert weights."
        )
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen1.5-MoE-A2.7B",
        help="ModelScope model id used when --model-dir is not already populated.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/tmp/mosaic/models/Qwen1.5-MoE-A2.7B"),
        help="Local HF/ModelScope checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/mosaic/sgl_hybrid_sparse/qwen1_5_moe"),
        help="Directory for SparseGEMM packed weights and manifest.json.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require --model-dir to already contain the checkpoint.",
    )
    parser.add_argument(
        "--mask-source",
        choices=("magnitude", "zeros"),
        default="magnitude",
        help=(
            "magnitude builds a fresh hybrid block sparse mask; zeros treats "
            "existing zero weights as the mask."
        ),
    )
    parser.add_argument("--block-h", type=int, default=64)
    parser.add_argument("--block-w", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=1)
    parser.add_argument("--block-m", type=int, default=2)
    parser.add_argument(
        "--score-mode",
        choices=("sum", "squared_sum", "max_row_squared"),
        default="sum",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "bf16", "float16", "fp16", "float32", "fp32"),
        default="bfloat16",
        help="Output value dtype before SparseGEMM packing.",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Pack only the first N MoE layers for smoke validation.",
    )
    parser.add_argument(
        "--keep-dense",
        action="store_true",
        help="Also save dense zero-filled weights in each .pt file for debugging.",
    )
    parser.add_argument(
        "--include-shared-expert",
        action="store_true",
        help="Also export the per-layer shared expert gate/up and down weights.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for layer-level export.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip layers whose .pt files already exist in the output directory.",
    )
    parser.add_argument(
        "--print-key-map",
        action="store_true",
        help="Print HF checkpoint source keys and exported SparseGEMM logical keys.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.no_download:
        maybe_download_model(args.model_id, args.model_dir)

    options = ExportOptions(
        block_h=args.block_h,
        block_w=args.block_w,
        block_n=args.block_n,
        block_m=args.block_m,
        score_mode=args.score_mode,
        mask_source=args.mask_source,
        dtype=args.dtype,
        max_layers=args.max_layers,
        keep_dense=args.keep_dense,
        include_shared_expert=args.include_shared_expert,
        num_workers=args.num_workers,
        skip_existing=args.skip_existing,
    )
    manifest_path = export_moe_hybrid_sparse(
        checkpoint_dir=args.model_dir,
        output_dir=args.output_dir,
        options=options,
    )
    if args.print_key_map:
        _print_key_map(manifest_path)
    print(f"Exported SparseGEMM hybrid sparse weights: {manifest_path}")


def _print_key_map(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    print("SparseGEMM export key map:")
    for weight in manifest["weights"]:
        print(f"- exported: {weight['logical_name']}")
        print(f"  file: {weight['file']}")
        print(f"  original_shape: {weight['original_shape']}")
        print(f"  sparsity: {weight['sparsity']:.6f}")
        print(f"  has_hardware_metadata: {weight['has_hardware_metadata']}")
        print("  tensor_shapes:")
        for name, shape in weight["tensor_shapes"].items():
            print(f"    {name}: {shape}")
        print("  source:")
        for key in weight["source_keys"]:
            print(f"    - {key}")


if __name__ == "__main__":
    main()
