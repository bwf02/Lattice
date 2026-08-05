import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from mosaic_moe.export.hybrid_sparse_checkpoint import (
    ExportOptions,
    export_qwen15_moe_hybrid_sparse,
)
from sparse_gemm.hybrid_sparse import (
    HybridBlockSparseLayout,
    HybridBlockSparseWeight,
    hybrid_block_sparse_to_dense,
)


class TestExportHybridSparseCheckpoint(unittest.TestCase):
    def test_exports_sglang_ready_grouped_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoint"
            output_dir = root / "packed"
            checkpoint_dir.mkdir()

            (checkpoint_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen2_moe",
                        "hidden_size": 8,
                        "moe_intermediate_size": 8,
                        "num_experts": 2,
                        "num_hidden_layers": 1,
                    }
                )
            )

            tensors = {}
            for expert in range(2):
                base = f"model.layers.0.mlp.experts.{expert}"
                tensors[f"{base}.gate_proj.weight"] = torch.arange(
                    64, dtype=torch.bfloat16
                ).reshape(8, 8) + expert
                tensors[f"{base}.up_proj.weight"] = torch.arange(
                    64, 128, dtype=torch.bfloat16
                ).reshape(8, 8) + expert
                tensors[f"{base}.down_proj.weight"] = torch.arange(
                    64, dtype=torch.bfloat16
                ).reshape(8, 8) + expert
            save_file(tensors, checkpoint_dir / "model.safetensors")

            manifest_path = export_qwen15_moe_hybrid_sparse(
                checkpoint_dir,
                output_dir,
                ExportOptions(
                    block_h=4,
                    block_w=4,
                    block_n=1,
                    block_m=2,
                    max_layers=1,
                ),
            )

            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(len(manifest["weights"]), 2)
            self.assertEqual(
                manifest["weights"][0]["logical_name"],
                "model.layers.0.mlp.experts.w13_weight",
            )
            self.assertEqual(
                manifest["weights"][0]["source_keys"],
                [
                    "model.layers.0.mlp.experts.0.gate_proj.weight",
                    "model.layers.0.mlp.experts.0.up_proj.weight",
                    "model.layers.0.mlp.experts.1.gate_proj.weight",
                    "model.layers.0.mlp.experts.1.up_proj.weight",
                ],
            )
            self.assertEqual(manifest["weights"][0]["original_shape"], [2, 16, 8])
            self.assertEqual(
                manifest["weights"][1]["logical_name"],
                "model.layers.0.mlp.experts.down_proj.weight",
            )
            self.assertEqual(
                manifest["weights"][1]["source_keys"],
                [
                    "model.layers.0.mlp.experts.0.down_proj.weight",
                    "model.layers.0.mlp.experts.1.down_proj.weight",
                ],
            )
            self.assertEqual(manifest["weights"][1]["original_shape"], [2, 8, 8])

            payload = torch.load(
                output_dir / manifest["weights"][0]["file"],
                map_location="cpu",
                weights_only=True,
            )
            packed = HybridBlockSparseWeight(
                original_shape=tuple(payload["original_shape"]),
                layout=HybridBlockSparseLayout(**payload["layout"]),
                block_selector=payload["block_selector"],
                dense_values=payload["dense_values"],
                sparse_values=payload["sparse_values"],
                sparse_metadata=payload["sparse_metadata"],
                hardware_metadata=payload["hardware_metadata"],
            )
            dense = hybrid_block_sparse_to_dense(packed)
            self.assertEqual(tuple(dense.shape), (2, 16, 8))
            self.assertEqual(manifest["weights"][0]["sparsity"], 0.25)


if __name__ == "__main__":
    unittest.main()
