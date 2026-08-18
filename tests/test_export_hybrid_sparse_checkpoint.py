import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from mosaic_moe.export.hybrid_sparse_checkpoint import (
    ExportOptions,
    export_moe_hybrid_sparse,
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
                        "shared_expert_intermediate_size": 8,
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
            shared = "model.layers.0.mlp.shared_expert"
            tensors[f"{shared}.gate_proj.weight"] = torch.arange(
                64, dtype=torch.bfloat16
            ).reshape(8, 8)
            tensors[f"{shared}.up_proj.weight"] = torch.arange(
                64, 128, dtype=torch.bfloat16
            ).reshape(8, 8)
            tensors[f"{shared}.down_proj.weight"] = torch.arange(
                64, dtype=torch.bfloat16
            ).reshape(8, 8)
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
                    include_shared_expert=True,
                ),
            )

            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(len(manifest["weights"]), 4)
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
            self.assertEqual(
                manifest["weights"][2]["logical_name"],
                "model.layers.0.mlp.shared_expert.gate_up_proj.weight",
            )
            self.assertEqual(manifest["weights"][2]["original_shape"], [16, 8])
            self.assertEqual(
                manifest["weights"][3]["logical_name"],
                "model.layers.0.mlp.shared_expert.down_proj.weight",
            )
            self.assertEqual(manifest["weights"][3]["original_shape"], [8, 8])

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

            with patch(
                "mosaic_moe.export.hybrid_sparse_checkpoint._pack_and_save",
                side_effect=AssertionError("existing layers must not be repacked"),
            ):
                resumed_manifest_path = export_moe_hybrid_sparse(
                    checkpoint_dir,
                    output_dir,
                    ExportOptions(
                        block_h=4,
                        block_w=4,
                        include_shared_expert=True,
                        skip_existing=True,
                    ),
                )
            self.assertEqual(
                json.loads(resumed_manifest_path.read_text())["weights"],
                manifest["weights"],
            )

    def test_exports_individual_experts_for_deepseek_and_qwen3(self):
        for model_type, expert_field in (
            ("deepseek_v2", {"n_routed_experts": 2}),
            ("qwen3_moe", {"num_experts": 2}),
            ("ernie4_5_moe", {"moe_num_experts": 2, "moe_k": 1}),
        ):
            with self.subTest(model_type=model_type), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                checkpoint_dir = root / "checkpoint"
                checkpoint_dir.mkdir()
                (checkpoint_dir / "config.json").write_text(
                    json.dumps(
                        {
                            "model_type": model_type,
                            "hidden_size": 8,
                            "moe_intermediate_size": 8,
                            "num_hidden_layers": 1,
                            **expert_field,
                        }
                    )
                )
                tensors = {}
                for expert in range(2):
                    base = f"model.layers.0.mlp.experts.{expert}"
                    for projection in ("gate_proj", "up_proj", "down_proj"):
                        tensors[f"{base}.{projection}.weight"] = torch.arange(
                            64, dtype=torch.bfloat16
                        ).reshape(8, 8) + expert
                save_file(tensors, checkpoint_dir / "model.safetensors")

                manifest_path = export_moe_hybrid_sparse(
                    checkpoint_dir,
                    root / "packed",
                    ExportOptions(block_h=4, block_w=4),
                )
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(manifest["model_type"], model_type)
                self.assertEqual(manifest["source_layout"], "individual_experts")
                self.assertEqual(manifest["model_config"]["num_experts"], 2)
                if model_type == "ernie4_5_moe":
                    self.assertEqual(
                        manifest["model_config"]["num_experts_per_tok"], 1
                    )
                self.assertEqual(manifest["weights"][0]["original_shape"], [2, 16, 8])

    def test_exports_ernie_shared_experts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoint"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "ernie4_5_moe",
                        "hidden_size": 8,
                        "moe_intermediate_size": 8,
                        "moe_num_experts": 1,
                        "moe_k": 1,
                        "moe_num_shared_experts": 2,
                        "num_hidden_layers": 1,
                    }
                )
            )
            tensors = {}
            routed = "model.layers.0.mlp.experts.0"
            shared = "model.layers.0.mlp.shared_experts"
            for prefix, size in ((routed, 8), (shared, 16)):
                tensors[f"{prefix}.gate_proj.weight"] = torch.arange(
                    size * 8, dtype=torch.bfloat16
                ).reshape(size, 8)
                tensors[f"{prefix}.up_proj.weight"] = torch.arange(
                    size * 8, dtype=torch.bfloat16
                ).reshape(size, 8)
                tensors[f"{prefix}.down_proj.weight"] = torch.arange(
                    size * 8, dtype=torch.bfloat16
                ).reshape(8, size)
            save_file(tensors, checkpoint_dir / "model.safetensors")

            manifest_path = export_moe_hybrid_sparse(
                checkpoint_dir,
                root / "packed",
                ExportOptions(
                    block_h=4,
                    block_w=4,
                    include_shared_expert=True,
                ),
            )
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(len(manifest["weights"]), 4)
            self.assertEqual(
                manifest["weights"][2]["source_keys"],
                [
                    "model.layers.0.mlp.shared_experts.gate_proj.weight",
                    "model.layers.0.mlp.shared_experts.up_proj.weight",
                ],
            )
            self.assertEqual(manifest["weights"][2]["original_shape"], [32, 8])
            self.assertEqual(manifest["weights"][3]["original_shape"], [8, 16])

    def test_transposes_llama4_fused_experts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoint"
            output_dir = root / "packed"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "llama4",
                        "text_config": {
                            "model_type": "llama4_text",
                            "hidden_size": 8,
                            "intermediate_size": 8,
                            "num_local_experts": 2,
                            "num_experts_per_tok": 1,
                            "num_hidden_layers": 1,
                        },
                    }
                )
            )
            prefix = "language_model.model.layers.0.feed_forward.experts"
            gate_up = torch.arange(256, dtype=torch.bfloat16).reshape(2, 8, 16)
            down = torch.arange(128, dtype=torch.bfloat16).reshape(2, 8, 8)
            save_file(
                {f"{prefix}.gate_up_proj": gate_up, f"{prefix}.down_proj": down},
                checkpoint_dir / "model.safetensors",
            )

            manifest_path = export_moe_hybrid_sparse(
                checkpoint_dir,
                output_dir,
                ExportOptions(block_h=4, block_w=4, keep_dense=True),
            )
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["model_type"], "llama4_text")
            self.assertEqual(manifest["source_layout"], "llama4_fused_experts")
            self.assertEqual(manifest["weights"][0]["original_shape"], [2, 16, 8])
            self.assertEqual(manifest["weights"][1]["original_shape"], [2, 8, 8])
            for entry, expected in zip(
                manifest["weights"],
                (gate_up.transpose(1, 2), down.transpose(1, 2)),
            ):
                payload = torch.load(
                    output_dir / entry["file"], weights_only=True
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
                pruned = payload["dense_zero_weight"]
                self.assertTrue(torch.equal(hybrid_block_sparse_to_dense(packed), pruned))
                retained = pruned != 0
                self.assertTrue(torch.equal(pruned[retained], expected[retained]))


if __name__ == "__main__":
    unittest.main()
