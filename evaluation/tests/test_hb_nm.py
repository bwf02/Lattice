import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


WANDA_DIR = Path(__file__).resolve().parents[1] / "wanda"
sys.path.insert(0, str(WANDA_DIR))

from lib.hb_nm import (  # noqa: E402
    HBNMConfig,
    build_hb_nm_prune_mask,
    find_routed_expert_linears,
    magnitude_importance,
    validate_hb_nm_shape,
    validate_hb_nm_options,
    validate_qwen2_moe_layout,
    wanda_importance,
)
from lib.prune import _move_to_device  # noqa: E402


def _mask_as_blocks(mask, config):
    rows, columns = mask.shape
    return (
        mask.reshape(
            rows // config.block_h,
            config.block_h,
            columns // config.block_w,
            config.block_w,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )


class TestHBNMMask(unittest.TestCase):
    def test_optional_calibration_tensor_device_move(self):
        tensor = torch.ones(2)
        self.assertIsNone(_move_to_device(None, torch.device("cpu")))
        self.assertIs(_move_to_device(tensor, torch.device("cpu")), tensor)

    def assert_hb_nm_constraints(self, importance, config):
        prune_mask = build_hb_nm_prune_mask(importance, config)
        keep_blocks = ~_mask_as_blocks(prune_mask, config)

        nonzero_blocks = keep_blocks.any(dim=(-1, -2))
        block_groups = nonzero_blocks.reshape(
            nonzero_blocks.shape[0], -1, config.block_m
        )
        self.assertTrue(
            torch.all(block_groups.sum(dim=-1) == config.block_n).item()
        )

        retained = keep_blocks[nonzero_blocks]
        groups_of_four = retained.reshape(
            -1, config.block_h, config.block_w // 4, 4
        )
        self.assertTrue(torch.all(groups_of_four.sum(dim=-1) == 2).item())

        expected_sparsity = config.sparsity
        actual_sparsity = prune_mask.float().mean().item()
        self.assertAlmostEqual(actual_sparsity, expected_sparsity, places=6)

    def test_default_mask_has_seventy_five_percent_sparsity(self):
        config = HBNMConfig()
        importance = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64)
        self.assert_hb_nm_constraints(importance, config)

    def test_configurable_block_shapes(self):
        cases = [
            (HBNMConfig(block_h=8, block_w=16), (16, 64)),
            (HBNMConfig(block_h=16, block_w=32), (32, 128)),
        ]
        for config, shape in cases:
            with self.subTest(config=config):
                importance = torch.rand(shape, generator=torch.Generator().manual_seed(7))
                self.assert_hb_nm_constraints(importance, config)

    def test_ties_keep_lower_element_and_block_indices(self):
        config = HBNMConfig(block_h=4, block_w=4, block_n=1, block_m=2)
        importance = torch.ones(4, 8)
        keep_mask = ~build_hb_nm_prune_mask(importance, config)

        self.assertTrue(torch.all(keep_mask[:, :2]).item())
        self.assertTrue(torch.all(~keep_mask[:, 2:]).item())

    def test_mask_is_deterministic(self):
        config = HBNMConfig(block_h=8, block_w=16)
        importance = torch.rand((16, 64), generator=torch.Generator().manual_seed(11))
        first = build_hb_nm_prune_mask(importance, config)
        second = build_hb_nm_prune_mask(importance, config)
        self.assertTrue(torch.equal(first, second))

    def test_importance_functions(self):
        weight = torch.tensor([[-2.0, 3.0], [4.0, -5.0]])
        scale = torch.tensor([1.0, 4.0])
        self.assertTrue(torch.equal(magnitude_importance(weight), weight.abs()))
        self.assertTrue(
            torch.equal(
                wanda_importance(weight, scale),
                torch.tensor([[2.0, 6.0], [4.0, 10.0]]),
            )
        )

    def test_invalid_shapes_and_configuration(self):
        cases = [
            ((31, 64), HBNMConfig(), "rows"),
            ((32, 63), HBNMConfig(), "columns"),
            ((32, 48), HBNMConfig(), "block columns"),
            ((32, 64), HBNMConfig(block_w=10), "divisible by 4"),
            ((32, 64), HBNMConfig(block_n=3, block_m=2), "block_n"),
        ]
        for shape, config, message in cases:
            with self.subTest(shape=shape, config=config):
                with self.assertRaisesRegex(ValueError, message):
                    validate_hb_nm_shape(shape, config)

    def test_hb_nm_rejects_legacy_nm_arguments(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            validate_hb_nm_options(
                HBNMConfig(), prune_method="wanda", prune_n=2, prune_m=8
            )

    def test_hb_nm_rejects_sparsegpt(self):
        with self.assertRaisesRegex(ValueError, "only magnitude and wanda"):
            validate_hb_nm_options(HBNMConfig(), prune_method="sparsegpt")


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(32, 32, bias=False)
        self.up_proj = nn.Linear(32, 32, bias=False)
        self.down_proj = nn.Linear(32, 32, bias=False)


class _MockQwenMoeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(32, 32, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(32, 2, bias=False)
        self.mlp.experts = nn.ModuleList([_Expert(), _Expert()])
        self.mlp.shared_expert = _Expert()


class TestRoutedExpertFiltering(unittest.TestCase):
    def test_only_routed_expert_projections_are_selected(self):
        layer = _MockQwenMoeLayer()
        selected = find_routed_expert_linears(layer)

        self.assertEqual(len(selected), 6)
        self.assertTrue(
            all(
                name.startswith("mlp.experts.")
                and name.endswith(("gate_proj", "up_proj", "down_proj"))
                for name in selected
            )
        )
        self.assertNotIn("self_attn.q_proj", selected)
        self.assertNotIn("mlp.gate", selected)
        self.assertNotIn("mlp.shared_expert.gate_proj", selected)

    def test_pruning_selected_layers_leaves_other_linears_unchanged(self):
        layer = _MockQwenMoeLayer()
        selected = find_routed_expert_linears(layer)
        untouched_before = {
            name: module.weight.detach().clone()
            for name, module in layer.named_modules()
            if isinstance(module, nn.Linear) and name not in selected
        }

        config = HBNMConfig()
        for module in selected.values():
            mask = build_hb_nm_prune_mask(
                magnitude_importance(module.weight), config
            )
            module.weight.data[mask] = 0

        for name, before in untouched_before.items():
            module = dict(layer.named_modules())[name]
            self.assertTrue(torch.equal(module.weight, before), name)
        for name, module in selected.items():
            self.assertAlmostEqual(
                (module.weight == 0).float().mean().item(), 0.75, places=6, msg=name
            )

    def test_qwen2_moe_layout_validation(self):
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="qwen2_moe"),
            model=SimpleNamespace(layers=[_MockQwenMoeLayer()]),
        )
        validate_qwen2_moe_layout(model)

    def test_qwen2_moe_layout_validation_rejects_fused_experts(self):
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="qwen2_moe"),
            model=SimpleNamespace(layers=[nn.Module()]),
        )
        with self.assertRaisesRegex(ValueError, "Transformers 4.57.3"):
            validate_qwen2_moe_layout(model)


if __name__ == "__main__":
    unittest.main()
