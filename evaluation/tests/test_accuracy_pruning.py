import unittest
from types import SimpleNamespace

import torch
from torch import nn

from evaluation.pruning import Recipe, expert_targets, make_mask, prune_model


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(16, 16, bias=False)
        self.up_proj = nn.Linear(16, 16, bias=False)
        self.down_proj = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.mlp = nn.Module()
        self.mlp.experts = nn.ModuleList([Expert(), Expert()])
        self.mlp.gate = nn.Linear(16, 2)
        self.mlp.shared_expert = Expert()

    def forward(self, x):
        return sum(expert(x) for expert in self.mlp.experts)


class PruningTests(unittest.TestCase):
    def test_nm_exact_counts_with_ties_and_expert_rows(self):
        for pattern, ratio, width in [("2:4", .5, 4), ("2:8", .75, 8)]:
            mask = make_mask(torch.ones(16, 32), Recipe(pattern=pattern, sparsity=ratio))
            self.assertTrue((mask.reshape(16, -1, width).sum(-1) == width - 2).all())
            self.assertTrue(torch.equal(mask, make_mask(torch.ones(16, 32), Recipe(pattern=pattern, sparsity=ratio))))

    def test_nm_rejects_wrong_ratio_and_partial_group(self):
        with self.assertRaises(ValueError):
            make_mask(torch.ones(4, 16), Recipe(pattern="2:8", sparsity=.5))
        with self.assertRaises(ValueError):
            make_mask(torch.ones(4, 10), Recipe(pattern="2:8", sparsity=.75))

    def test_hibnm_preserves_sensitive_row(self):
        importance = torch.ones(4, 8)
        importance[0, :4] = 10
        mask = make_mask(importance, Recipe(block_h=4, block_w=4))
        self.assertFalse(mask[:, :4].any())
        self.assertEqual(int(mask[:, 4:].sum()), 8)

    def test_unstructured_boundaries(self):
        for ratio in (0., 1.):
            mask = make_mask(torch.ones(4, 16), Recipe(pattern="unstructured", sparsity=ratio))
            self.assertEqual(int(mask.sum()), int(64 * ratio))

    def test_linear_wanda_and_protected_parameters(self):
        model = LinearModel()
        original = {k: v.clone() for k, v in model.state_dict().items()}
        records = prune_model(model, Recipe(pattern="2:8", sparsity=.75), [{"x": torch.randn(8, 16)}])
        self.assertEqual(len(records), 6)
        for name, value in model.state_dict().items():
            if ".experts." in name:
                self.assertTrue(((value != 0).reshape(16, -1, 8).sum(-1) == 2).all())
            else:
                self.assertTrue(torch.equal(value, original[name]))
        self.assertTrue(model.config.use_cache)

    def test_missing_calibration_fails_without_mutation(self):
        model = LinearModel()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with self.assertRaises(ValueError):
            prune_model(model, Recipe(pattern="2:4", sparsity=.5), [])
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))
        self.assertTrue(model.config.use_cache)
        self.assertFalse(any(m._forward_pre_hooks for m in model.modules()))

    def test_calibration_failure_removes_hooks(self):
        model = LinearModel()
        with self.assertRaises(TypeError):
            prune_model(model, Recipe(), [{"wrong_argument": torch.ones(2)}])
        self.assertTrue(model.config.use_cache)
        self.assertFalse(any(m._forward_pre_hooks for m in model.modules()))

    def test_real_transformers_qwen3_stacked_wanda(self):
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
        for pattern, ratio, width in [("2:4", .5, 4), ("2:8", .75, 8)]:
            config = Qwen3MoeConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                head_dim=8, num_experts=2, num_experts_per_tok=2, moe_intermediate_size=16)
            model = Qwen3MoeForCausalLM(config).eval()
            model.set_experts_implementation("eager")
            router = model.model.layers[0].mlp.gate.weight.detach().clone()
            records = prune_model(model, Recipe(pattern=pattern, sparsity=ratio),
                [{"input_ids": torch.randint(0, 32, (1, 8))}])
            self.assertEqual(len(records), 6)
            targets, _ = expert_targets(model)
            for target in targets:
                self.assertTrue(((target.weight != 0).reshape(16, -1, width).sum(-1) == 2).all())
            self.assertTrue(torch.equal(router, model.model.layers[0].mlp.gate.weight))
            self.assertTrue(torch.isfinite(model(torch.randint(0, 32, (1, 5))).logits).all())


if __name__ == "__main__":
    unittest.main()
