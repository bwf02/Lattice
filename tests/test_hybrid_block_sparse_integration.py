import unittest

import torch

from lattice.patterns import (
    HybridBlockSparseConfig,
    build_hybrid_block_sparse_prune_mask,
)
from lattice.weight_convert import dense_to_hybrid_block_sparse
from sparse_gemm.hybrid_sparse import hybrid_block_sparse_gemm_ref


class TestHybridBlockSparseIntegration(unittest.TestCase):
    def test_pruning_mask_packs_and_runs_reference(self):
        config = HybridBlockSparseConfig(
            block_h=2,
            block_w=4,
            block_n=1,
            block_m=2,
            score_mode="sum",
        )
        weight = torch.randn(4, 8, generator=torch.Generator().manual_seed(11))
        activation = torch.randn(3, 8, generator=torch.Generator().manual_seed(12))
        prune_mask = build_hybrid_block_sparse_prune_mask(weight.abs(), config)

        packed = dense_to_hybrid_block_sparse(weight, prune_mask, config)
        actual = hybrid_block_sparse_gemm_ref(activation, packed)
        expected_weight = weight.masked_fill(prune_mask, 0)

        self.assertEqual(packed.layout.block_h, config.block_h)
        self.assertEqual(packed.layout.block_w, config.block_w)
        self.assertEqual(packed.layout.block_n, config.block_n)
        self.assertEqual(packed.layout.block_m, config.block_m)
        self.assertFalse(hasattr(packed.layout, "score_mode"))
        torch.testing.assert_close(actual, activation @ expected_weight.t())


if __name__ == "__main__":
    unittest.main()
