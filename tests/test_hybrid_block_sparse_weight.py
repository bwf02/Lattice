import unittest

import torch

from mosaic_moe.patterns import (
    HybridBlockSparseConfig,
    build_hybrid_block_sparse_prune_mask,
)
from mosaic_moe.weight_convert import (
    HybridBlockSparseWeight,
    dense_to_hybrid_block_sparse,
)


class TestHybridBlockSparseWeight(unittest.TestCase):
    def test_shared_selector_and_compact_streams(self):
        config = HybridBlockSparseConfig(
            block_h=2, block_w=4, block_n=1, block_m=2
        )
        weight = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8)
        prune_mask = torch.zeros_like(weight, dtype=torch.bool)
        prune_mask[:, 2:4] = True

        packed = dense_to_hybrid_block_sparse(weight, prune_mask, config)

        self.assertEqual(packed.block_selector.dtype, torch.int64)
        self.assertEqual(packed.block_selector.tolist(), [[1]])
        self.assertEqual(tuple(packed.dense_values.shape), (1, 1, 1, 2, 4))
        self.assertEqual(tuple(packed.sparse_values.shape), (1, 1, 1, 2, 2))
        self.assertEqual(tuple(packed.sparse_metadata.shape), (1, 1, 1, 2, 1))
        self.assertTrue(torch.equal(packed.dense_values[0, 0, 0], weight[:, 4:]))
        self.assertTrue(torch.equal(packed.sparse_values[0, 0, 0], weight[:, :2]))
        self.assertTrue(
            torch.equal(packed.to_dense(), weight.masked_fill(prune_mask, 0))
        )

    def test_round_trip_for_multiple_experts_and_outer_nm(self):
        config = HybridBlockSparseConfig(
            block_h=2, block_w=4, block_n=2, block_m=4
        )
        generator = torch.Generator().manual_seed(41)
        weight = torch.randn((3, 4, 32), generator=generator)
        masks = torch.stack(
            [
                build_hybrid_block_sparse_prune_mask(weight[i].abs(), config)
                for i in range(weight.shape[0])
            ]
        )

        packed = dense_to_hybrid_block_sparse(weight, masks, config)

        self.assertEqual(tuple(packed.block_selector.shape), (3, 2, 2))
        self.assertTrue(
            torch.equal(packed.to_dense(), weight.masked_fill(masks, 0))
        )
        popcounts = torch.stack(
            [
                ((packed.block_selector >> bit) & 1)
                for bit in range(config.block_m)
            ],
            dim=-1,
        ).sum(dim=-1)
        self.assertTrue(torch.all(popcounts == config.block_n).item())

    def test_all_six_two_of_four_metadata_codes_round_trip(self):
        config = HybridBlockSparseConfig(
            block_h=1, block_w=24, block_n=1, block_m=1
        )
        weight = torch.arange(1, 25, dtype=torch.float32).reshape(1, 24)
        prune_mask = torch.ones_like(weight, dtype=torch.bool).reshape(1, 6, 4)
        pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        for quartet, pair in enumerate(pairs):
            prune_mask[0, quartet, pair[0]] = False
            prune_mask[0, quartet, pair[1]] = False
        prune_mask = prune_mask.reshape_as(weight)

        packed = dense_to_hybrid_block_sparse(weight, prune_mask, config)

        self.assertEqual(
            packed.sparse_metadata.flatten().tolist(), list(range(6))
        )
        self.assertTrue(
            torch.equal(packed.to_dense(), weight.masked_fill(prune_mask, 0))
        )

    def test_rejects_invalid_outer_topology(self):
        config = HybridBlockSparseConfig(
            block_h=2, block_w=4, block_n=1, block_m=2
        )
        weight = torch.randn(2, 8)
        prune_mask = torch.zeros_like(weight, dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "select exactly 1 sparse blocks"):
            dense_to_hybrid_block_sparse(weight, prune_mask, config)

    def test_rejects_invalid_inner_two_of_four_mask(self):
        config = HybridBlockSparseConfig(
            block_h=2, block_w=4, block_n=1, block_m=2
        )
        weight = torch.randn(2, 8)
        prune_mask = torch.zeros_like(weight, dtype=torch.bool)
        prune_mask[:, 3] = True

        with self.assertRaisesRegex(ValueError, "exactly 2:4"):
            dense_to_hybrid_block_sparse(weight, prune_mask, config)

    def test_rejects_corrupt_metadata(self):
        config = HybridBlockSparseConfig(
            block_h=2, block_w=4, block_n=1, block_m=2
        )
        weight = torch.randn(2, 8)
        importance = weight.abs()
        mask = build_hybrid_block_sparse_prune_mask(importance, config)
        packed = dense_to_hybrid_block_sparse(weight, mask, config)
        corrupt_metadata = packed.sparse_metadata.clone()
        corrupt_metadata.flatten()[0] = 6
        corrupt = HybridBlockSparseWeight(
            original_shape=packed.original_shape,
            config=packed.config,
            block_selector=packed.block_selector,
            dense_values=packed.dense_values,
            sparse_values=packed.sparse_values,
            sparse_metadata=corrupt_metadata,
        )

        with self.assertRaisesRegex(ValueError, "invalid 2:4 pair code"):
            corrupt.to_dense()

    def test_rejects_selector_bits_outside_group(self):
        config = HybridBlockSparseConfig(
            block_h=2, block_w=4, block_n=1, block_m=2
        )
        weight = torch.randn(2, 8)
        mask = build_hybrid_block_sparse_prune_mask(weight.abs(), config)
        packed = dense_to_hybrid_block_sparse(weight, mask, config)
        corrupt_selector = packed.block_selector.clone()
        corrupt_selector.flatten()[0] |= 1 << config.block_m
        corrupt = HybridBlockSparseWeight(
            original_shape=packed.original_shape,
            config=packed.config,
            block_selector=corrupt_selector,
            dense_values=packed.dense_values,
            sparse_values=packed.sparse_values,
            sparse_metadata=packed.sparse_metadata,
        )

        with self.assertRaisesRegex(ValueError, "outside the block group"):
            corrupt.to_dense()


if __name__ == "__main__":
    unittest.main()
