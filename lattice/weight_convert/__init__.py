"""Compatibility exports for SparseGEMM-owned weight conversion."""

from sparse_gemm.hybrid_sparse import (
    HybridBlockSparseWeight,
    dense_to_hybrid_block_sparse,
    hybrid_block_sparse_to_dense,
)

__all__ = [
    "HybridBlockSparseWeight",
    "dense_to_hybrid_block_sparse",
    "hybrid_block_sparse_to_dense",
]
