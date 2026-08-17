# Scripts

Utility scripts for pruning, checkpoint conversion, kernel builds, and
experiment orchestration.

`export_moe_sparse_gemm.py` exports Qwen1.5-MoE, DeepSeek-V2-Lite, Qwen3 MoE,
and Llama 4 checkpoints. Individual `gate_proj`/`up_proj`/`down_proj` experts
and Llama 4 fused `gate_up_proj`/`down_proj` tensors are normalized to the same
SGLang SparseGEMM manifest layout.

```bash
python scripts/export_moe_sparse_gemm.py \
  --model-dir /path/to/huggingface-checkpoint \
  --output-dir /tmp/sparse-gemm-export \
  --no-download --num-workers 4
```
