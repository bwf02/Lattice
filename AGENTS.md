# Global Codex Rules

## Git Operations

- `git add`, `git commit`, and `git push` require explicit human approval before running.
- Commit messages must use Conventional Commits, for example:
  - `fix: ...`
  - `feat: ...`
  - `docs: ...`
  - `test: ...`
- Commit logs must be written in English.
- Commit bodies should briefly explain why the change was made and what validation or tests were run, using one or two natural-language sentences without required labels.

## Repository Setup

```bash
git clone https://github.com/bwf02/MosaicMoE.git
cd MosaicMoE
git checkout mosaic_moe_dev_eval
```

## Evaluation Usage

Install the evaluation dependencies from the repository root:

```bash
python -m venv /tmp/mosaicmoe-eval-venv
source /tmp/mosaicmoe-eval-venv/bin/activate
pip install -r requirements.txt
```

Run the complete setup, model download, and smoke evaluation pipeline from the
repository root:

```bash
./evaluation/eval.sh smoke
```

Run the full benchmark suite with the same setup:

```bash
./evaluation/eval.sh full
```

Run Wanda from its own directory. In this project, `N:M` denotes the number of
nonzero weights kept, while `--prune_n` denotes the number of weights pruned.

```bash
cd evaluation/wanda
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model Qwen/Qwen2.5-14B \
  --prune_method wanda \
  --sparsity_type 6:8 \
  --prune_n 2 --prune_m 8 \
  --save /path/to/results/wanda_6_8 \
  --save_model /path/to/checkpoints/wanda_6_8
```

Run SparseGPT from its own directory:

```bash
cd evaluation/sparsegpt
CUDA_VISIBLE_DEVICES=0 python llama.py Qwen/Qwen2.5-7B wikitext2 \
  --prunen 2 --prunem 8 \
  --save /path/to/checkpoints/sparsegpt_6_8
```

Run the benchmark suite from the repository root after producing a checkpoint:

```bash
./evaluation/run_eval.sh /path/to/checkpoint /path/to/results 0 auto
```
