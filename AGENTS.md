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

## Remote Validation Workflow

Use this sequence for implementation and validation:

1. Modify and run static checks in the local repository.
2. Review the local diff, then commit and push after explicit approval.
3. Connect to the current SSH port provided by the user; the port may change.
4. Update `/ossfs/workspace/MosaicMoE` with `git pull --ff-only`.
5. Run unit tests and model smoke tests in the remote GPU environment.

```bash
ssh -p <current-port> 127.0.0.1
cd /ossfs/workspace/MosaicMoE
git pull --ff-only origin mosaic_moe_dev_eval
```

## Evaluation Usage

Install the evaluation dependencies from the repository root:

```bash
uv venv /tmp/mosaicmoe-eval-venv \
  --clear --system-site-packages --python /usr/bin/python3
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

Run HB-N:M Wanda pruning on Qwen1.5-MoE routed experts. `block_n:block_m`
controls inter-block retention, while each retained block uses intra-block 2:4:

```bash
cd evaluation/wanda
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --prune_method wanda \
  --sparsity_type hb_nm \
  --block_h 16 --block_w 16 \
  --block_n 1 --block_m 2 \
  --nsamples 16 \
  --save /path/to/results/hb_nm \
  --save_model /path/to/checkpoints/hb_nm
```

Run the HB-N:M unit tests from the repository root:

```bash
python -m unittest discover -s evaluation/tests -v
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
