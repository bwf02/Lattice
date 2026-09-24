# LATTICE accuracy evaluation

This is a reconstructed implementation of the missing accuracy workflow, not
the recovered source of the historical experiments. Do not substitute its
results for archived paper numbers without rerunning and validating the protocol.
The archived runner is preserved as `run_sparse_accuracy_legacy.py`; old suite
scripts still call it. New experiments should use the commands below.

## Supported paths

- Qwen1.5/Qwen3 routed experts represented by `ModuleList` Linear projections.
- Qwen3 stacked expert tensors `[experts, 2*intermediate, hidden]` and
  `[experts, hidden, intermediate]`, using Transformers' eager expert execution.
- Dense, row-wise unstructured, HiBNM, uniform 2:4 and uniform 2:8 masks.
  `2:8` means **keep two of eight**, hence 75% sparsity.
- Wanda activation-weighted importance or magnitude importance. HiBNM uses
  `max_row_squared` (MRSS) by default; `sum` and `squared_sum` are ablation options.
- GSM8K strict-match, MATH-500 math-verify, MMLU 5-shot accuracy, and ASDiv accuracy
  through the pinned lm-eval task implementations. MMLU is micro-averaged over
  all subject examples, not an unweighted mean of subjects or GPU shards.
- Optional WikiText-2 PPL, merged by total negative log-likelihood / predicted
  tokens, not by averaging perplexities.

Only routed `mlp.experts` projections are pruned. Router, attention and shared
experts remain untouched. Unsupported expert layouts fail explicitly.
For Wanda, a dense-model calibration pass collects input-channel squared
activation sums before any masks are applied. Stacked experts collect energies
for the tokens actually routed to each expert; down-projection inputs are
computed from that expert's gate/up activation. This is **not** sequential
layer-by-layer SparseGPT compensation, nor a claim of identical calibration to
the missing historical implementation.

## Install

From the Lattice repository root, on a Linux GPU host with a compatible driver:

```bash
python3.11 -m venv .venv-accuracy
source .venv-accuracy/bin/activate
# Install the appropriate CUDA-enabled PyTorch build for your host first.
python -m pip install -r evaluation/requirements-accuracy.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()'
```

Use a separate environment from SGLang. Checkpoint weights, tokenizer files and
datasets are not bundled. Obtain the model under its license and pass a local
checkpoint directory. Remote model code is disabled unless explicitly enabled
with `--trust-remote-code`. The runner uses BF16 by default; CPU smoke tests can
use `--dtype float32 --device-map cpu`.

The task harness downloads datasets on first use. For offline operation, prepare
its dataset cache beforehand, or supply `--task-config-dir /path/to/yamls` with
lm-eval task overrides pointing to local data while preserving task names.
Calibration reads only the WikiText-2 **training** split. To avoid a download,
provide `--calibration-file /path/to/train.parquet` (or JSON with a `text` column).

## 1. Prune once and save a checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/run_sparse_accuracy.py \
  --model /models/Qwen3-30B-A3B \
  --label hibnm25 --sparsity-type hibnm --sparsity-ratio 0.25 \
  --prune-method wanda --hybrid-block-score max_row_squared \
  --block-h 64 --block-w 64 --block-n 1 --block-m 2 \
  --nsamples 32 --calibration-length 2048 --seed 0 \
  --prune-only --save-model results/checkpoints/hibnm25 \
  --output results/accuracy/hibnm25-pruning.json
```

`--save-model` must point to an empty directory. The checkpoint includes
`lattice_pruning.json`, recording the source/checkpoint file hashes, recipe,
calibration-token digest, actual zero counts, and per-projection coverage.
Loading it with `--skip-pruning` verifies its files before evaluation. This
hashing reads all weights and can take time, but prevents mixing checkpoints.

Experts receiving no calibration tokens cause an error **before any weights are
modified**. Increase calibration coverage, or explicitly choose
`--uncalibrated magnitude`; every such fallback is recorded. Never silently
treat unobserved expert importance as zero.

To create comparison checkpoints, change the label, output paths and pattern:

| Pattern | Arguments |
|---|---|
| Unstructured 25% | `--sparsity-type unstructured --sparsity-ratio 0.25` |
| Uniform 2:4 | `--sparsity-type 2:4 --sparsity-ratio 0.50` |
| Uniform 2:8 | `--sparsity-type 2:8 --sparsity-ratio 0.75` |

Use `--prune-method magnitude` for a magnitude-only baseline; keep all remaining
settings fixed. HiBNM's `block_n/block_m` selects the fraction of blocks made
2:4 sparse, so overall sparsity is `block_n / (2*block_m)`, at most 50%.
The old `hb_nm` pattern is not an alias for HiBNM in the new runner.

## 2. Evaluate the same checkpoint on multiple GPUs

Each GPU holds a full replica. This is data parallelism, not tensor parallelism;
ensure each visible GPU can hold the checkpoint and evaluation workload.

```bash
python evaluation/run_accuracy_multigpu.py \
  --gpus 0,1,2,3 --output-dir results/accuracy/hibnm25 \
  -- --model results/checkpoints/hibnm25 --skip-pruning --label hibnm25 \
  --tasks gsm8k math500 mmlu asdiv --batch-size 1 --seed 0
```

For a dense baseline, use the original model and omit `--skip-pruning`:

```bash
python evaluation/run_accuracy_multigpu.py \
  --gpus 0,1,2,3 --output-dir results/accuracy/dense \
  -- --model /models/Qwen3-30B-A3B --sparsity-type dense --label dense \
  --tasks gsm8k math500 mmlu asdiv --batch-size 1 --seed 0
```

Start with `--limit 8` for a smoke test (eight examples **per leaf task**, before
partitioning), then remove it for full evaluation. Do not compare smoke and full
results. For one GPU, call `run_sparse_accuracy.py` directly with `--output`.
Add `--save-samples` to retain prompts, outputs and scorer records; these may
contain licensed dataset text, so they are not automatically committed.

GSM8K and MMLU use five examples from the start of their task-defined few-shot
split; MATH-500 and ASDiv use zero shots. This explicit `first_n` policy makes
prompts independent of shard count. No chat template is applied. Generation
limits, stop strings and scoring come from the pinned task YAMLs and are included
in the task-config digest. These choices must be reviewed against any historical
protocol before comparing with historical scores.

The launcher writes one log/JSON per shard and `merged.json` only if all workers
succeed. It refuses a nonempty output directory. Manual merge is also supported:

```bash
python evaluation/merge_accuracy_shards.py results/accuracy/hibnm25/shard-*.json \
  --output results/accuracy/hibnm25/merged.json
```

Merge checks shard identities, complete/disjoint sample coverage, checkpoint and
protocol identity, dataset contents, task configuration, and score bounds.
Empty shards are retained and never interpreted as “evaluate everything.”
Do not merge schema-v2 outputs with older archived JSON formats.

Optional PPL uses `--ppl-file /path/to/wikitext-test.parquet`. Windows are
non-overlapping; the first token of every window is not predicted. This policy
and the test-file digest are recorded. PPL is not included in the accuracy mean.

## 3. Summarize

```bash
python evaluation/summarize_accuracy.py \
  results/accuracy/dense/merged.json results/accuracy/hibnm25/merged.json \
  --dense-label dense --output results/accuracy/summary.csv
python evaluation/summarize_mmlu_accuracy.py \
  results/accuracy/dense/merged.json results/accuracy/hibnm25/merged.json \
  --dense-label dense --output results/accuracy/mmlu.csv
```

Scores are percentages; dense-relative deltas are percentage points. Missing
tasks remain blank, not zero. `average` is the arithmetic mean of evaluated
benchmark scores, with MMLU first aggregated across its subjects. Comparisons
reject different protocols or dataset/task fingerprints.

## Tests and validation boundary

```bash
python -m pytest evaluation/tests -q
```

Tests include mask ratios/ties, protected weights, real tiny Qwen3 forward passes,
calibration cleanup, missing/duplicate shards, unequal subject sizes, PPL merging,
CSV deltas, and offline integration with the actual lm-eval evaluator. CPU tests
do not validate full Qwen3 accuracy, GPU memory usage, or reproduce paper results.
