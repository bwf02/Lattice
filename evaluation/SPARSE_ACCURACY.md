# Sparse Accuracy Evaluation

This experiment evaluates Qwen1.5-MoE-A2.7B after Wanda pruning of routed
experts only. It uses seed 0 and 128 WikiText-2 calibration samples.

## Configurations

| Label | Pattern | Target sparsity |
|---|---|---:|
| `dense` | dense | 0% |
| `unstructured_25` | unstructured | 25% |
| `unstructured_50` | unstructured | 50% |
| `unstructured_75` | unstructured | 75% |
| `nm_2_4` | keep 2 of 4 | 50% |
| `nm_1_4` | keep 1 of 4 | 75% |

Accuracy is reported on GSM8K and MATH-500, followed by their unweighted
arithmetic mean. HumanEval is disabled by default because its long code
generations dominate the evaluation time; set `SKIP_HUMANEVAL=0` to include
greedy pass@1 when needed.

## Setup

```bash
bash evaluation/setup_sparse_accuracy.sh
```

The default model is
`/ossfs/workspace/Lattice/models/Qwen1.5-MoE-A2.7B`. The Python environment,
datasets, logs, and results are stored under `/tmp`. Existing model files are
reused and are not downloaded again.

## Run

```bash
OUTPUT_DIR=/tmp/sparse_accuracy_results \
  bash evaluation/run_sparse_accuracy_suite.sh
```

The suite uses GPUs 0 through 5, one configuration per GPU. Batch size 16 is
the measured default on H20; override it with `BATCH_SIZE=<value>`. Use
`LIMIT=<count>` and `NSAMPLES=<count>` for smoke tests. To rerun a subset into
an existing output directory, set a comma-separated list such as
`CONFIGS=unstructured_25,unstructured_50,nm_2_4`.

The HB-N:M suite evaluates the paper's three hybrid patterns with 16x16 blocks:

```bash
bash evaluation/run_hb_nm_accuracy_suite.sh
```

The 25% pattern makes one of every two blocks 2:4 sparse, the 50% pattern makes
all blocks 2:4 sparse, and the 75% pattern keeps one of every two blocks and
applies 2:4 within each retained block. All three use Wanda calibration on
routed experts and skip HumanEval.

Each configuration writes a JSON result and log. Successful completion also
writes `summary.md` and `summary.csv` in `OUTPUT_DIR`.

If `/tmp` runs out of space, only previously validated exported weights under
`/tmp/sparse_gemm_export_nm12_shared/Qwen1.5-MoE-A2.7B` may be removed. Do not
remove the original model, datasets, environments, logs, or unrelated files.
