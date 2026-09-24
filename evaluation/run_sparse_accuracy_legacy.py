#!/usr/bin/env python3
"""Prune routed MoE experts and evaluate reasoning/code accuracy in memory."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


ROOT_DIR = Path(__file__).resolve().parents[1]
WANDA_DIR = ROOT_DIR / "evaluation" / "wanda"
sys.path.insert(0, str(WANDA_DIR))

from lib.prune import check_sparsity, prune_wanda  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-root", default="/tmp/lattice-eval-datasets")
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--sparsity-type", default="dense")
    parser.add_argument("--sparsity-ratio", type=float, default=0.0)
    parser.add_argument("--prune-n", type=int, default=0, help="Weights pruned per M")
    parser.add_argument("--prune-m", type=int, default=0)
    parser.add_argument("--block-h", type=int, default=16)
    parser.add_argument("--block-w", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=1)
    parser.add_argument("--block-m", type=int, default=2)
    parser.add_argument("--hybrid-block-score", default="sum")
    parser.add_argument("--kept-pattern", default=None)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--humaneval-batch-size", type=int, default=16)
    parser.add_argument("--ppl-max-length", type=int, default=2048)
    parser.add_argument("--ppl-max-tokens", type=int, default=None)
    parser.add_argument("--ppl-only", action="store_true")
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--skip-humaneval", action="store_true")
    parser.add_argument("--save-samples", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_local_task_configs(dataset_root):
    import lm_eval
    from lm_eval.utils import load_yaml_config

    task_root = Path(lm_eval.__file__).parent / "tasks"
    specs = [
        (
            "gsm8k",
            task_root / "gsm8k" / "gsm8k.yaml",
            "parquet",
            {
                "train": str(dataset_root / "gsm8k_data/main/train-00000-of-00001.parquet"),
                "test": str(dataset_root / "gsm8k_data/main/test-00000-of-00001.parquet"),
            },
        ),
        (
            "minerva_math500",
            task_root / "minerva_math" / "minerva_math500.yaml",
            "json",
            {"test": str(dataset_root / "math_500/test.jsonl")},
        ),
    ]
    configs = []
    for task_name, yaml_path, dataset_path, data_files in specs:
        for data_file in data_files.values():
            if not Path(data_file).is_file():
                raise FileNotFoundError(data_file)
        config = load_yaml_config(yaml_path=yaml_path)
        config.update(
            task=task_name,
            dataset_path=dataset_path,
            dataset_name=None,
            dataset_kwargs={"data_files": data_files},
        )
        configs.append(config)
    return configs


def metric_value(results, task, metric, filter_name=None):
    task_results = results["results"][task]
    matches = [
        value
        for key, value in task_results.items()
        if key.split(",", 1)[0] == metric
        and (filter_name is None or key.split(",", 1)[-1] == filter_name)
    ]
    if len(matches) != 1:
        suffix = f" ({filter_name})" if filter_name else ""
        raise KeyError(f"Expected one {metric}{suffix} metric for {task}, got {task_results}")
    return float(matches[0])


def run_python_test(source, timeout=5):
    with tempfile.TemporaryDirectory(prefix="humaneval-") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(source)
        try:
            completed = subprocess.run(
                [sys.executable, str(path)],
                cwd=directory,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            )
            return completed.returncode == 0
        except subprocess.TimeoutExpired:
            return False


class StopOnText(StoppingCriteria):
    def __init__(self, tokenizer, stop_sequences, prompt_length):
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        tail_start = max(self.prompt_length, input_ids.shape[1] - 16)
        stopped = []
        for row in input_ids:
            tail = self.tokenizer.decode(row[tail_start:], skip_special_tokens=False)
            stopped.append(any(stop in tail for stop in self.stop_sequences))
        return torch.tensor(stopped, dtype=torch.bool, device=input_ids.device)


def evaluate_wikitext2_ppl(model, tokenizer, dataset_root, max_length=2048, max_tokens=None):
    from datasets import Dataset

    path = dataset_root / "wikitext_salesforce/wikitext-2-raw-v1/test-00000-of-00001.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    dataset = Dataset.from_parquet(str(path))
    text = "\n\n".join(dataset["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    if max_tokens is not None:
        input_ids = input_ids[:, :max_tokens]

    total_nll = 0.0
    total_tokens = 0
    for start in range(0, input_ids.shape[1], max_length):
        window = input_ids[:, start : start + max_length].to(model.device)
        if window.shape[1] < 2:
            continue
        with torch.inference_mode():
            loss = model(window, labels=window).loss
        predicted_tokens = window.shape[1] - 1
        total_nll += float(loss) * predicted_tokens
        total_tokens += predicted_tokens
        print(
            f"PPL tokens {min(start + max_length, input_ids.shape[1])}/{input_ids.shape[1]}",
            flush=True,
        )
    return float(np.exp(total_nll / total_tokens)), total_tokens


def evaluate_humaneval(
    model, tokenizer, dataset_root, limit=None, batch_size=16, save_samples=False
):
    from datasets import Dataset

    path = (
        dataset_root
        / "humaneval/openai_humaneval/test-00000-of-00001.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    dataset = Dataset.from_parquet(str(path))
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    stop_sequences = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
    passed = 0
    sample_records = []
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    for batch_start in range(0, len(dataset), batch_size):
        docs = [
            dataset[index]
            for index in range(batch_start, min(batch_start + batch_size, len(dataset)))
        ]
        inputs = tokenizer(
            [doc["prompt"] for doc in docs], return_tensors="pt", padding=True
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=1024,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList(
                    [
                        StopOnText(
                            tokenizer, stop_sequences, inputs.input_ids.shape[1]
                        )
                    ]
                ),
            )
        for offset, doc in enumerate(docs):
            completion = tokenizer.decode(
                generated[offset, inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            stop_positions = [
                completion.find(stop) for stop in stop_sequences if stop in completion
            ]
            if stop_positions:
                completion = completion[: min(stop_positions)]
            source = (
                doc["prompt"]
                + completion
                + "\n"
                + doc["test"]
                + f"\ncheck({doc['entry_point']})\n"
            )
            is_correct = run_python_test(source)
            passed += int(is_correct)
            index = batch_start + offset
            print(f"HumanEval {index + 1}/{len(dataset)} passed={passed}", flush=True)
            if save_samples:
                sample_records.append(
                    {
                        "task_id": doc.get("task_id"),
                        "prompt": doc["prompt"],
                        "completion": completion,
                        "entry_point": doc["entry_point"],
                        "passed": is_correct,
                    }
                )
    return passed / len(dataset), sample_records


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=False, trust_remote_code=True
    )
    model.seqlen = min(model.config.max_position_embeddings, 2048)
    model.eval()

    if args.sparsity_type != "dense":
        prune_args = SimpleNamespace(
            sparsity_type=args.sparsity_type,
            sparsity_ratio=args.sparsity_ratio,
            prune_method="wanda",
            routed_experts_only=True,
            nsamples=args.nsamples,
            seed=args.seed,
            use_variant=False,
            block_h=args.block_h,
            block_w=args.block_w,
            block_n=args.block_n,
            block_m=args.block_m,
            hybrid_block_score=args.hybrid_block_score,
        )
        prune_wanda(
            prune_args,
            model,
            tokenizer,
            torch.device("cuda:0"),
            prune_n=args.prune_n,
            prune_m=args.prune_m,
        )

    actual_sparsity = check_sparsity(model, routed_experts_only=True)

    ppl = None
    ppl_tokens = None
    if not args.skip_ppl:
        ppl, ppl_tokens = evaluate_wikitext2_ppl(
            model,
            tokenizer,
            Path(args.dataset_root),
            args.ppl_max_length,
            args.ppl_max_tokens,
        )

    scores = {}
    task_samples = {}
    humaneval_samples = []
    if not args.ppl_only:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM

        task_configs = load_local_task_configs(Path(args.dataset_root))
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_batch_size=64,
        )
        results = simple_evaluate(
            model=lm,
            tasks=task_configs,
            limit=args.limit,
            bootstrap_iters=0,
            log_samples=args.save_samples,
            confirm_run_unsafe_code=True,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
        )
        scores = {
            "gsm8k": metric_value(results, "gsm8k", "exact_match", "flexible-extract"),
            "math500": metric_value(results, "minerva_math500", "math_verify"),
        }
        if not args.skip_humaneval:
            humaneval, humaneval_samples = evaluate_humaneval(
                model,
                tokenizer,
                Path(args.dataset_root),
                args.limit,
                args.humaneval_batch_size,
                args.save_samples,
            )
            scores["humaneval"] = humaneval
        scores["average"] = sum(scores.values()) / len(scores)
        task_samples = results.get("samples", {})
    output = {
        "label": args.label,
        "model": args.model,
        "prune_method": "dense" if args.sparsity_type == "dense" else "wanda",
        "scope": "routed_experts_only",
        "sparsity_type": args.sparsity_type,
        "target_sparsity": args.sparsity_ratio,
        "actual_sparsity": actual_sparsity,
        "kept_pattern": args.kept_pattern,
        "prune_n": args.prune_n,
        "prune_m": args.prune_m,
        "block_h": args.block_h,
        "block_w": args.block_w,
        "block_n": args.block_n,
        "block_m": args.block_m,
        "hybrid_block_score": args.hybrid_block_score,
        "nsamples": args.nsamples,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "humaneval_batch_size": args.humaneval_batch_size,
        "ppl": ppl,
        "ppl_tokens": ppl_tokens,
        "ppl_max_length": args.ppl_max_length,
        "scores": scores,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    if args.save_samples:
        samples_path = output_path.with_suffix(".samples.json")
        samples_path.write_text(
            json.dumps(
                {"lm_eval": task_samples, "humaneval": humaneval_samples},
                indent=2,
                default=str,
            )
            + "\n"
        )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
