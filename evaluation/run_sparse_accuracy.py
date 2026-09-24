#!/usr/bin/env python3
"""Reconstructed routed-MoE accuracy workflow. See ACCURACY_REPRODUCTION.md."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.accuracy_io import digest, partition, scores_from_tasks, write_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Local checkpoint directory")
    p.add_argument("--output", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--sparsity-type", choices=["dense", "hibnm", "hybrid_block_sparse", "unstructured", "2:4", "2:8"], default="dense")
    p.add_argument("--sparsity-ratio", type=float)
    p.add_argument("--prune-method", choices=["wanda", "magnitude"], default="wanda")
    p.add_argument("--block-h", type=int, default=64)
    p.add_argument("--block-w", type=int, default=64)
    p.add_argument("--block-n", type=int, default=1)
    p.add_argument("--block-m", type=int, default=2)
    p.add_argument("--hybrid-block-score", choices=["sum", "squared_sum", "max_row_squared"], default="max_row_squared")
    p.add_argument("--nsamples", type=int, default=32)
    p.add_argument("--calibration-length", type=int, default=2048)
    p.add_argument("--calibration-file", help="Train-only JSON/Parquet with a text column")
    p.add_argument("--uncalibrated", choices=["error", "magnitude"], default="error")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--save-model")
    p.add_argument("--prune-only", action="store_true")
    p.add_argument("--skip-pruning", action="store_true")
    p.add_argument("--tasks", nargs="+", choices=["gsm8k", "math500", "mmlu", "asdiv"], default=["gsm8k", "math500", "mmlu", "asdiv"])
    p.add_argument("--task-config-dir", help="Optional lm-eval YAML task overrides (e.g. local datasets)")
    p.add_argument("--batch-size", default="1")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--limit", type=int, help="Global per-leaf-task smoke limit, before sharding")
    p.add_argument("--save-samples", action="store_true")
    p.add_argument("--ppl-file", help="Optional WikiText-2 test Parquet, text column")
    p.add_argument("--ppl-max-length", type=int, default=2048)
    args = p.parse_args(argv)
    partition(0, args.shard_index, args.num_shards)
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    if args.ppl_max_length < 2:
        p.error("--ppl-max-length must be >= 2")
    if args.prune_only and (not args.save_model or args.num_shards != 1):
        p.error("--prune-only requires --save-model and a single process")
    if args.num_shards > 1 and not args.skip_pruning and args.sparsity_type != "dense":
        p.error("prune once, save, then use --skip-pruning on every evaluation shard")
    if args.save_model and args.num_shards > 1:
        p.error("evaluation shards must not write a shared checkpoint")
    return args


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_files(path):
    root = Path(path)
    files = sorted(p for p in root.iterdir() if p.is_file() and
                   (p.suffix in {".safetensors", ".bin", ".json", ".model", ".txt", ".jinja"})
                   and p.name != "lattice_pruning.json")
    if not (root / "config.json").is_file() or not any(p.suffix in {".bin", ".safetensors"} for p in files):
        raise ValueError("--model must be a complete local HF checkpoint")
    return {p.name: file_hash(p) for p in files}


def task_group(name):
    if name.startswith("mmlu_"):
        return "mmlu"
    return {"minerva_math500": "math500"}.get(name, name)


def task_metric(group):
    return {"gsm8k": "exact_match,strict-match", "math500": "math_verify,none",
            "mmlu": "acc,none", "asdiv": "acc,none"}[group]


def evaluate_tasks(model, tokenizer, args):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager, get_task_dict
    if "samples" not in inspect.signature(simple_evaluate).parameters:
        raise RuntimeError("lm-eval version lacks explicit sample-index selection")
    names = ["minerva_math500" if t == "math500" else t for t in args.tasks]
    tree = get_task_dict(names, task_manager=TaskManager(include_path=args.task_config_dir))
    leaves = {}
    def visit(branch):
        for key, task in branch.items():
            if isinstance(task, dict):
                visit(task)
            else:
                if isinstance(task, tuple):
                    task = task[-1]
                leaves[str(task.config.task)] = task
    visit(tree)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    records, samples = {}, {}
    for name, task in sorted(leaves.items()):
        group = task_group(name)
        metric = task_metric(group)
        # Fix few-shot settings explicitly; never use test samples for few-shot context.
        fewshot = 5 if group in {"gsm8k", "mmlu"} else 0
        task.set_config("num_fewshot", fewshot)
        if fewshot:
            from lm_eval.api.samplers import FirstNSampler
            fewshot_docs = list(task.fewshot_docs())
            if len(fewshot_docs) < fewshot:
                raise ValueError(f"{name}: insufficient few-shot examples")
            task.sampler = FirstNSampler(fewshot_docs)
        docs = task.eval_docs
        total = min(len(docs), args.limit) if args.limit else len(docs)
        ids = partition(total, args.shard_index, args.num_shards)
        # Hash content, not just length; changed datasets must not merge silently.
        dataset_digest = digest([docs[i] for i in range(total)])
        config_digest = digest({"task": task.config.to_dict(),
            "fewshot_policy": "first_n", "fewshot_examples": fewshot_docs[:fewshot] if fewshot else []})
        score_sum = 0.
        if ids:  # An empty samples list means ALL documents in some harness versions.
            result = simple_evaluate(model=lm, tasks=[task], samples={name: ids},
                bootstrap_iters=0, log_samples=True, random_seed=args.seed,
                numpy_random_seed=args.seed, torch_random_seed=args.seed,
                fewshot_random_seed=args.seed, confirm_run_unsafe_code=False)
            observed = result["samples"][name]
            if len(observed) != len(ids):
                raise ValueError(f"{name}: evaluated sample count mismatch")
            score = float(result["results"][name][metric])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"{name}: invalid accuracy")
            score_sum = score * len(ids)
            if args.save_samples:
                samples[name] = {"source_ids": ids, "harness_samples": observed}
        records[name] = {"group": group, "metric": metric, "ids": ids,
            "total": total, "count": len(ids), "sum": score_sum,
            "dataset_digest": dataset_digest, "config_digest": config_digest}
    if not records:
        raise ValueError("no evaluation tasks resolved")
    return records, samples


def evaluate_ppl(model, tokenizer, args):
    import torch
    from datasets import Dataset
    data = Dataset.from_parquet(args.ppl_file)
    ids = tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids
    windows = [ids[:, i:i + args.ppl_max_length] for i in range(0, ids.shape[1], args.ppl_max_length)]
    windows = [w for w in windows if w.shape[1] > 1]
    selected = partition(len(windows), args.shard_index, args.num_shards)
    nll, tokens = 0., 0
    device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        for index in selected:
            window = windows[index].to(device)
            predicted = window.shape[1] - 1
            nll += float(model(window, labels=window).loss) * predicted
            tokens += predicted
    return {"nll": nll, "tokens": tokens, "value": math.exp(nll/tokens) if tokens else None,
            "total_windows": len(windows), "ids": selected}


def main():
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from evaluation.data import calibration_batches
    from evaluation.pruning import Recipe, prune_model
    set_seed(args.seed)
    pattern = "hibnm" if args.sparsity_type == "hybrid_block_sparse" else args.sparsity_type
    default_ratio = {"dense": 0., "2:4": .5, "2:8": .75, "unstructured": .25,
                     "hibnm": args.block_n / (2 * args.block_m) if args.block_m else -1}[pattern]
    recipe = Recipe(pattern, args.prune_method, args.sparsity_ratio if args.sparsity_ratio is not None else default_ratio,
                    args.block_h, args.block_w, args.block_n, args.block_m, args.hybrid_block_score)
    recipe.validate()
    source_files = checkpoint_files(args.model)
    manifest_path = Path(args.model) / "lattice_pruning.json"
    manifest = None
    if args.skip_pruning:
        if not manifest_path.is_file():
            raise ValueError("--skip-pruning requires a lattice_pruning.json export manifest")
        manifest = json.loads(manifest_path.read_text())
        if manifest["files"] != source_files:
            raise ValueError("checkpoint content differs from pruning manifest")
    elif manifest_path.exists() and pattern != "dense":
        raise ValueError("checkpoint already pruned; use --skip-pruning to avoid double pruning")
    elif manifest_path.exists():
        raise ValueError("a pruned checkpoint cannot be evaluated as dense")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype),
        device_map=args.device_map, attn_implementation="eager", trust_remote_code=args.trust_remote_code)
    if hasattr(model, "set_experts_implementation"):
        model.set_experts_implementation("eager")
    model.eval()
    calibration_digest = hashlib.sha256()
    def calibration():
        batches = calibration_batches(tokenizer, samples=args.nsamples, length=args.calibration_length,
            seed=args.seed, device=model.get_input_embeddings().weight.device, path=args.calibration_file)
        for batch in batches:
            calibration_digest.update(batch["input_ids"].cpu().numpy().tobytes())
            yield batch
    if not args.skip_pruning:
        records = prune_model(model, recipe, calibration(), args.uncalibrated)
        manifest = {"schema_version": 2, "recipe": asdict(recipe), "source_files": source_files,
            "calibration": {"samples": args.nsamples, "length": args.calibration_length,
                "seed": args.seed, "token_digest": calibration_digest.hexdigest(),
                "uncalibrated": args.uncalibrated}, "projections": records}
        if args.save_model:
            target = Path(args.save_model)
            if target.exists() and any(target.iterdir()):
                raise ValueError("save directory must be empty; refusing to overwrite checkpoint")
            model.save_pretrained(target, safe_serialization=True)
            tokenizer.save_pretrained(target)
            manifest["files"] = checkpoint_files(target)
            write_json(target / "lattice_pruning.json", manifest)
    if args.prune_only:
        write_json(args.output, manifest)
        return
    versions = {p: importlib.metadata.version(p) for p in ("torch", "transformers", "lm_eval", "datasets")}
    protocol = {"checkpoint_digest": digest(manifest), "dtype": args.dtype, "seed": args.seed,
        "tasks": sorted(set(args.tasks)), "limit": args.limit, "versions": versions,
        "batch_size": args.batch_size, "chat_template": False, "fewshot_policy": "first_n",
        "ppl_digest": file_hash(args.ppl_file) if args.ppl_file else None,
        "ppl_max_length": args.ppl_max_length}
    tasks, samples = evaluate_tasks(model, tokenizer, args)
    result = {"schema_version": 2, "label": args.label, "protocol": protocol,
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "pruning": manifest, "tasks": tasks, "scores": scores_from_tasks(tasks),
        "ppl": evaluate_ppl(model, tokenizer, args) if args.ppl_file else None}
    write_json(args.output, result)
    if args.save_samples:
        write_json(Path(args.output).with_suffix(".samples.json"), samples)
    print(json.dumps(result["scores"], indent=2))


if __name__ == "__main__":
    main()
