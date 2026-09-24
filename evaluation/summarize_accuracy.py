"""Summarize schema-v2 complete evaluations; values in percent, deltas in pp."""
import argparse
import csv
import json
from pathlib import Path


def summarize(results, tasks, dense_label=None):
    labels = [r["label"] for r in results]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate labels")
    for result in results:
        if result.get("schema_version") != 2:
            raise ValueError("expected schema-v2 result")
        if result["num_shards"] > 1 and result["shard_index"] is not None:
            raise ValueError("merge shards before summarizing")
    if dense_label and dense_label not in labels:
        raise ValueError("dense baseline label not found")
    baseline = results[labels.index(dense_label)] if dense_label else None
    if baseline:
        for result in results:
            protocol = {k: v for k, v in result["protocol"].items() if k != "checkpoint_digest"}
            reference = {k: v for k, v in baseline["protocol"].items() if k != "checkpoint_digest"}
            if protocol != reference:
                raise ValueError("incomparable evaluation protocols")
            if set(result["tasks"]) != set(baseline["tasks"]):
                raise ValueError("incomparable task coverage")
            for name, task in result["tasks"].items():
                for key in ("dataset_digest", "config_digest", "total", "metric", "group"):
                    if task[key] != baseline["tasks"][name][key]:
                        raise ValueError("incomparable datasets/task settings")
    rows = []
    for result in results:
        row = {"label": result["label"]}
        for task in tasks:
            value = result["scores"].get(task)
            row[task + "_percent"] = "" if value is None else value * 100
            if baseline:
                dense = baseline["scores"].get(task)
                row[task + "_delta_pp"] = "" if value is None or dense is None else (value - dense) * 100
        rows.append(row)
    return rows


def main(default_tasks=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dense-label")
    p.add_argument("--tasks", nargs="+", default=default_tasks or ["gsm8k", "math500", "mmlu", "asdiv", "average"])
    args = p.parse_args()
    rows = summarize([json.loads(path.read_text()) for path in args.inputs], args.tasks, args.dense_label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
