"""Dependency-free provenance, partitioning, and strict result merging."""
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str,
                                   allow_nan=False) + "\n")
    temporary.replace(path)


def partition(total, index, count):
    if total < 0 or count < 1 or not 0 <= index < count:
        raise ValueError("invalid shard dimensions")
    return list(range(index, total, count))


def scores_from_tasks(tasks):
    sums, counts = {}, {}
    for task in tasks.values():
        group = task["group"]
        n = task["count"]
        sums[group] = sums.get(group, 0.) + task["sum"]
        counts[group] = counts.get(group, 0) + n
    scores = {group: sums[group] / n for group, n in counts.items() if n}
    if scores:
        scores["average"] = sum(scores.values()) / len(scores)
    return scores


def merge_shards(shards):
    if not shards:
        raise ValueError("no shards")
    first = shards[0]
    count = first["num_shards"]
    if len(shards) != count or sorted(s["shard_index"] for s in shards) != list(range(count)):
        raise ValueError("missing or duplicate shard")
    for shard in shards:
        if shard.get("schema_version") != 2 or shard["num_shards"] != count:
            raise ValueError("incompatible schema or shard count")
        if shard["protocol"] != first["protocol"] or shard["label"] != first["label"]:
            raise ValueError("checkpoint/evaluation protocol mismatch")
        if set(shard["tasks"]) != set(first["tasks"]):
            raise ValueError("task coverage mismatch")
    tasks = {}
    for name, template in first["tasks"].items():
        ids, total_sum = [], 0.
        for shard in shards:
            task = shard["tasks"][name]
            for key in ("total", "dataset_digest", "metric", "group", "config_digest"):
                if task[key] != template[key]:
                    raise ValueError(f"{name}: mismatched {key}")
            expected = partition(task["total"], shard["shard_index"], count)
            if task["ids"] != expected or task["count"] != len(expected):
                raise ValueError(f"{name}: incorrect sample partition")
            if not math.isfinite(task["sum"]) or not 0 <= task["sum"] <= task["count"]:
                raise ValueError(f"{name}: invalid accuracy numerator")
            ids.extend(task["ids"])
            total_sum += task["sum"]
        if sorted(ids) != list(range(template["total"])):
            raise ValueError(f"{name}: duplicate or missing samples")
        tasks[name] = dict(template, ids=list(range(template["total"])),
                           count=len(ids), sum=total_sum)
    result = dict(first, shard_index=None, tasks=tasks,
                  scores=scores_from_tasks(tasks))
    if any(s.get("ppl") is not None for s in shards):
        if not all(s.get("ppl") is not None for s in shards):
            raise ValueError("incomplete perplexity shards")
        total_windows = first["ppl"]["total_windows"]
        for shard in shards:
            ppl = shard["ppl"]
            if (ppl["total_windows"] != total_windows or
                ppl["ids"] != partition(total_windows, shard["shard_index"], count) or
                ppl["tokens"] < 0 or not math.isfinite(ppl["nll"])):
                raise ValueError("invalid perplexity shard")
        nll = sum(s["ppl"]["nll"] for s in shards)
        tokens = sum(s["ppl"]["tokens"] for s in shards)
        if not tokens:
            raise ValueError("no perplexity tokens")
        result["ppl"] = {"nll": nll, "tokens": tokens, "value": math.exp(nll / tokens),
                         "total_windows": total_windows, "ids": list(range(total_windows))}
    return result
