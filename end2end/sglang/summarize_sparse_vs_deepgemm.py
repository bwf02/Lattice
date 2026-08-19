#!/usr/bin/env python3
"""Summarize paired SGLang DeepGEMM and SparseGEMM serving runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


RESULT_RE = re.compile(
    r"^(?P<model>.+)-(?P<backend>deep_gemm|sparse_gemm)-"
    r"chunk(?P<chunk>\d+)-c(?P<concurrency>\d+)-r(?P<repeat>\d+)\.jsonl$"
)
METRICS = (
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_throughput",
    "median_ttft_ms",
    "p90_ttft_ms",
    "median_itl_ms",
    "p90_itl_ms",
    "median_e2e_latency_ms",
    "p90_e2e_latency_ms",
)


def load_results(result_dir: Path) -> dict:
    grouped = defaultdict(list)
    for path in sorted(result_dir.glob("*.jsonl")):
        match = RESULT_RE.match(path.name)
        if match is None:
            continue
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if not lines:
            continue
        result = json.loads(lines[-1])
        key = (
            match["model"],
            int(match["chunk"]),
            int(match["concurrency"]),
            match["backend"],
        )
        result["repeat"] = int(match["repeat"])
        grouped[key].append(result)
    return grouped


def median_metrics(runs: list[dict]) -> dict:
    summary = {metric: statistics.median(run[metric] for run in runs) for metric in METRICS}
    summary["completed"] = min(run["completed"] for run in runs)
    summary["repeats"] = len(runs)
    values = [run["total_throughput"] for run in runs]
    summary["throughput_spread_pct"] = (
        100 * (max(values) - min(values)) / statistics.median(values)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    grouped = load_results(args.result_root / "results")
    rows = []
    configs = sorted({key[:3] for key in grouped})
    for model, chunk, concurrency in configs:
        deep_runs = grouped.get((model, chunk, concurrency, "deep_gemm"), [])
        sparse_runs = grouped.get((model, chunk, concurrency, "sparse_gemm"), [])
        if not deep_runs or not sparse_runs:
            continue
        deep = median_metrics(deep_runs)
        sparse = median_metrics(sparse_runs)
        row = {
            "model": model,
            "chunk_size": chunk,
            "concurrency": concurrency,
            "deep_repeats": deep["repeats"],
            "sparse_repeats": sparse["repeats"],
            "deep_completed": deep["completed"],
            "sparse_completed": sparse["completed"],
            "deep_total_tok_s": deep["total_throughput"],
            "sparse_total_tok_s": sparse["total_throughput"],
            "throughput_speedup": sparse["total_throughput"] / deep["total_throughput"],
            "deep_median_ttft_ms": deep["median_ttft_ms"],
            "sparse_median_ttft_ms": sparse["median_ttft_ms"],
            "ttft_speedup": deep["median_ttft_ms"] / sparse["median_ttft_ms"],
            "deep_median_itl_ms": deep["median_itl_ms"],
            "sparse_median_itl_ms": sparse["median_itl_ms"],
            "itl_speedup": deep["median_itl_ms"] / sparse["median_itl_ms"],
            "deep_median_e2e_ms": deep["median_e2e_latency_ms"],
            "sparse_median_e2e_ms": sparse["median_e2e_latency_ms"],
            "e2e_speedup": deep["median_e2e_latency_ms"] / sparse["median_e2e_latency_ms"],
            "deep_spread_pct": deep["throughput_spread_pct"],
            "sparse_spread_pct": sparse["throughput_spread_pct"],
        }
        rows.append(row)

    output = args.result_root / "summary.csv"
    if rows:
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"paired_configs={len(rows)} summary={output}")


if __name__ == "__main__":
    main()
