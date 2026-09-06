#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


FILE_RE = re.compile(
    r"(?P<model>.+)-(?P<backend>deep_gemm|sparse_gemm|slidesparse)-"
    r"(?P<profile>prefill|decode|mixed)-(?P<case>m\d+|c\d+)-r(?P<repeat>\d+)\.jsonl"
)
METRICS = (
    "duration",
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "mean_e2e_latency_ms",
    "median_e2e_latency_ms",
    "p95_e2e_latency_ms",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p95_ttft_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p95_itl_ms",
)


def read_result(path: Path) -> dict:
    with path.open() as f:
        return json.loads(f.readline())


def normalize_result(result: dict) -> dict:
    if "random_input_len" in result:
        return {
            "input_len": result["random_input_len"],
            "output_len": result["random_output_len"],
            "concurrency": result["max_concurrency"],
            "completed": result["completed"],
            **{metric: result.get(metric) for metric in METRICS},
        }

    latency_ms = float(result["latency"]) * 1000
    ttft_ms = float(result["last_ttft"]) * 1000
    batch_size = int(result["batch_size"])
    normalized = {
        "input_len": result["input_len"],
        "output_len": result["output_len"],
        "concurrency": batch_size,
        "completed": batch_size,
        "duration": result["latency"],
        "request_throughput": batch_size / float(result["latency"]),
        "input_throughput": result["input_throughput"],
        "output_throughput": result["output_throughput"],
        "mean_e2e_latency_ms": latency_ms,
        "median_e2e_latency_ms": latency_ms,
        "p95_e2e_latency_ms": latency_ms,
        "mean_ttft_ms": ttft_ms,
        "median_ttft_ms": ttft_ms,
        "p95_ttft_ms": ttft_ms,
        "mean_itl_ms": math.nan,
        "median_itl_ms": math.nan,
        "p95_itl_ms": math.nan,
    }
    return normalized


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def speedup_metric(profile: str) -> str:
    if profile == "prefill":
        return "input_throughput"
    if profile == "decode":
        return "output_throughput"
    return "mean_e2e_latency_ms"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--reference-csv", type=Path,
                        help="Archived paper figure data; compare without rerunning baselines")
    args = parser.parse_args()
    raw_dir = args.result_root / "raw"
    rows = []

    for path in sorted(raw_dir.glob("*.jsonl")):
        match = FILE_RE.fullmatch(path.name)
        if not match:
            continue
        result = normalize_result(read_result(path))
        row = match.groupdict()
        row["repeat"] = int(row["repeat"])
        row.update(
            input_len=result["input_len"],
            output_len=result["output_len"],
            concurrency=result["concurrency"],
            completed=result["completed"],
            source_file=str(path),
        )
        row.update({metric: result.get(metric) for metric in METRICS})
        rows.append(row)

    if not rows:
        raise SystemExit(f"No benchmark JSONL files found under {raw_dir}")

    per_run_path = args.result_root / "per_run.csv"
    with per_run_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        key = (row["model"], row["backend"], row["profile"], row["case"])
        groups[key].append(row)

    aggregate = []
    for key, group in sorted(groups.items()):
        model, backend, profile, case = key
        out = {
            "model": model,
            "backend": backend,
            "profile": profile,
            "case": case,
            "input_len": group[0]["input_len"],
            "output_len": group[0]["output_len"],
            "concurrency": group[0]["concurrency"],
            "repeats": len(group),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group if row[metric] is not None]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_stdev"] = stdev(values)
        aggregate.append(out)

    aggregate_path = args.result_root / "aggregate.csv"
    with aggregate_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    by_case = defaultdict(dict)
    for row in aggregate:
        by_case[(row["model"], row["profile"], row["case"])][row["backend"]] = row

    comparisons = []
    for (model, profile, case), backends in sorted(by_case.items()):
        if "deep_gemm" not in backends or "sparse_gemm" not in backends:
            continue
        deep = backends["deep_gemm"]
        sparse = backends["sparse_gemm"]
        metric = speedup_metric(profile)
        deep_value = float(deep[f"{metric}_mean"])
        sparse_value = float(sparse[f"{metric}_mean"])
        speedup = deep_value / sparse_value if metric.endswith("latency_ms") else sparse_value / deep_value
        deep_itl = float(deep["mean_itl_ms_mean"])
        sparse_itl = float(sparse["mean_itl_ms_mean"])
        comparisons.append(
            {
                "model": model,
                "profile": profile,
                "case": case,
                "input_len": deep["input_len"],
                "output_len": deep["output_len"],
                "concurrency": deep["concurrency"],
                "primary_metric": metric,
                "deep_gemm": deep_value,
                "sparse_gemm": sparse_value,
                "speedup": speedup,
                "deep_e2e_ms": deep["mean_e2e_latency_ms_mean"],
                "sparse_e2e_ms": sparse["mean_e2e_latency_ms_mean"],
                "e2e_speedup": float(deep["mean_e2e_latency_ms_mean"])
                / float(sparse["mean_e2e_latency_ms_mean"]),
                "deep_ttft_ms": deep["mean_ttft_ms_mean"],
                "sparse_ttft_ms": sparse["mean_ttft_ms_mean"],
                "ttft_speedup": float(deep["mean_ttft_ms_mean"])
                / float(sparse["mean_ttft_ms_mean"]),
                "deep_itl_ms": deep_itl,
                "sparse_itl_ms": sparse_itl,
                "itl_speedup": deep_itl / sparse_itl
                if (
                    not math.isnan(deep_itl)
                    and not math.isnan(sparse_itl)
                    and sparse_itl != 0
                )
                else math.nan,
            }
        )

    if comparisons:
        comparison_path = args.result_root / "comparisons.csv"
        with comparison_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)

    if args.reference_csv:
        with args.reference_csv.open() as f:
            reference_rows = list(csv.DictReader(f))
        references = {}
        for row in reference_rows:
            key = (row["model"], row["profile"], row["case"])
            if key in references:
                raise ValueError(f"Duplicate reference case: {key}")
            references[key] = row
        slidesparse_comparisons = []
        for row in aggregate:
            if row["backend"] != "slidesparse":
                continue
            key = (row["model"], row["profile"], row["case"])
            reference = references[key]
            for field in ("input_len", "output_len", "concurrency"):
                if int(row[field]) != int(reference[field]):
                    raise ValueError(f"Workload mismatch for {key}: {field}")
            metric = speedup_metric(row["profile"])
            if reference["primary_metric"] != metric:
                raise ValueError(f"Metric mismatch for {key}")
            value = float(row[f"{metric}_mean"])
            deep = float(reference["deep_gemm"])
            sparse = float(reference["sparse_gemm"])
            if not all(math.isfinite(v) and v > 0 for v in (value, deep, sparse)):
                raise ValueError(f"Invalid primary measurement for {key}")
            latency = metric.endswith("latency_ms")
            slidesparse_comparisons.append({
                "model": row["model"], "profile": row["profile"], "case": row["case"],
                "input_len": row["input_len"], "output_len": row["output_len"],
                "concurrency": row["concurrency"], "primary_metric": metric,
                "deep_gemm": deep, "sparse_gemm": sparse, "slidesparse": value,
                "slidesparse_over_deepgemm": deep / value if latency else value / deep,
                "losparse_over_slidesparse": value / sparse if latency else sparse / value,
                "slidesparse_e2e_ms": row["mean_e2e_latency_ms_mean"],
                "slidesparse_ttft_ms": row["mean_ttft_ms_mean"],
                "slidesparse_itl_ms": row["mean_itl_ms_mean"],
                "repeats": row["repeats"],
                "reference_csv": str(args.reference_csv.resolve()),
                "reference_run": reference.get("source_run", ""),
            })
        if slidesparse_comparisons:
            with (args.result_root / "slidesparse_comparisons.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(slidesparse_comparisons[0]))
                writer.writeheader()
                writer.writerows(slidesparse_comparisons)


if __name__ == "__main__":
    main()
