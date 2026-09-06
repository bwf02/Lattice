#!/usr/bin/env python3
"""Join complete SlideSparse aggregates to the unchanged archived paper CSV."""

import argparse
import csv
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("result_roots", nargs="+", type=Path)
    args = parser.parse_args()
    with args.reference.open() as f:
        references = list(csv.DictReader(f))
    expected = {(r["model"], r["profile"], r["case"]) for r in references}
    if len(expected) != len(references):
        raise ValueError("Duplicate reference cases")
    collected = {}
    for root in args.result_roots:
        with (root / "aggregate.csv").open() as f:
            for row in csv.DictReader(f):
                if row["backend"] != "slidesparse":
                    continue
                key = (row["model"], row["profile"], row["case"])
                if key in collected:
                    raise ValueError(f"Duplicate collected case: {key}")
                collected[key] = (row, root.resolve())
    if set(collected) != expected:
        raise ValueError(f"Missing: {expected - set(collected)}; unexpected: {set(collected) - expected}")
    rows = []
    for reference in references:
        key = (reference["model"], reference["profile"], reference["case"])
        measured, root = collected[key]
        for field in ("input_len", "output_len", "concurrency"):
            if int(reference[field]) != int(measured[field]):
                raise ValueError(f"Workload mismatch: {key}, {field}")
        metric = reference["primary_metric"]
        value = float(measured[f"{metric}_mean"])
        deep = float(reference["deep_gemm"])
        sparse = float(reference["sparse_gemm"])
        if not all(math.isfinite(v) and v > 0 for v in (value, deep, sparse)):
            raise ValueError(f"Invalid measurement: {key}")
        latency = metric.endswith("latency_ms")
        rows.append({
            **reference,
            "slidesparse": value,
            "slidesparse_speedup": deep / value if latency else value / deep,
            "losparse_over_slidesparse": value / sparse if latency else sparse / value,
            "slidesparse_e2e_ms": measured["mean_e2e_latency_ms_mean"],
            "slidesparse_ttft_ms": measured["mean_ttft_ms_mean"],
            "slidesparse_itl_ms": measured["mean_itl_ms_mean"],
            "slidesparse_repeats": measured["repeats"],
            "slidesparse_source": str(root),
        })
    if args.output.resolve() == args.reference.resolve():
        raise ValueError("Refusing to overwrite the archived reference")
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Merged {len(rows)} cases into {args.output}")


if __name__ == "__main__":
    main()
