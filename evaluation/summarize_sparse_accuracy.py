#!/usr/bin/env python3
"""Summarize sparse-accuracy JSON outputs as Markdown and CSV tables."""

import argparse
import csv
import json
from pathlib import Path


ORDER = [
    "dense",
    "unstructured_25",
    "unstructured_50",
    "unstructured_75",
    "unstructured_90",
    "nm_2_4",
    "nm_1_4",
    "nm_2_16",
]
FIELDS = [
    "label",
    "pattern",
    "target_sparsity",
    "actual_sparsity",
    "gsm8k",
    "math500",
    "average",
    "ppl",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--ppl-dir")
    parser.add_argument("--labels", default=",".join(ORDER))
    parser.add_argument("--include-humaneval", action="store_true")
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--csv", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    ppl_dir = Path(args.ppl_dir) if args.ppl_dir else None
    rows = []
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    fields = FIELDS.copy()
    if args.include_humaneval:
        fields.insert(-2, "humaneval")
    for label in labels:
        path = input_dir / f"{label}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        result = json.loads(path.read_text())
        ppl = result.get("ppl")
        if ppl is None and ppl_dir is not None:
            ppl_path = ppl_dir / f"{label}.json"
            if ppl_path.is_file():
                ppl = json.loads(ppl_path.read_text()).get("ppl")
        scores = result["scores"]
        rows.append(
            {
                "label": label,
                "pattern": result.get("kept_pattern") or result["sparsity_type"],
                "target_sparsity": result["target_sparsity"],
                "actual_sparsity": result["actual_sparsity"],
                "gsm8k": scores.get("gsm8k"),
                "math500": scores.get("math500"),
                "average": scores.get("average"),
                "ppl": ppl,
            }
        )
        if args.include_humaneval:
            rows[-1]["humaneval"] = scores.get("humaneval")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Configuration | Pattern | Target sparsity | Actual sparsity | GSM8K | MATH-500 | Average | PPL |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    if args.include_humaneval:
        lines = [
            "| Configuration | Pattern | Target sparsity | Actual sparsity | GSM8K | MATH-500 | HumanEval | Average | PPL |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    for row in rows:
        values = [
            row["label"],
            row["pattern"],
            f"{100 * row['target_sparsity']:.2f}%",
            f"{100 * row['actual_sparsity']:.2f}%",
            *(
                "-" if row[field] is None else f"{100 * row[field]:.2f}%"
                for field in fields[4:-1]
            ),
            "-" if row["ppl"] is None else f"{row['ppl']:.2f}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
