import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ArchivedComparisonTest(unittest.TestCase):
    def test_prefill_reference_and_workload_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            raw = root / "raw/qwen15-slidesparse-prefill-m4096-r1.jsonl"
            raw.write_text(json.dumps({
                "batch_size": 4, "input_len": 1024, "output_len": 1,
                "latency": 2, "last_ttft": 1.9,
                "input_throughput": 2048, "output_throughput": 2,
            }) + "\n")
            reference = root / "reference.csv"
            reference.write_text(
                "model,profile,case,input_len,output_len,concurrency,primary_metric,deep_gemm,sparse_gemm\n"
                "qwen15,prefill,m4096,1024,1,4,input_throughput,4096,8192\n"
            )
            command = [sys.executable, str(Path(__file__).with_name("summarize_paper_e2e.py")),
                       str(root), "--reference-csv", str(reference)]
            subprocess.run(command, check=True, capture_output=True)
            with (root / "slidesparse_comparisons.csv").open() as f:
                row = next(csv.DictReader(f))
            self.assertEqual(float(row["slidesparse_over_deepgemm"]), 0.5)
            self.assertEqual(float(row["lattice_over_slidesparse"]), 4.0)
            merge = [sys.executable, str(Path(__file__).with_name("merge_slidesparse_reference.py")),
                     str(reference), str(root / "merged.csv"), str(root)]
            subprocess.run(merge, check=True, capture_output=True)
            with (root / "merged.csv").open() as f:
                merged = next(csv.DictReader(f))
            self.assertEqual(merged["deep_gemm"], "4096")
            self.assertEqual(float(merged["slidesparse_speedup"]), 0.5)
            duplicate = subprocess.run(merge + [str(root)], capture_output=True, text=True)
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertIn("Duplicate collected case", duplicate.stderr)
            reference.write_text(reference.read_text().replace("1024,1,4", "1024,1,8"))
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Workload mismatch", failed.stderr)


if __name__ == "__main__":
    unittest.main()
