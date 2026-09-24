"""Launch independent evaluation shards, then merge only after all succeed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

try:
    from .accuracy_io import merge_shards, write_json
except ImportError:
    from accuracy_io import merge_shards, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", required=True, help="Comma-separated GPU IDs, one model replica per GPU")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("args", nargs=argparse.REMAINDER, help="Runner options after --")
    args = p.parse_args()
    gpu_ids = args.gpus.split(",")
    if any(not s.strip() for s in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        p.error("GPU IDs must be nonempty and distinct")
    forwarded = args.args[1:] if args.args[:1] == ["--"] else args.args
    reserved = {"--output", "--num-shards", "--shard-index", "--save-model", "--prune-only"}
    if any(s.split("=", 1)[0] in reserved for s in forwarded):
        p.error("output/shard/export options are controlled by this launcher")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error("output directory must be empty; stale shards cannot be reused")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_sparse_accuracy.py")
    processes, logs = [], []
    try:
        for index, gpu in enumerate(gpu_ids):
            log = (args.output_dir / f"shard-{index}.log").open("w")
            logs.append(log)
            command = [sys.executable, str(runner), *forwarded,
                       "--num-shards", str(len(gpu_ids)), "--shard-index", str(index),
                       "--output", str(args.output_dir / f"shard-{index}.json")]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu.strip())
            processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
        failures = [index for index, process in enumerate(processes) if process.wait()]
        if failures:
            raise RuntimeError(f"shards {failures} failed; inspect logs; no merged result written")
        shards = [json.loads((args.output_dir / f"shard-{i}.json").read_text())
                  for i in range(len(gpu_ids))]
        write_json(args.output_dir / "merged.json", merge_shards(shards))
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
