"""Merge all shards, rejecting incomplete or incompatible runs."""
import argparse
import json
from pathlib import Path

try:
    from .accuracy_io import merge_shards, write_json
except ImportError:
    from accuracy_io import merge_shards, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = merge_shards([json.loads(p.read_text()) for p in args.inputs])
    write_json(args.output, result)
    print(json.dumps(result["scores"], indent=2))


if __name__ == "__main__":
    main()
