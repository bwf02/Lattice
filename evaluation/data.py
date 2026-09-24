"""Explicit train-only calibration; evaluation tasks come from lm-eval."""
import random


def calibration_batches(tokenizer, *, samples, length, seed, device,
                        path=None, dataset="wikitext", config="wikitext-2-raw-v1"):
    from datasets import load_dataset
    if samples < 1 or length < 2:
        raise ValueError("calibration requires positive samples and length >= 2")
    if path:
        kind = "parquet" if str(path).endswith(".parquet") else "json"
        data = load_dataset(kind, data_files={"train": str(path)}, split="train")
    else:
        data = load_dataset(dataset, config, split="train")
    ids = tokenizer(" ".join(data["text"]), return_tensors="pt").input_ids
    if ids.shape[1] < length:
        raise ValueError("calibration corpus shorter than requested window")
    rng = random.Random(seed)
    for _ in range(samples):
        start = rng.randrange(ids.shape[1] - length + 1)
        yield {"input_ids": ids[:, start:start + length].to(device)}
