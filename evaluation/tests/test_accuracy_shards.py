from copy import deepcopy
import math
import unittest

from evaluation.accuracy_io import merge_shards, partition
from evaluation.summarize_accuracy import summarize


def fixtures(count=3):
    shards = []
    for index in range(count):
        tasks = {}
        for name, total, correct in [("mmlu_a", 7, {0, 2, 3, 5}), ("mmlu_b", 2, {0})]:
            ids = partition(total, index, count)
            tasks[name] = {"group": "mmlu", "metric": "acc,none", "total": total,
                "count": len(ids), "ids": ids, "sum": sum(i in correct for i in ids),
                "dataset_digest": name, "config_digest": "config"}
        windows = partition(2, index, count)
        shards.append({"schema_version": 2, "label": "dense", "protocol": {"checkpoint_digest": "same"},
            "num_shards": count, "shard_index": index, "tasks": tasks,
            "ppl": {"total_windows": 2, "ids": windows, "tokens": len(windows)*3,
                    "nll": len(windows)*6}})
    return shards


class ShardTests(unittest.TestCase):
    def test_weighted_mmlu_and_ppl(self):
        result = merge_shards(fixtures())
        self.assertAlmostEqual(result["scores"]["mmlu"], 5/9)
        self.assertAlmostEqual(result["ppl"]["value"], math.exp(2))
        self.assertEqual(result["ppl"]["tokens"], 6)

    def test_partition_is_disjoint_complete(self):
        for total in range(12):
            for count in range(1, 8):
                ids = sum([partition(total, i, count) for i in range(count)], [])
                self.assertEqual(sorted(ids), list(range(total)))

    def test_missing_duplicate_and_changed_config(self):
        for mutate in (lambda s: s.pop(), lambda s: s.append(s[0]),
                       lambda s: s[1]["protocol"].update(checkpoint_digest="other"),
                       lambda s: s[1]["tasks"]["mmlu_a"].update(dataset_digest="other"),
                       lambda s: s[1]["tasks"]["mmlu_a"].update(ids=[0]),
                       lambda s: s[1]["tasks"]["mmlu_a"].update(sum=float("nan"))):
            shards = fixtures()
            mutate(shards)
            with self.assertRaises(ValueError):
                merge_shards(shards)

    def test_summary_percent_pp_and_blanks(self):
        dense = merge_shards(fixtures())
        sparse = deepcopy(dense)
        sparse["label"] = "hibnm"
        sparse["scores"]["mmlu"] += .01
        rows = summarize([dense, sparse], ["mmlu", "asdiv"], "dense")
        self.assertAlmostEqual(rows[1]["mmlu_delta_pp"], 1.)
        self.assertEqual(rows[1]["asdiv_percent"], "")

    def test_summary_rejects_shards_and_incomparable_runs(self):
        with self.assertRaises(ValueError):
            summarize(fixtures(), ["mmlu"])
        dense = merge_shards(fixtures())
        other = deepcopy(dense)
        other["label"] = "other"
        other["protocol"]["limit"] = 8
        with self.assertRaises(ValueError):
            summarize([dense, other], ["mmlu"], "dense")


if __name__ == "__main__":
    unittest.main()
