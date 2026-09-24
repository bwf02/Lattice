"""Offline integration with the real harness; no model or dataset downloads."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset, DatasetDict
from lm_eval.api.task import ConfigurableTask
from lm_eval.models.dummy import DummyLM

from evaluation.accuracy_io import merge_shards
from evaluation.run_sparse_accuracy import evaluate_tasks


class LocalTask(ConfigurableTask):
    def download(self, *args, **kwargs):
        self.dataset = DatasetDict({
            "train": Dataset.from_dict({"question": [f"train {i}" for i in range(6)], "answer": ["yes"]*6}),
            "test": Dataset.from_dict({"question": [f"test {i}" for i in range(7)], "answer": ["yes"]*7})})


def task():
    return LocalTask(config={"task": "mmlu_local", "dataset_path": "local", "training_split": "train",
        "test_split": "test", "output_type": "loglikelihood", "doc_to_text": "{{question}}",
        "doc_to_target": "{{answer}}", "num_fewshot": 5,
        "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}]})


class AlwaysCorrect(DummyLM):
    def loglikelihood(self, requests, **kwargs):
        return [(-1., True) for _ in requests]


class HarnessTests(unittest.TestCase):
    def test_real_harness_partition_and_merge(self):
        shards = []
        for index in range(3):
            args = SimpleNamespace(tasks=["mmlu"], task_config_dir=None, batch_size="1", limit=None,
                                   shard_index=index, num_shards=3, seed=0, save_samples=True)
            with patch("lm_eval.tasks.get_task_dict", return_value={"mmlu": {"mmlu_local": task()}}), \
                 patch("lm_eval.tasks.TaskManager"), \
                 patch("lm_eval.models.huggingface.HFLM", return_value=AlwaysCorrect()):
                records, samples = evaluate_tasks(None, None, args)
            self.assertEqual(records["mmlu_local"]["ids"], list(range(index, 7, 3)))
            self.assertTrue(samples)
            shards.append({"schema_version": 2, "label": "test", "protocol": {},
                           "num_shards": 3, "shard_index": index, "tasks": records, "ppl": None})
        result = merge_shards(shards)
        self.assertEqual(result["scores"]["mmlu"], 1.)


if __name__ == "__main__":
    unittest.main()
