import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM

from evaluation.run_sparse_accuracy import checkpoint_files, main, parse_args


class ManifestTests(unittest.TestCase):
    def test_real_checkpoint_export_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source", root / "pruned"
            config = Qwen3MoeConfig(vocab_size=8, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                head_dim=8, num_experts=2, num_experts_per_tok=2, moe_intermediate_size=16)
            model = Qwen3MoeForCausalLM(config)
            model.save_pretrained(source)
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(
                WordLevel({"[UNK]": 0, "[PAD]": 1, "hello": 2}, unk_token="[UNK]")),
                unk_token="[UNK]", pad_token="[PAD]")
            tokenizer.save_pretrained(source)
            argv = ["runner", "--model", str(source), "--label", "nm28", "--dtype", "float32",
                    "--device-map", "cpu", "--sparsity-type", "2:8", "--prune-method", "magnitude",
                    "--prune-only", "--save-model", str(target), "--output", str(root / "prune.json")]
            with patch("sys.argv", argv):
                main()
            manifest = json.loads((target / "lattice_pruning.json").read_text())
            self.assertEqual(manifest["files"], checkpoint_files(target))
            self.assertEqual(manifest["recipe"]["sparsity"], .75)
            self.assertTrue(all(t["masked"] == t["parameters"] * .75 for t in manifest["projections"]))
            restored = Qwen3MoeForCausalLM.from_pretrained(target)
            self.assertTrue(torch.isfinite(restored(torch.tensor([[1, 2, 1]])).logits).all())
            (target / "config.json").write_text((target / "config.json").read_text() + "\n")
            with patch("sys.argv", ["runner", "--model", str(target), "--skip-pruning",
                                    "--label", "nm28", "--output", str(root / "eval.json")]):
                with self.assertRaisesRegex(ValueError, "differs from pruning manifest"):
                    main()

    def test_disallow_pruning_in_multiple_shards(self):
        with self.assertRaises(SystemExit):
            parse_args(["--model", "model", "--output", "out", "--label", "x",
                        "--sparsity-type", "2:4", "--num-shards", "2"])


if __name__ == "__main__":
    unittest.main()
