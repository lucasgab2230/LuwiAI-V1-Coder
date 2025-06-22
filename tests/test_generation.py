# tests/test_generation.py

import unittest
import torch
import json
import sys
import os

# Ensure the model directory is importable
sys.path.insert(0, os.path.abspath("model"))

from model.architecture import SmallCodeGenModel

class TestSmallCodeGenModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("model/config.json", "r") as f:
            cfg = json.load(f)
        cls.model = SmallCodeGenModel(
            vocab_size=cfg["vocab_size"],
            dim=cfg["dim"],
            depth=cfg["depth"],
            heads=cfg["heads"],
            ff_dim=cfg["ff_dim"],
            max_len=cfg["max_position_embeddings"]
        )
        cls.vocab_size = cfg["vocab_size"]
        cls.seq_len = 32
        cls.batch_size = 2

    def test_forward_shape(self):
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        output = self.model(input_ids)
        self.assertEqual(output.logits.shape, (self.batch_size, self.seq_len, self.vocab_size))

    def test_forward_with_labels_loss(self):
        input_ids = torch.randint(0, self.vocab_size, (1, self.seq_len))
        output = self.model(input_ids, labels=input_ids)
        self.assertTrue(hasattr(output, "loss"))
        self.assertGreaterEqual(output.loss.item(), 0.0)

if __name__ == "__main__":
    unittest.main()
