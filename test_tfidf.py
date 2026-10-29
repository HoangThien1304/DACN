"""Unit tests for train-only IDF and report-position TF-IDF weights."""
import math
import unittest

import torch

from data.tfidf import compute_idf_weights, tfidf_weights
from data.tokenizer import SimpleTokenizer
from model.losses import term_weighted_cross_entropy


class TfidfTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SimpleTokenizer({
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
            "common": 4,
            "repeat": 5,
            "rare": 6,
        })

    def test_idf_uses_document_frequency_over_training_reports(self):
        idf = compute_idf_weights(
            self.tokenizer,
            ["common repeat repeat", "common rare", "common repeat"],
        )

        self.assertAlmostEqual(idf[4].item(), 0.0, places=7)
        self.assertAlmostEqual(idf[5].item(), math.log(3 / 2), places=7)
        self.assertAlmostEqual(idf[6].item(), math.log(3), places=7)
        self.assertEqual(idf[3].item(), 0.0)

    def test_tf_is_report_specific_and_special_token_weights_are_explicit(self):
        idf = compute_idf_weights(
            self.tokenizer,
            ["common repeat repeat", "common rare", "common repeat"],
        )
        weights = tfidf_weights(
            [1, 4, 5, 5, 3, 2, 0], idf,
            pad_id=0, bos_id=1, eos_id=2, unk_id=3,
        )

        self.assertAlmostEqual(weights[2], 0.5 * math.log(3 / 2), places=7)
        self.assertAlmostEqual(weights[3], 0.5 * math.log(3 / 2), places=7)
        self.assertEqual(weights[0], 1.0)  # BOS
        self.assertEqual(weights[4], 0.0)  # UNK
        self.assertEqual(weights[5], 1.0)  # EOS
        self.assertEqual(weights[6], 0.0)  # PAD

    def test_position_weighted_rg_matches_manual_weighted_cross_entropy(self):
        logits = torch.tensor([[[0.2, 0.8, -0.1], [0.5, -0.2, 0.3]]])
        targets = torch.tensor([[1, 2]])
        weights = torch.tensor([[0.25, 1.0]])
        token_losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 3), targets.reshape(-1), reduction="none"
        ).reshape_as(targets)
        expected = (token_losses * weights).sum() / weights.sum()

        actual = term_weighted_cross_entropy(logits, targets, weights, pad_idx=0)

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
