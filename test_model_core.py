"""Unit tests for the paper-critical loss and model behavior."""
import math
import unittest

import torch
from collections import Counter

from config import ModelConfig, TrainConfig
from model.decoder import MemoryAugmentedDecoder, MemoryAugmentedDecoderLayer
from model.full_model import MulticriteriaTransformer
from model.heads import ITMScorer, build_same_matrix, sample_negatives
from model.losses import itm_triplet_loss, term_weighted_cross_entropy
from model.vit_encoder import TransformerEncoderLayer, ViTEncoder


class ITMTripletLossTests(unittest.TestCase):
    def test_optimization_pushes_positive_score_above_negative(self):
        score_pos = torch.nn.Parameter(torch.zeros(1))
        score_neg = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([score_pos, score_neg], lr=0.1)

        for _ in range(8):
            optimizer.zero_grad()
            loss = itm_triplet_loss(score_pos, score_neg, epsilon=0.05, mu_itm=0.025)
            loss.backward()
            optimizer.step()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(score_pos.item(), score_neg.item())

    def test_satisfied_margin_leaves_only_scaled_l2_term(self):
        score_pos = torch.tensor([0.8, 0.6])
        score_neg = torch.tensor([0.2, 0.1])
        mu_itm = 0.025
        lambda_itm = 5.0

        weighted_loss = lambda_itm * itm_triplet_loss(
            score_pos, score_neg, epsilon=0.05, mu_itm=mu_itm
        )
        expected = lambda_itm * mu_itm * (
            score_pos.square() + score_neg.square()
        ).mean()

        self.assertGreaterEqual(weighted_loss.item(), 0.0)
        self.assertTrue(torch.isfinite(weighted_loss))
        torch.testing.assert_close(weighted_loss, expected)


class GenerationLossTests(unittest.TestCase):
    def test_pad_positions_do_not_contribute_to_rg(self):
        targets = torch.tensor([[1, 2, 0]])
        weights = torch.ones_like(targets, dtype=torch.float)
        logits = torch.zeros(1, 3, 4, requires_grad=True)
        changed_pad_logits = logits.detach().clone()
        changed_pad_logits[:, 2, :] = torch.tensor([100.0, -100.0, 50.0, -50.0])
        changed_pad_logits.requires_grad_()

        loss = term_weighted_cross_entropy(logits, targets, weights, pad_idx=0)
        changed_loss = term_weighted_cross_entropy(
            changed_pad_logits, targets, weights, pad_idx=0
        )
        torch.testing.assert_close(loss, changed_loss)

        changed_loss.backward()
        torch.testing.assert_close(
            changed_pad_logits.grad[:, 2, :], torch.zeros_like(changed_pad_logits.grad[:, 2, :])
        )


class ITMScorerTests(unittest.TestCase):
    def test_score_is_cosine_similarity_to_pooled_image(self):
        scorer = ITMScorer(d_model=3, alpha=0.5)
        image = torch.tensor([[1.0, 0.0, 0.0]])
        hidden = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])

        aligned_score = scorer(image, hidden)
        opposite_score = scorer(-image, hidden)

        self.assertEqual(aligned_score.shape, (1,))
        torch.testing.assert_close(aligned_score, torch.ones(1))
        torch.testing.assert_close(opposite_score, -torch.ones(1))

    def test_pad_steps_do_not_change_itm_score(self):
        scorer = ITMScorer(d_model=3, alpha=0.5)
        image = torch.tensor([[1.0, 0.0, 0.0]])
        hidden = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])
        valid_mask = torch.tensor([[True, True, False]])

        masked_score = scorer(image, hidden, valid_mask)
        shorter_score = scorer(image, hidden[:, :2])
        unmasked_score = scorer(image, hidden)

        torch.testing.assert_close(masked_score, shorter_score)
        self.assertNotAlmostEqual(masked_score.item(), unmasked_score.item())

    def test_step_scores_and_temporal_formula_shapes_and_values(self):
        scorer = ITMScorer(d_model=3, alpha=0.5)
        image = torch.tensor([[1.0, 0.0, 0.0]])
        hidden = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]])

        step_scores = scorer.step_scores(image, hidden)
        total_score = scorer(image, hidden)
        weights = torch.tensor([3.0 ** -0.5, 2.0 ** -0.5, 1.0])
        expected = (weights * torch.tensor([1.0, 0.0, -1.0])).sum() / weights.sum()

        self.assertEqual(step_scores.shape, (1, 3))
        self.assertEqual(total_score.shape, (1,))
        torch.testing.assert_close(total_score[0], expected)


class NegativeSamplingTests(unittest.TestCase):
    def test_same_matrix_ignores_trailing_pad_and_negative_is_different(self):
        reports = torch.tensor([
            [1, 4, 2, 0, 0],
            [1, 4, 2, 0, 0],
            [1, 5, 2, 0, 0],
        ])
        same = build_same_matrix(reports, pad_idx=0)
        expected = torch.tensor([
            [True, True, False],
            [True, True, False],
            [False, False, True],
        ])
        torch.testing.assert_close(same, expected)

        generator = torch.Generator().manual_seed(7)
        negative_indices, valid = sample_negatives(same, generator=generator)
        self.assertTrue(valid.all())
        self.assertTrue((~same[torch.arange(3), negative_indices]).all())

    def test_all_duplicate_reports_mark_every_anchor_invalid(self):
        reports = torch.tensor([[1, 4, 2, 0], [1, 4, 2, 0]])
        same = build_same_matrix(reports, pad_idx=0)
        negative_indices, valid = sample_negatives(
            same, generator=torch.Generator().manual_seed(11)
        )

        self.assertEqual(int((~valid).sum()), 2)
        self.assertEqual(negative_indices.shape, (2,))

    def test_model_returns_zero_itm_and_counts_when_batch_has_only_duplicates(self):
        cfg = ModelConfig()
        cfg.img_size = 32
        cfg.patch_size = 32
        cfg.d_model = 32
        cfg.nhead = 4
        cfg.dim_feedforward = 64
        cfg.num_encoder_layers = 1
        cfg.num_decoder_layers = 1
        cfg.memory_slots = 2
        cfg.memory_head = 4
        cfg.max_seq_len = 6
        cfg.vocab_size = 8
        cfg.dropout = 0.0
        model = MulticriteriaTransformer(cfg, TrainConfig()).eval()
        images = torch.randn(2, 2, cfg.in_channels, cfg.img_size, cfg.img_size)
        report_ids = torch.tensor([[1, 4, 5, 2, 0, 0]]).repeat(2, 1)
        term_weights = torch.ones(2, cfg.max_seq_len - 1)

        loss, loss_dict = model(images, report_ids, term_weights=term_weights)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss_dict["itm"], 0.0)
        self.assertEqual(loss_dict["itm_skipped"], 2)


class PreNormTests(unittest.TestCase):
    def test_encoder_normalizes_before_attention_and_ffn(self):
        layer = TransformerEncoderLayer(d_model=8, nhead=2, dim_ff=16, dropout=0.0)
        events = []
        layer.norm1.register_forward_pre_hook(lambda *_: events.append("norm1"))
        layer.attn.register_forward_pre_hook(lambda *_: events.append("attn"))
        layer.norm2.register_forward_pre_hook(lambda *_: events.append("norm2"))
        layer.ffn.register_forward_pre_hook(lambda *_: events.append("ffn"))

        layer(torch.randn(2, 3, 8))

        self.assertEqual(events, ["norm1", "attn", "norm2", "ffn"])

    def test_decoder_keeps_memory_layernorm_before_each_sublayer(self):
        layer = MemoryAugmentedDecoderLayer(d_model=8, nhead=2, dim_ff=16, dropout=0.0)
        events = []
        for name in ("mln1", "self_attn", "mln2", "cross_attn", "mln3", "ffn"):
            getattr(layer, name).register_forward_pre_hook(
                lambda _module, _inputs, name=name: events.append(name)
            )

        layer(
            torch.randn(2, 3, 8),
            torch.randn(2, 4, 8),
            torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1),
            torch.randn(2, 8),
        )

        self.assertEqual(
            events, ["mln1", "self_attn", "mln2", "cross_attn", "mln3", "ffn"]
        )


class PaperShapeTests(unittest.TestCase):
    def test_paper_encoder_decoder_and_itm_shapes(self):
        cfg = ModelConfig()
        cfg.d_model = 768
        cfg.vocab_size = 23
        cfg.dropout = 0.0
        encoder = ViTEncoder(cfg).eval()
        decoder = MemoryAugmentedDecoder(cfg).eval()
        scorer = ITMScorer(cfg.d_model, cfg.itm_alpha)
        images = torch.randn(1, 2, cfg.in_channels, cfg.img_size, cfg.img_size)

        with torch.no_grad():
            encoder_features = encoder(images)
            visual = encoder_features[:, 1:]
            pooled_image = encoder_features[:, 0]
            logits, hidden = decoder(torch.tensor([[1, 3, 4, 5]]), visual)
            scores = scorer(pooled_image, hidden)

        self.assertEqual(encoder_features.shape, (1, 2 * 196 + 1, 768))
        self.assertEqual(visual.shape, (1, 2 * 196, 768))
        self.assertEqual(logits.shape, (1, 4, cfg.vocab_size))
        self.assertEqual(scores.shape, (1,))


def corpus_bleu4(hypotheses, references):
    clipped = [0] * 4
    totals = [0] * 4
    hypothesis_length = 0
    reference_length = 0
    for hypothesis, reference in zip(hypotheses, references):
        hypothesis_length += len(hypothesis)
        reference_length += len(reference)
        for n in range(1, 5):
            hyp_counts = Counter(
                tuple(hypothesis[i:i + n])
                for i in range(max(0, len(hypothesis) - n + 1))
            )
            ref_counts = Counter(
                tuple(reference[i:i + n])
                for i in range(max(0, len(reference) - n + 1))
            )
            clipped[n - 1] += sum(
                min(count, ref_counts[gram]) for gram, count in hyp_counts.items()
            )
            totals[n - 1] += sum(hyp_counts.values())
    if hypothesis_length == 0 or any(total == 0 or match == 0 for match, total in zip(clipped, totals)):
        return 0.0
    brevity_penalty = min(1.0, math.exp(1 - reference_length / hypothesis_length))
    geometric_mean = math.exp(
        sum(math.log(match / total) for match, total in zip(clipped, totals)) / 4
    )
    return brevity_penalty * geometric_mean


class OverfitTests(unittest.TestCase):
    def test_model_overfits_sixteen_fake_reports_with_high_bleu4(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(17)
            cfg = ModelConfig()
            cfg.img_size = 32
            cfg.patch_size = 32
            cfg.d_model = 32
            cfg.nhead = 4
            cfg.dim_feedforward = 64
            cfg.num_encoder_layers = 1
            cfg.num_decoder_layers = 1
            cfg.memory_slots = 2
            cfg.memory_head = 4
            cfg.max_seq_len = 7
            cfg.vocab_size = 9
            cfg.dropout = 0.0
            cfg.num_mlc_labels = 14
            model = MulticriteriaTransformer(cfg, TrainConfig()).eval()
            images = torch.randn(16, 2, cfg.in_channels, cfg.img_size, cfg.img_size)
            reports = torch.tensor([[1, 4, 5, 6, 7, 2, 0]]).repeat(16, 1)
            weights = torch.ones_like(reports[:, 1:], dtype=torch.float)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            model.train()
            for _ in range(200):
                optimizer.zero_grad()
                encoder_features = model.encoder(images)
                visual = encoder_features[:, 1:]
                logits, _ = model.decoder(reports[:, :-1], visual)
                rg_loss = term_weighted_cross_entropy(
                    logits, reports[:, 1:], weights, pad_idx=cfg.pad_idx
                )
                rg_loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                generated = model.generate(images, bos_idx=1, eos_idx=2)

            def content_tokens(sequence):
                tokens = []
                for token in sequence:
                    if token == 2:
                        break
                    if token not in (0, 1):
                        tokens.append(token)
                return tokens

            hypotheses = [content_tokens(row.tolist()) for row in generated]
            references = [content_tokens(row.tolist()) for row in reports]
            bleu4 = corpus_bleu4(hypotheses, references)

            self.assertLess(rg_loss.item(), 0.01)
            self.assertGreaterEqual(bleu4, 0.9)
            print(f"Overfit smoke: RG={rg_loss.item():.6f}, BLEU-4={bleu4:.4f}")
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
