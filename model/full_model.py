"""
full_model.py — Mô hình tổng thể kết hợp Encoder, Decoder và các nhánh Loss.
"""
import torch
import torch.nn as nn

from .vit_encoder import ViTEncoder
from .decoder import MemoryAugmentedDecoder
from .heads import MLCHead, ITMScorer, build_same_matrix, sample_negatives
from .losses import (
    term_weighted_cross_entropy,
    mlc_bce_loss,
    itm_triplet_loss,
)


class MulticriteriaTransformer(nn.Module):
    """
    Mô hình tích hợp:
    1. ViT Encoder trích xuất đặc trưng vùng ảnh và CLS token toàn cục.
    2. Memory-Augmented Decoder sinh báo cáo tuần tự.
    3. Nhánh phân loại đa nhãn (MLC) từ CLS token.
    4. Nhánh so khớp ảnh - văn bản (ITM) thông qua Decoder và Triplet Loss.
    """
    def __init__(self, mcfg, tcfg):
        super().__init__()
        self.mcfg = mcfg
        self.tcfg = tcfg
        self.encoder = ViTEncoder(mcfg)
        self.decoder = MemoryAugmentedDecoder(mcfg)
        self.mlc_head = MLCHead(mcfg.d_model, mcfg.num_mlc_labels)
        self.itm_scorer = ITMScorer(mcfg.d_model, mcfg.itm_alpha)

    def forward(
        self,
        images:            torch.Tensor,
        report_ids:        torch.Tensor,
        mlc_labels:        torch.Tensor = None,
        term_weights:      torch.Tensor = None,
        epoch:             int = 0,
    ):
        """
        Forward pass tính tổng các hàm mất mát huấn luyện:
        - L_RG: sinh báo cáo có trọng số từ.
        - L_MLC: phân loại nhãn bệnh.
        - L_ITM: tương đồng cặp đúng và cặp sai hoán đổi trong batch.
        """
        # 1. Trích xuất đặc trưng ảnh
        encoder_features = self.encoder(images)
        cls_global = encoder_features[:, 0]
        regional_features = encoder_features[:, 1:]

        # 2. Decoder với báo cáo đúng (positive pair)
        dec_input_pos = report_ids[:, :-1]
        logits, h_pos = self.decoder(dec_input_pos, regional_features)

        total_loss = torch.tensor(0.0, device=images.device, requires_grad=True)
        loss_dict = {}

        # 3. Loss sinh báo cáo (RG)
        if term_weights is not None:
            l_rg = term_weighted_cross_entropy(
                logits,
                report_ids[:, 1:],
                term_weights,
                self.mcfg.pad_idx,
            )
            total_loss = total_loss + self.tcfg.lambda_rg * l_rg
            loss_dict["rg"] = l_rg.item()

        # 4. Loss phân loại nhãn bệnh (MLC)
        if mlc_labels is not None:
            mlc_logits = self.mlc_head(cls_global)
            l_mlc = mlc_bce_loss(mlc_logits, mlc_labels)
            total_loss = total_loss + self.tcfg.lambda_mlc * l_mlc
            loss_dict["mlc"] = l_mlc.item()

        # 5. Loss ITM: loại cặp trùng, rồi lấy ngẫu nhiên một báo cáo âm hợp lệ.
        same = build_same_matrix(report_ids, self.mcfg.pad_idx)
        negative_indices, valid_anchors = sample_negatives(same)
        n_skipped = int((~valid_anchors).sum().item())

        if valid_anchors.any():
            valid_rows = valid_anchors.nonzero(as_tuple=False).squeeze(1)
            negative_ids = report_ids[negative_indices[valid_rows]]
            negative_input = negative_ids[:, :-1]
            _, h_neg = self.decoder(negative_input, regional_features[valid_rows])

            positive_mask = report_ids[:, 1:] != self.mcfg.pad_idx
            negative_mask = negative_ids[:, 1:] != self.mcfg.pad_idx
            score_pos = self.itm_scorer(
                cls_global[valid_rows], h_pos[valid_rows], positive_mask[valid_rows]
            )
            score_neg = self.itm_scorer(
                cls_global[valid_rows], h_neg, negative_mask
            )
            l_itm = itm_triplet_loss(
                score_pos, score_neg,
                epsilon=self.mcfg.itm_epsilon,
                mu_itm=self.mcfg.itm_mu,
            )
        else:
            # Không có cặp âm hợp lệ: loss bằng 0 và vẫn nối với graph để backward an toàn.
            l_itm = regional_features.sum() * 0.0

        total_loss = total_loss + self.tcfg.lambda_itm * l_itm
        loss_dict["itm"] = l_itm.item()
        loss_dict["itm_skipped"] = n_skipped

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict

    @torch.no_grad()
    def generate(
        self, images: torch.Tensor, bos_idx: int, eos_idx: int
    ) -> torch.Tensor:
        """
        Sinh báo cáo tự hồi quy bằng chiến lược greedy decoding.
        """
        encoder_features = self.encoder(images)
        regional_features = encoder_features[:, 1:]
        B = images.size(0)
        ys = torch.full((B, 1), bos_idx, dtype=torch.long, device=images.device)

        for _ in range(self.mcfg.max_seq_len - 1):
            logits, _ = self.decoder(ys, regional_features)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if (next_tok == eos_idx).all():
                break
        return ys
