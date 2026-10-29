"""
losses.py — Định nghĩa các hàm loss: Generation (RG), Classification (MLC), Matching (ITM).
"""
import torch
import torch.nn.functional as F


def term_weighted_cross_entropy(
    logits:            torch.Tensor,   # (B, T, V)
    targets:           torch.Tensor,   # (B, T)
    term_weights:      torch.Tensor,   # (B, T) — TF-IDF theo từng vị trí báo cáo
    pad_idx:           int,
) -> torch.Tensor:
    """
    Cross-Entropy có trọng số từ vựng (Term-Weighted CE).
    Chuẩn hóa bằng tổng trọng số của các token hợp lệ thay vì số lượng token đơn thuần.
    """
    if logits.shape[:2] != targets.shape or targets.shape != term_weights.shape:
        raise ValueError(
            "logits, targets và term_weights phải có shape (B,T,V), (B,T), (B,T)."
        )

    B, T, V = logits.shape
    ce = F.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        ignore_index=pad_idx,
        reduction="none",
    ).reshape(B, T)

    # PAD luôn có trọng số 0, kể cả khi dữ liệu preprocess không nhất quán.
    weights = term_weights.to(device=logits.device, dtype=ce.dtype)
    weights = weights.masked_fill(targets == pad_idx, 0.0)
    numerator = (ce * weights).sum()
    denominator = weights.sum().clamp(min=1e-8)
    return numerator / denominator


def mlc_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Binary Cross-Entropy Loss cho bài toán phân loại đa nhãn.
    """
    return F.binary_cross_entropy_with_logits(logits, labels)


def itm_triplet_loss(
    score_pos: torch.Tensor,   # (B,) — điểm tương đồng cặp ảnh-báo cáo đúng
    score_neg: torch.Tensor,   # (B,) — điểm tương đồng cặp sai
    epsilon:   float = 0.05,   # margin
    mu_itm:    float = 0.025,  # hệ số L2 regularization
) -> torch.Tensor:
    """
    Triplet loss kết hợp L2 regularization:
    - Ép điểm cặp đúng lớn hơn cặp sai ít nhất một khoảng epsilon.
    - Giữ biên độ điểm số không bị bão hòa quá mức.
    """
    triplet_part = torch.relu(score_neg - score_pos + epsilon)
    l2_part = mu_itm * (score_neg.pow(2) + score_pos.pow(2))
    return (triplet_part + l2_part).mean()
