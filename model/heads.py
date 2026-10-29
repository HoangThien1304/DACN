"""
heads.py — Các head dự đoán phụ trợ: MLC và ITM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_same_matrix(report_ids: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Tính same[i,j] cho chuỗi token, bỏ qua các vị trí PAD ở cuối báo cáo."""
    left = report_ids.unsqueeze(1)
    right = report_ids.unsqueeze(0)
    both_pad = (left == pad_idx) & (right == pad_idx)
    return ((left == right) | both_pad).all(dim=-1)


def sample_negatives(same: torch.Tensor, generator=None):
    """Chọn ngẫu nhiên một báo cáo khác nội dung cho mỗi anchor có ứng viên."""
    if same.ndim != 2 or same.shape[0] != same.shape[1]:
        raise ValueError("same phải là ma trận vuông shape (B,B).")

    candidates = ~same.to(dtype=torch.bool)
    valid = candidates.any(dim=1)
    random_scores = torch.rand(
        same.shape, device=same.device, generator=generator
    )
    random_scores = random_scores.masked_fill(~candidates, -1.0)
    negative_indices = random_scores.argmax(dim=1)
    return negative_indices, valid


class MLCHead(nn.Module):
    """
    Multi-Label Classification Head.
    Chiếu CLS token từ encoder thành logits cho các nhãn phân loại bệnh.
    """
    def __init__(self, d_model: int, num_labels: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, num_labels)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_token: (B, d_model)
        Returns:
            logits: (B, num_labels)
        """
        return self.proj(self.norm(cls_token))


class ITMScorer(nn.Module):
    """
    Image-Text Matching Scorer.
    Tính điểm tương đồng giữa ảnh và chuỗi văn bản thông qua hidden state của decoder.
    Dùng trọng số theo vị trí từ (temporal weight) để ưu tiên các từ sinh về sau.
    """
    def __init__(self, d_model: int, alpha: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.alpha = alpha

    def forward(
        self,
        v: torch.Tensor,
        h_seq: torch.Tensor,
        token_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            v: (B, d_model) — đặc trưng ảnh đã pooling từ CLS của các view.
            h_seq: (B, Nr, d_model) — chuỗi hidden states từ decoder.
            token_mask: (B, Nr) — True cho bước token hợp lệ, False cho PAD.
        Returns:
            scores: (B,) — điểm tương đồng trung bình có trọng số.
        """
        B, Nr, D = h_seq.shape
        if v.shape != (B, D) or D != self.d_model:
            raise ValueError("v và h_seq phải có cùng batch, chiều ẩn d_model.")

        # Điểm tương đồng ảnh-văn bản tại từng bước decoder.
        s_seq = self.step_scores(v, h_seq)

        if token_mask is None:
            token_mask = torch.ones(B, Nr, dtype=torch.bool, device=h_seq.device)
        elif token_mask.shape != (B, Nr):
            raise ValueError("token_mask phải có shape (B,Nr) như chuỗi hidden states.")
        else:
            token_mask = token_mask.to(device=h_seq.device, dtype=torch.bool)

        # Đánh số lại chỉ các token hợp lệ từ i=1 để PAD không ảnh hưởng lambda_i.
        lengths = token_mask.sum(dim=1, keepdim=True)
        token_positions = token_mask.to(torch.long).cumsum(dim=1)
        remaining = (lengths - token_positions + 1).clamp(min=1)
        lam = remaining.to(dtype=h_seq.dtype).pow(-self.alpha)
        lam = lam * token_mask.to(dtype=h_seq.dtype)

        weighted_sum = (lam * s_seq).sum(dim=-1)
        return weighted_sum / lam.sum(dim=-1).clamp_min(torch.finfo(lam.dtype).eps)

    def step_scores(self, v: torch.Tensor, h_seq: torch.Tensor) -> torch.Tensor:
        """Trả s_i = cos(v, f_i) cho từng bước, shape (B,Nr)."""
        if h_seq.ndim != 3:
            raise ValueError("h_seq phải có shape (B,Nr,d_model).")
        B, _, D = h_seq.shape
        if v.shape != (B, D) or D != self.d_model:
            raise ValueError("v và h_seq phải có cùng batch, chiều ẩn d_model.")
        return F.cosine_similarity(v.unsqueeze(1), h_seq, dim=-1)
