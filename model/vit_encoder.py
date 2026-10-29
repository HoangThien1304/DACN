"""
vit_encoder.py — Vision Transformer Encoder trích xuất đặc trưng ảnh X-quang.
"""
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Chia ảnh thành các patch vuông, làm phẳng và chiếu tuyến tính lên chiều d_model.
    """
    def __init__(self, img_size: int, patch_size: int, in_channels: int, d_model: int):
        super().__init__()
        assert img_size % patch_size == 0, (
            f"img_size={img_size} phải chia hết cho patch_size={patch_size}"
        )
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * patch_size * in_channels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_patches, d_model)
        """
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.reshape(B, C, -1, p * p).permute(0, 2, 1, 3)
        x = x.reshape(B, -1, C * p * p)
        return self.proj(x)


class TransformerEncoderLayer(nn.Module):
    """
    Khối Transformer Encoder tiêu chuẩn: Self-Attention kết hợp Feed-Forward.
    """
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        a, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Encoder xử lý ảnh đơn hoặc cặp 2 góc chụp (frontal + lateral):
    - Trả về đặc trưng từng vùng patch cho decoder.
    - Trả về CLS token đại diện toàn bộ ca chụp cho bài toán phân loại bệnh.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbedding(
            cfg.img_size, cfg.patch_size, cfg.in_channels, cfg.d_model
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.cls_proj = nn.Linear(2 * cfg.d_model, cfg.d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, cfg.d_model) * 0.02
        )
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout)
            for _ in range(cfg.num_encoder_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)

    def encode_single(self, x: torch.Tensor):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x[:, 1:], x[:, 0]

    def forward(self, images: torch.Tensor):
        """
        Args:
            images: (B, 2, C, H, W), thứ tự view là frontal rồi lateral.
        Returns:
            encoder_features: (B, 1 + 2 * num_patches, d_model).
                Vị trí 0 là CLS hai view đã nối và chiếu; phần còn lại là patch.
        """
        if images.dim() == 4:
            images = images.unsqueeze(1)
        B, V = images.shape[:2]
        if V != 2:
            raise ValueError(f"ViTEncoder cần đúng 2 view (frontal, lateral); nhận được V={V}.")

        regionals, clss = [], []
        for v in range(V):
            # Mã hoá riêng từng góc chụp bằng cùng bộ patch embedding và encoder.
            reg, cls = self.encode_single(images[:, v])
            regionals.append(reg)
            clss.append(cls)

        # Nối đặc trưng patch của frontal và lateral cho cross-attention decoder.
        regional_features = torch.cat(regionals, dim=1)
        # Nối hai CLS rồi chiếu 2D -> D; không lấy trung bình hai CLS.
        cls_global = self.cls_proj(torch.cat(clss, dim=-1))
        # Đặt CLS hợp nhất ở đầu chuỗi để MLC và ITM cùng dùng đặc trưng này.
        return torch.cat([cls_global.unsqueeze(1), regional_features], dim=1)
