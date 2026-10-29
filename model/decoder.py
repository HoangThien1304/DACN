"""
decoder.py — Memory-augmented Transformer Decoder.
"""
import torch
import torch.nn as nn
from .memory_layernorm import RelationalMemory, MemoryDrivenLayerNorm


class MemoryAugmentedDecoderLayer(nn.Module):
    """
    Một lớp decoder sử dụng Memory-Driven LayerNorm thay cho LayerNorm thông thường.

    Luồng xử lý:
      1. Masked self-attention (chỉ nhìn các từ đã sinh)
      2. Cross-attention tới vùng ảnh (regional features)
      3. Feed-forward network
    Mỗi bước đều dùng MLN với vector bộ nhớ m_t để điều chỉnh chuẩn hoá.
    """
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout),
        )
        # Mỗi sublayer có một MLN riêng, cùng nhận m_t từ RelationalMemory
        self.mln1 = MemoryDrivenLayerNorm(d_model, d_model)
        self.mln2 = MemoryDrivenLayerNorm(d_model, d_model)
        self.mln3 = MemoryDrivenLayerNorm(d_model, d_model)

    def forward(
        self,
        x:           torch.Tensor,   # (B, T, d_model)
        visual_mem:  torch.Tensor,   # (B, V*N, d_model) — vùng ảnh
        causal_mask: torch.Tensor,   # (T, T) bool
        m_t:         torch.Tensor,   # (B, d_model) — vector bộ nhớ
    ) -> torch.Tensor:
        # Pre-norm masked self-attention. Keep the residual stream unnormalized.
        normalized = self.mln1(x, m_t)
        a, _ = self.self_attn(
            normalized, normalized, normalized,
            attn_mask=causal_mask, need_weights=False,
        )
        x = x + a

        # Pre-norm cross-attention to visual memory.
        normalized = self.mln2(x, m_t)
        a, _ = self.cross_attn(normalized, visual_mem, visual_mem, need_weights=False)
        x = x + a

        # Pre-norm feed-forward sublayer.
        x = x + self.ffn(self.mln3(x, m_t))
        return x


class MemoryAugmentedDecoder(nn.Module):
    """
    Toàn bộ decoder với N lớp và RelationalMemory dùng chung.

    Điểm đặc biệt: output của các lớp được trung bình hoá trước khi chiếu
    ra vocab — giúp tổng hợp thông tin từ nhiều mức độ trừu tượng khác nhau.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_idx)
        self.pos_embed   = nn.Parameter(torch.randn(1, cfg.max_seq_len, cfg.d_model) * 0.02)

        self.layers = nn.ModuleList([
            MemoryAugmentedDecoderLayer(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout)
            for _ in range(cfg.num_decoder_layers)
        ])

        # Bộ nhớ dùng chung cho tất cả lớp decoder
        self.relational_memory = RelationalMemory(cfg.d_model, cfg.memory_slots, cfg.memory_head)

        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head    = nn.Linear(cfg.d_model, cfg.vocab_size)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        """Mặt nạ nhân quả: True tại vị trí không được nhìn (j > i). Shape (T, T)."""
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, tgt_ids: torch.Tensor, visual_mem: torch.Tensor):
        """
        Args:
            tgt_ids   : (B, T) — token IDs đã shift-right.
            visual_mem: (B, V*N, d_model) — regional features từ encoder.

        Returns:
            logits : (B, T, vocab_size)
            h_seq  : (B, T, d_model) — hidden state dùng cho ITM scoring.
        """
        B, T = tgt_ids.shape
        x    = self.token_embed(tgt_ids) + self.pos_embed[:, :T, :]
        mask = self._causal_mask(T, x.device)

        # Khởi tạo bộ nhớ cho batch này
        mem_state = self.relational_memory.init_memory(B, x.device)

        layer_outs = []
        for layer in self.layers:
            # Cập nhật bộ nhớ từ hidden state hiện tại, lấy vector m_t
            mem_state, m_t = self.relational_memory(x, mem_state)
            x = layer(x, visual_mem, mask, m_t)
            layer_outs.append(x)

        # Trung bình hoá output các lớp để tổng hợp đặc trưng đa mức
        # F là đặc trưng hợp nhất theo phương trình (5), dùng trực tiếp cho ITM.
        fused_features = torch.stack(layer_outs, dim=0).mean(dim=0)
        # Giữ đường logits hiện có; chuẩn hoá đầu ra sẽ được đối chiếu ở mục 2.
        logits = self.lm_head(self.final_norm(fused_features))
        return logits, fused_features
