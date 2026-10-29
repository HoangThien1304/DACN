"""
memory_layernorm.py — Relational Memory và Memory-Driven LayerNorm.
"""
import torch
import torch.nn as nn


class RelationalMemory(nn.Module):
    """
    Module bộ nhớ quan hệ — lưu giữ ngữ cảnh trong quá trình sinh từng từ.

    Mỗi bước sinh từ, bộ nhớ đọc thông tin từ hidden state hiện tại qua
    Multi-Head Attention, sau đó cập nhật trạng thái nội tại.
    Kết quả là vector m_t đại diện cho "những gì đã nhớ" đến bước này.

    Args:
        d_model   : Chiều ẩn chung.
        num_slots : Số slot bộ nhớ (mặc định 40).
        num_heads : Số attention head.
    """
    def __init__(self, d_model: int, num_slots: int = 40, num_heads: int = 8):
        super().__init__()
        self.num_slots = num_slots
        self.d_model   = d_model

        # Trạng thái bộ nhớ ban đầu — learnable, dùng chung mọi sample
        self.initial_memory = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)

        # Attention: bộ nhớ (Q) đọc từ hidden state (K, V)
        self.attn      = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn  = nn.LayerNorm(d_model)

        # FFN cập nhật nội tại các slot bộ nhớ
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def init_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Khởi tạo bộ nhớ cho batch mới. Trả về (B, num_slots, d_model)."""
        return self.initial_memory.expand(batch_size, -1, -1).contiguous()

    def forward(self, r_t: torch.Tensor, memory: torch.Tensor):
        """
        Cập nhật bộ nhớ từ hidden state hiện tại.

        Args:
            r_t    : (B, T, d_model) — hidden state từ decoder.
            memory : (B, num_slots, d_model) — trạng thái bộ nhớ hiện tại.

        Returns:
            new_memory : (B, num_slots, d_model) — bộ nhớ sau cập nhật.
            m_t        : (B, d_model) — vector đại diện để điều khiển LayerNorm.
        """
        # Bộ nhớ đọc thông tin từ hidden state qua attention
        attn_out, _ = self.attn(memory, r_t, r_t, need_weights=False)
        memory = self.norm_attn(memory + attn_out)

        # FFN cập nhật nội tại
        memory = self.norm_ffn(memory + self.ffn(memory))

        # Mean-pool các slot → vector đại diện ngắn gọn
        m_t = memory.mean(dim=1)   # (B, d_model)
        return memory, m_t


class MemoryDrivenLayerNorm(nn.Module):
    """
    LayerNorm có tham số điều chỉnh động theo trạng thái bộ nhớ m_t.

    Thay vì dùng gamma, beta cố định như LayerNorm thông thường,
    thêm phần bổ sung từ m_t:
        gamma_hat = gamma + W_gamma(m_t)
        beta_hat  = beta  + W_beta(m_t)

    Khởi tạo W_gamma, W_beta bằng zeros → ban đầu hoạt động như LayerNorm bình thường.

    Args:
        d_model    : Chiều ẩn.
        memory_dim : Chiều vector bộ nhớ m_t (thường = d_model).
    """
    def __init__(self, d_model: int, memory_dim: int = None):
        super().__init__()
        if memory_dim is None:
            memory_dim = d_model

        self.ln      = nn.LayerNorm(d_model, elementwise_affine=True)
        self.W_gamma = nn.Linear(memory_dim, d_model, bias=False)
        self.W_beta  = nn.Linear(memory_dim, d_model, bias=False)

        # Init zeros để lúc đầu không làm xáo trộn học ban đầu
        nn.init.zeros_(self.W_gamma.weight)
        nn.init.zeros_(self.W_beta.weight)

    def forward(self, r: torch.Tensor, m_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r   : (B, T, d_model) — tensor cần chuẩn hoá.
            m_t : (B, d_model)   — vector trạng thái bộ nhớ.

        Returns:
            (B, T, d_model)
        """
        x_norm = self.ln(r)

        # Tính offset từ bộ nhớ, unsqueeze để broadcast theo chiều T
        delta_gamma = self.W_gamma(m_t).unsqueeze(1)   # (B, 1, d_model)
        delta_beta  = self.W_beta(m_t).unsqueeze(1)    # (B, 1, d_model)

        return x_norm * (1.0 + delta_gamma) + delta_beta
