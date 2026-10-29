"""
config.py — Toàn bộ siêu tham số của mô hình và quá trình huấn luyện.
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    # --- Ảnh đầu vào ---
    img_size: int = 448        # Kích thước ảnh vuông sau resize
    patch_size: int = 32       # Mỗi patch là 32×32 pixel → 14×14 = 196 patches/ảnh
    in_channels: int = 1       # X-quang grayscale = 1 kênh

    # --- Kích thước chung ---
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # --- Văn bản ---
    max_seq_len: int = 60      # Độ dài tối đa báo cáo (số token)
    vocab_size: int = 1000     # Cập nhật sau khi build vocab từ dataset
    pad_idx: int = 0           # Index token <pad>

    # --- Nhãn bệnh ---
    num_mlc_labels: int = 14   # 14 nhãn theo CheXpert

    # --- Bộ nhớ quan hệ (Relational Memory) ---
    memory_slots: int = 40     # Số slot bộ nhớ
    memory_head: int = 8       # Số attention head trong bộ nhớ

    # --- Tham số ITM ---
    itm_alpha: float = 0.5     # Mũ tính trọng số theo vị trí từ
    itm_epsilon: float = 0.05  # Margin trong triplet loss
    itm_mu: float = 0.025      # Hệ số L2 regularization

    # --- IU-Xray dùng 2 view ---
    use_paired_views: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 30
    lr_encoder: float = 5e-5  # LR nhỏ hơn cho encoder
    lr_rest: float = 1e-4     # LR cho decoder + head
    weight_decay: float = 1e-4

    # Hệ số các thành phần loss
    lambda_rg: float = 1.0
    lambda_mlc: float = 2.0
    lambda_itm: float = 5.0
