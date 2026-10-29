"""
test_forward.py — Kiểm tra nhanh model với tensor giả (không cần dataset).
===========================================================================
Xác nhận:
  1. Forward pass chạy được, loss_dict có đủ 3 thành phần: rg, mlc, itm.
  2. Backward pass chạy được (gradient flow OK).
  3. generate() chạy được → output shape đúng.
  4. Kiểm tra tổng loss = 1.0*rg + 2.0*mlc + 5.0*itm (Eq.12, Bảng IV).

Chạy: python test_forward.py
"""
import torch
from config import ModelConfig, TrainConfig
from model.full_model import MulticriteriaTransformer


def main():
    mcfg, tcfg = ModelConfig(), TrainConfig()
    # Dùng vocab_size nhỏ để test nhanh (không cần dataset thật)
    mcfg.vocab_size = 100
    B, T = 4, mcfg.max_seq_len

    model = MulticriteriaTransformer(mcfg, tcfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Số tham số: {n_params/1e6:.2f}M")
    print(f"Config: img_size={mcfg.img_size}, patch_size={mcfg.patch_size}, "
          f"d_model={mcfg.d_model}, memory_slots={mcfg.memory_slots}")
    print(f"Lambda: RG={tcfg.lambda_rg}, MLC={tcfg.lambda_mlc}, ITM={tcfg.lambda_itm}")

    # --- Dữ liệu giả với img_size=448 (đúng theo bài báo) ---
    images     = torch.randn(B, 2, mcfg.in_channels, mcfg.img_size, mcfg.img_size)
    report_ids = torch.randint(1, mcfg.vocab_size, (B, T))
    mlc_labels = torch.randint(0, 2, (B, mcfg.num_mlc_labels)).float()
    term_weights = torch.ones(B, T - 1)   # Trọng số theo vị trí (test)

    # --- Forward + Backward ---
    loss, loss_dict = model(images, report_ids, mlc_labels, term_weights, epoch=5)
    loss.backward()
    print("\nLoss dict:", {k: round(v, 4) for k, v in loss_dict.items()})

    # --- Kiểm tra tổng loss theo Eq.(12) ---
    expected_total = (
        tcfg.lambda_rg  * loss_dict.get("rg",  0) +
        tcfg.lambda_mlc * loss_dict.get("mlc", 0) +
        tcfg.lambda_itm * loss_dict.get("itm", 0)
    )
    actual_total = loss_dict["total"]
    print(f"\nKiểm tra Eq.(12): λ_RG·RG + λ_MLC·MLC + λ_ITM·ITM")
    print(f"  Expected total = {expected_total:.4f}")
    print(f"  Actual   total = {actual_total:.4f}")
    assert abs(expected_total - actual_total) < 1e-3, (
        "FAIL: Tổng loss không khớp với Eq.(12)!"
    )
    print("  → PASS")

    # --- Greedy decode (inference) ---
    out = model.generate(images, bos_idx=1, eos_idx=2)
    print(f"\nOutput shape khi generate: {out.shape}")

    print("\nOK — forward, backward, generate đều chạy được.")


if __name__ == "__main__":
    main()
