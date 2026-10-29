"""
train.py — Vòng lặp huấn luyện MCT.
=====================================
- Không có --data_dir  : chạy với DUMMY data (kiểm tra luồng/shape).
- Có --data_dir        : train thật trên IU-Xray (cần chạy preprocess trước).

Cấu hình train theo bài báo (Sec. IV-B.2):
  - batch_size = 16
  - epochs = 30
  - lr_encoder = 5e-5 (encoder ViT có lr nhỏ hơn)
  - lr_rest    = 1e-4 (decoder + head)
  - Loss: λ_RG=1.0, λ_MLC=2.0, λ_ITM=5.0 — Eq.(12), Bảng IV

Ví dụ:
  python data/preprocess_iu_xray.py --data_dir ./iu_xray_data
  python train.py --data_dir ./iu_xray_data --epochs 30
"""
import os
import argparse
import json
import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig
from model.full_model import MulticriteriaTransformer
from data.tokenizer import SimpleTokenizer


# ==================== DUMMY data (không cần dataset thật) ====================
class DummyDataIterator:
    """
    Tạo batch giả để kiểm tra luồng forward/backward mà không cần dataset.
    Dùng với img_size=448, patch_size=32 (đúng theo bài báo).
    """
    def __init__(self, mcfg, tcfg, num_batches: int = 5):
        self.mcfg, self.tcfg = mcfg, tcfg
        self.num_batches = num_batches

    def __iter__(self):
        B, T = self.tcfg.batch_size, self.mcfg.max_seq_len
        for _ in range(self.num_batches):
            # Mỗi mẫu IU-Xray gồm frontal và lateral, theo đúng thứ tự hai view.
            images = torch.randn(
                B, 2, self.mcfg.in_channels, self.mcfg.img_size, self.mcfg.img_size
            )
            report_ids = torch.randint(1, self.mcfg.vocab_size, (B, T))
            mlc_labels = torch.randint(0, 2, (B, self.mcfg.num_mlc_labels)).float()
            term_weights = torch.ones(B, T - 1)
            yield images, report_ids, mlc_labels, term_weights


def build_optimizer(model: MulticriteriaTransformer, tcfg: TrainConfig):
    """
    Tạo optimizer với learning rate riêng cho encoder và phần còn lại.

    Bài báo (Sec. IV-B.2): lr_encoder = 5e-5, lr_rest = 1e-4.
    ViT encoder có lr nhỏ hơn để tránh overfit (đặc biệt khi encoder đã pretrain).

    Args:
        model: MulticriteriaTransformer
        tcfg : TrainConfig

    Returns:
        torch.optim.Adam với 2 param group khác lr.
    """
    encoder_params = list(model.encoder.parameters())
    rest_params    = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    return torch.optim.Adam(
        [
            {"params": encoder_params, "lr": tcfg.lr_encoder},
            {"params": rest_params,    "lr": tcfg.lr_rest},
        ],
        weight_decay=tcfg.weight_decay,
    )


def maybe_preprocess(data_dir: str):
    """
    Nếu chưa có file processed_train.json thì tự chạy preprocess.
    Gọi khi train.py được chạy trực tiếp mà chưa preprocess.
    """
    processed_paths = [
        os.path.join(data_dir, f"processed_{split}.json")
        for split in ("train", "val", "test")
    ]
    needs_preprocess = not all(os.path.exists(path) for path in processed_paths)
    if not needs_preprocess:
        for path in processed_paths:
            with open(path, encoding="utf-8") as stream:
                records = json.load(stream)
            if any("term_weights" not in record for record in records):
                needs_preprocess = True
                break

    if needs_preprocess:
        print("Thiếu processed split hoặc TF-IDF theo báo cáo — chạy lại preprocess ...")
        from data.preprocess_iu_xray import main as preprocess_main
        import sys
        sys.argv = ["preprocess", "--data_dir", data_dir]
        preprocess_main()


def main():
    ap = argparse.ArgumentParser(description="Huấn luyện MCT (IEEE TMI 2022).")
    ap.add_argument("--data_dir",   default=None,   help="Thư mục IU-Xray (annotation.json + images/)")
    ap.add_argument("--epochs",     type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    args = ap.parse_args()

    mcfg, tcfg = ModelConfig(), TrainConfig()
    if args.epochs:     tcfg.epochs     = args.epochs
    if args.batch_size: tcfg.batch_size = args.batch_size

    if args.data_dir:
        # ===== Chế độ train thật với IU-Xray =====
        maybe_preprocess(args.data_dir)

        tokenizer = SimpleTokenizer.load(os.path.join(args.data_dir, "vocab.json"))
        mcfg.vocab_size       = len(tokenizer)
        mcfg.pad_idx          = 0
        mcfg.use_paired_views = True   # IU-Xray luôn dùng 2 view

        from data.iu_xray_dataset import IUXrayDataset
        ds = IUXrayDataset(
            os.path.join(args.data_dir, "processed_train.json"),
            image_root=args.data_dir,
            max_len=mcfg.max_seq_len,
            img_size=mcfg.img_size,    # 448 — Sec. IV-B.2
        )
        train_loader = DataLoader(
            ds, batch_size=tcfg.batch_size, shuffle=True, num_workers=0
        )
        print(f"Train thật: {len(ds)} mẫu, vocab={mcfg.vocab_size}, img_size={mcfg.img_size}")
    else:
        # ===== Chế độ DUMMY (kiểm tra luồng) =====
        train_loader = DummyDataIterator(mcfg, tcfg)
        print("Chạy với DUMMY data (img_size=448). Dùng --data_dir để train thật.")

    model     = MulticriteriaTransformer(mcfg, tcfg)
    optimizer = build_optimizer(model, tcfg)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Số tham số: {n_params/1e6:.2f}M")

    for epoch in range(tcfg.epochs):
        model.train()
        for step, (images, report_ids, mlc_labels, term_weights) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward — truyền epoch để log (không ảnh hưởng đến loss hệ số)
            loss, loss_dict = model(
                images, report_ids, mlc_labels, term_weights, epoch
            )
            loss.backward()

            # Gradient clipping — tránh gradient exploding khi train deep model
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if step % 10 == 0:
                print(
                    f"[epoch {epoch:02d} step {step:04d}] " +
                    "  ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
                )

        # Lưu checkpoint sau mỗi epoch
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = f"checkpoints/epoch{epoch:02d}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Đã lưu checkpoint: {ckpt_path}")

    print("Xong.")


if __name__ == "__main__":
    main()
