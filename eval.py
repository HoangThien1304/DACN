"""
eval.py — Đánh giá trên tập test: sinh báo cáo -> tính BLEU/METEOR/ROUGE-L/CIDEr.
Yêu cầu: pip install pycocoevalcap (cần java để chạy METEOR/CIDEr).
Chạy: python eval.py --data_dir ./iu_xray_data --checkpoint checkpoints/best.pt
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig
from model.full_model import MulticriteriaTransformer
from data.tokenizer import SimpleTokenizer, BOS, EOS
from data.iu_xray_dataset import IUXrayDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    mcfg, tcfg = ModelConfig(), TrainConfig()
    tokenizer = SimpleTokenizer.load(os.path.join(args.data_dir, "vocab.json"))
    mcfg.vocab_size = len(tokenizer)
    mcfg.use_paired_views = True

    model = MulticriteriaTransformer(mcfg, tcfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    ds = IUXrayDataset(os.path.join(args.data_dir, "processed_test.json"), args.data_dir, mcfg.max_seq_len)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    hyps, refs = [], []
    bos_idx = tokenizer.vocab[BOS]
    eos_idx = tokenizer.vocab[EOS]
    with torch.no_grad():
        for images, report_ids, _, _ in loader:
            out_ids = model.generate(images, bos_idx, eos_idx)
            for i in range(out_ids.size(0)):
                hyps.append(tokenizer.decode(out_ids[i].tolist()))
                refs.append(tokenizer.decode(report_ids[i].tolist()))

    try:
        from pycocoevalcap.eval import COCOEvalCap
        # TODO: đóng gói hyps/refs theo định dạng COCO rồi gọi COCOEvalCap.
        # Skeleton: in ra vài cặp ví dụ trước khi tích hợp pycocoevalcap đầy đủ.
        print("=== Ví dụ (generated | ground-truth) ===")
        for h, r in list(zip(hyps, refs))[:3]:
            print(f"GEN : {h}\nGT  : {r}\n")
        print("Cài pycocoevalcap để tính BLEU/METEOR/ROUGE-L/CIDEr đầy đủ.")
    except ImportError:
        print("Chưa cài pycocoevalcap. Vẫn in ra báo cáo sinh được để kiểm tra định tính.")
        for h, r in list(zip(hyps, refs))[:3]:
            print(f"GEN: {h}\nGT : {r}\n")


if __name__ == "__main__":
    main()
