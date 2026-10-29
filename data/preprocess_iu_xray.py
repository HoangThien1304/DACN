"""
preprocess_iu_xray.py — Tiền xử lý dữ liệu IU-Xray.
Chia split train/val/test, tạo vocabulary, tính trọng số IDF và nhãn bệnh.
"""
import os
import json
import argparse
import random

try:
    from .tokenizer import SimpleTokenizer
    from .chexpert_labeler import rule_based_label
    from .tfidf import compute_idf_weights, encode_report_with_tfidf
except ImportError:
    from tokenizer import SimpleTokenizer
    from chexpert_labeler import rule_based_label
    from tfidf import compute_idf_weights, encode_report_with_tfidf


def load_annotation(data_dir: str):
    """
    Đọc file annotation.json hoặc chuyển đổi từ các file CSV gốc nếu có.
    """
    ann_path = os.path.join(data_dir, "annotation.json")
    if os.path.exists(ann_path):
        with open(ann_path) as f:
            return json.load(f)

    csv_reports = os.path.join(data_dir, "indiana_reports.csv")
    csv_proj = os.path.join(data_dir, "indiana_projections.csv")
    if os.path.exists(csv_reports) and os.path.exists(csv_proj):
        print("Đang đọc từ file CSV gốc để tạo danh sách dữ liệu...")
        return build_annotation_from_csv(csv_reports, csv_proj)

    raise FileNotFoundError(
        f"Không tìm thấy annotation.json hay indiana_*.csv trong {data_dir}."
    )


def build_annotation_from_csv(csv_reports: str, csv_proj: str):
    """
    Ghép findings và impression từ indiana_reports.csv cùng đường dẫn ảnh từ indiana_projections.csv.
    """
    import csv
    reports = {}
    with open(csv_reports) as f:
        for row in csv.DictReader(f):
            findings = (row.get("findings") or "").strip()
            impression = (row.get("impression") or "").strip()
            combined = (findings + " " + impression).strip()
            reports[row["uid"]] = combined

    proj = {}
    with open(csv_proj) as f:
        for row in csv.DictReader(f):
            proj.setdefault(row["uid"], []).append(row["filename"])

    return [
        {"id": uid, "image_path": imgs, "report": reports.get(uid, "")}
        for uid, imgs in proj.items()
        if uid in reports and reports.get(uid, "").strip()
    ]


def main():
    ap = argparse.ArgumentParser(description="Tiền xử lý tập dữ liệu IU-Xray.")
    ap.add_argument("--data_dir", required=True, help="Thư mục chứa annotation và ảnh")
    ap.add_argument("--max_len", type=int, default=60, help="Độ dài tối đa chuỗi token")
    ap.add_argument("--min_freq", type=int, default=1, help="Tần suất xuất hiện tối thiểu của từ")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    data = load_annotation(args.data_dir)

    # Giữ các ca chụp có đủ tối thiểu 2 ảnh để làm cặp view
    data_2view = [d for d in data if len(d.get("image_path", [])) >= 2]
    n_removed = len(data) - len(data_2view)
    if n_removed > 0:
        print(f"Bỏ qua {n_removed} mẫu không đủ 2 ảnh.")
    data = data_2view

    random.shuffle(data)
    n = len(data)
    splits = {
        "train": data[:int(0.8 * n)],
        "val": data[int(0.8 * n): int(0.9 * n)],
        "test": data[int(0.9 * n):],
    }
    print(f"Tổng số mẫu: {n} (train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])})")

    # Xây dựng từ điển chỉ từ tập train
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab([d["report"] for d in splits["train"]], min_freq=args.min_freq)
    tokenizer.save(os.path.join(args.data_dir, "vocab.json"))
    print(f"Kích thước từ điển: {len(tokenizer)}")

    # Tính IDF chỉ từ train; TF sẽ được tính riêng cho từng báo cáo.
    idf_weights = compute_idf_weights(tokenizer, [d["report"] for d in splits["train"]])
    print("Đã tính IDF trên tập train; TF-IDF sẽ được lưu theo từng vị trí báo cáo.")

    for split, items in splits.items():
        processed = []
        for d in items:
            report_ids, term_weights = encode_report_with_tfidf(
                d["report"], tokenizer, idf_weights, args.max_len
            )
            processed.append({
                "id": d.get("id", ""),
                "image_path": d["image_path"][:2],
                "report": d["report"],
                "report_ids": report_ids,
                "term_weights": term_weights,
                "mlc_labels": rule_based_label(d["report"]),
            })
        out_path = os.path.join(args.data_dir, f"processed_{split}.json")
        with open(out_path, "w") as f:
            json.dump(processed, f, indent=2)
        print(f"Đã lưu {out_path} ({len(processed)} mẫu)")

    print("\nHoàn tất tiền xử lý.")


if __name__ == "__main__":
    main()
