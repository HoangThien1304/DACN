# Multicriteria Transformer — Skeleton code

Dựng lại bài báo *"Automated Radiographic Report Generation Purely on Transformer:
A Multicriteria Supervised Approach"* (IEEE TMI 2022). Bộ khung này **chưa cần dataset**
— dùng tensor giả để verify kiến trúc, shape và luồng train.

## Cấu trúc

```
mct_skeleton/
├── config.py                 # toàn bộ siêu tham số (sửa TODO khi có dataset)
├── train.py                  # vòng lặp train skeleton (DummyDataIterator)
├── test_forward.py           # kiểm tra forward/backward/generate
├── requirements.txt
└── model/
    ├── vit_encoder.py        # ViT encoder (3 lớp, patch embedding, CLS, 2-view)
    ├── decoder.py            # memory-augmented decoder (causal mask, cross-attn)
    ├── memory_layernorm.py   # Memory-driven LayerNorm
    ├── heads.py              # MLC, cosine ITM và chọn negative trong batch
    ├── losses.py             # RG+term-weight, MLC-BCE, ITM+temporal-weight
    └── full_model.py         # gộp tất cả + generate (greedy decode)
```

## Chạy thử

```bash
pip install -r requirements.txt
python test_forward.py     # in ra số tham số, loss dict, shape output
python train.py            # chạy vài bước train với dữ liệu giả
```

## Cắm IU-Xray thật vào chạy (đã có sẵn pipeline, đã test)

1. **Tải IU-Xray** về, cấu trúc thư mục:
   ```
   iu_xray_data/
   ├── annotation.json      # list [{id, image_path: ["images/...png",...], report: "..."}]
   └── images/              # file ảnh PNG/JPG
   ```
   Bản `annotation.json` + images được dùng chung trong các repo R2Gen/KERP/CoAtt;
   nếu chỉ có CSV gốc Open-i, hàm `build_annotation_from_csv` trong preprocess sẽ tự build.

2. **Tiền xử lý** (tự build vocab, term-weight, nhãn 14 bệnh):
   ```bash
   python data/preprocess_iu_xray.py --data_dir ./iu_xray_data
   ```

3. **Train** (IU-Xray 2 view frontal+lateral):
   ```bash
   python train.py --data_dir ./iu_xray_data --epochs 30
   ```

4. **Đánh giá** (sinh báo cáo test + BLEU/METEOR/ROUGE-L/CIDEr):
   ```bash
   python eval.py --data_dir ./iu_xray_data --checkpoint checkpoints/epochXX.pt
   ```

5. **Demo**: viết giao diện Streamlit/Gradio — chọn ảnh X-quang → gọi `model.generate()` → in báo cáo.

## Lưu ý
- Nhãn MLC hiện dùng rule-based có xử lý phủ định (đã test). Để chính xác như paper,
  chạy **CheXpert Labeler** chính thức (Stanford) trên báo cáo rồi ghi đè `mlc_labels`.
- `test_iu_xray_pipeline.py`: tạo data IU-Xray giả và chạy toàn bộ pipeline để kiểm tra
  mà không cần tải dataset thật.
- `test_forward.py`: kiểm tra model với tensor giả.

## Đơn giản hoá so với bài báo

- Bài báo không nêu trọng số cho token đặc biệt: `<unk>` được đặt trọng số TF-IDF bằng 0; BOS và EOS được đặt trọng số 1. PAD luôn bằng 0.
- Bài báo không quy định cách kết hợp hai CLS. Dự án nối CLS frontal rồi lateral, sau đó dùng `Linear(2D → D)` chung cho MLC và ITM.
- Với negative ITM, mỗi anchor lấy ngẫu nhiên một báo cáo khác nội dung trong batch. Anchor không có ứng viên hợp lệ bị bỏ khỏi ITM; số lượng được ghi trong `itm_skipped`.
- Dữ liệu IU-Xray thật chưa có trong workspace nên chưa tính được số anchor bị bỏ trên train và chưa chạy benchmark.
