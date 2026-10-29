"""
test_iu_xray_pipeline.py — Test toàn bộ pipeline IU-Xray với data MOCK (không cần tải dataset thật).
Tạo 3 mẫu giả, chạy preprocess -> Dataset -> model forward/backward.
"""
import os, json, subprocess, sys
import torch
from PIL import Image
import numpy as np

MOCK = "/tmp/mock_iu"
os.makedirs(f"{MOCK}/images", exist_ok=True)

# 1) Tạo ảnh PNG giả (grayscale 512x512 nhiễu)
for name in ["a_front.png", "a_lat.png", "b_front.png", "b_lat.png", "c_front.png", "c_lat.png"]:
    arr = (np.random.rand(512, 512) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(f"{MOCK}/images/{name}")

# 2) annotation.json giả (định dạng chuẩn R2Gen/KERP)
annotation = [
    {"id": "1", "image_path": ["images/a_front.png", "images/a_lat.png"],
     "report": "the heart is enlarged. cardiomegaly is present. no pleural effusion."},
    {"id": "2", "image_path": ["images/b_front.png", "images/b_lat.png"],
     "report": "left pneumothorax with slight mediastinal shift. lungs are otherwise clear."},
    {"id": "3", "image_path": ["images/c_front.png", "images/c_lat.png"],
     "report": "no acute cardiopulmonary abnormality. the lungs are clear. normal study."},
]
with open(f"{MOCK}/annotation.json", "w") as f:
    json.dump(annotation, f)

# 3) Chạy preprocess
print("=== Chạy preprocess ===")
subprocess.run([sys.executable, "data/preprocess_iu_xray.py", "--data_dir", MOCK], check=True)

# 4) Load dataset + chạy 1 batch qua model
print("\n=== Load dataset + model ===")
from data.tokenizer import SimpleTokenizer
from data.iu_xray_dataset import IUXrayDataset
from torch.utils.data import DataLoader
from config import ModelConfig, TrainConfig
from model.full_model import MulticriteriaTransformer

tokenizer = SimpleTokenizer.load(f"{MOCK}/vocab.json")
mcfg, tcfg = ModelConfig(), TrainConfig()
mcfg.vocab_size = len(tokenizer)
mcfg.use_paired_views = True
tcfg.batch_size = 2

ds = IUXrayDataset(f"{MOCK}/processed_train.json", MOCK, mcfg.max_seq_len, mcfg.img_size)
loader = DataLoader(ds, batch_size=tcfg.batch_size, shuffle=False)
images, report_ids, mlc_labels, term_weights = next(iter(loader))
print("Batch shapes:", images.shape, report_ids.shape, mlc_labels.shape, term_weights.shape)
print("MLC labels mẫu (14 bệnh):", mlc_labels[0].tolist())

model = MulticriteriaTransformer(mcfg, tcfg)
loss, loss_dict = model(images, report_ids, mlc_labels, term_weights, epoch=3)
loss.backward()
print("Loss dict:", {k: round(v, 4) for k, v in loss_dict.items()})
print("\nOK — Pipeline IU-Xray chạy hoàn chỉnh (preprocess -> dataset -> train).")
