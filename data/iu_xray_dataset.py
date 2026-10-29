"""
iu_xray_dataset.py — PyTorch Dataset đọc dữ liệu IU-Xray đã tiền xử lý.
"""
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class IUXrayDataset(Dataset):
    """
    Dataset đọc ảnh và nhãn từ file processed_<split>.json.
    Mỗi sample trả về:
      - images: (2, 1, img_size, img_size) — 2 view chụp X-quang.
      - report_ids: (max_len,) — chuỗi token ID.
      - term_weights: (max_len - 1,) — trọng số ứng với các target report_ids[:, 1:].
      - mlc_labels: (14,) — nhãn phân loại bệnh.
    """
    def __init__(
        self,
        processed_json: str,
        image_root:     str,
        max_len:        int = 60,
        img_size:       int = 448,
    ):
        with open(processed_json) as f:
            raw_items = json.load(f)

        # Chỉ giữ các mẫu có tối thiểu 2 góc chụp
        self.items = [d for d in raw_items if len(d.get("image_path", [])) >= 2]
        n_removed = len(raw_items) - len(self.items)
        if n_removed > 0:
            print(f"[IUXrayDataset] Bỏ qua {n_removed} mẫu không đủ 2 ảnh.")

        self.image_root = image_root
        self.max_len    = max_len

        # Chuẩn hóa ảnh: grayscale, resize và chuyển sang tensor
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        full_path = os.path.join(self.image_root, rel_path)
        img = Image.open(full_path).convert("L")
        return self.transform(img)

    def __getitem__(self, idx: int):
        d = self.items[idx]

        # Lấy 2 ảnh đầu tiên
        paths = d["image_path"][:2]
        images = torch.stack([self._load_image(p) for p in paths], dim=0)

        # Padding hoặc cắt ngắn chuỗi token theo max_len
        ids = d["report_ids"][:self.max_len]
        ids = ids + [0] * (self.max_len - len(ids))

        if "term_weights" not in d:
            raise ValueError(
                "Processed report thiếu term_weights; hãy chạy lại preprocess_iu_xray.py."
            )
        weights = d["term_weights"][:self.max_len]
        weights = weights + [0.0] * (self.max_len - len(weights))

        mlc = torch.tensor(d["mlc_labels"], dtype=torch.float)
        # Bỏ trọng số tại BOS vì targets được dịch phải một vị trí.
        target_weights = torch.tensor(weights[1:], dtype=torch.float)
        return images, torch.tensor(ids, dtype=torch.long), mlc, target_weights
