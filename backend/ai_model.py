"""Mô hình AI demo cho portal chẩn đoán X-quang ngực.

File này đóng vai trò 3 chức năng quan trọng:
1. định nghĩa 14 nhãn bệnh chuẩn theo chuẩn radiology
2. chuyển nhãn tiếng Anh sang tiếng Việt để hiển thị cho người dùng
3. tải checkpoint nếu có và dự đoán bệnh lý từ 2 ảnh frontal/lateral

Lưu ý: đây là baseline mẫu, không phải mô hình đã được kiểm định lâm sàng.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Danh sách 14 nhãn bệnh chuẩn được dùng trong bài toán multi-label classification
LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]

# Bản dịch tiếng Việt để frontend và báo cáo Word dễ đọc với người bệnh
LABELS_VI = {
    "Atelectasis": "Xẹp phổi", "Cardiomegaly": "Tim to",
    "Consolidation": "Đông đặc phổi", "Edema": "Phù phổi",
    "Enlarged Cardiomediastinum": "Bóng tim - trung thất to",
    "Fracture": "Gãy xương", "Lung Lesion": "Tổn thương phổi",
    "Lung Opacity": "Đám mờ phổi", "No Finding": "Không ghi nhận bất thường",
    "Pleural Effusion": "Tràn dịch màng phổi", "Pleural Other": "Bất thường màng phổi khác",
    "Pneumonia": "Viêm phổi", "Pneumothorax": "Tràn khí màng phổi",
    "Support Devices": "Dụng cụ hỗ trợ y tế",
}


def localized_labels(labels: dict[str, float]) -> list[dict[str, Any]]:
    """Chuyển điểm số dự đoán thành list có cả tên tiếng Việt và mức độ cảnh báo."""
    return [
        {"name_en": name, "name_vi": LABELS_VI.get(name, name), "score": score,
         "level": "Cao" if score >= .65 else "Trung bình" if score >= .4 else "Thấp"}
        for name, score in sorted(labels.items(), key=lambda item: item[1], reverse=True)
    ]


try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # Nếu chưa cài gói AI, demo vẫn chạy nhưng không có model prediction.
    torch = None  # type: ignore[assignment]


if torch is not None:
    class PatchViT(nn.Module):
        """Encoder hình ảnh dựa trên Vision Transformer với patch 32x32."""

        def __init__(self, image_size: int = 448, patch_size: int = 32,
                     dim: int = 256, depth: int = 3, heads: int = 8):
            super().__init__()
            grid = image_size // patch_size
            self.patch = nn.Conv2d(1, dim, patch_size, stride=patch_size)
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos = nn.Parameter(torch.zeros(1, grid * grid + 1, dim))
            layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True,
                                               norm_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, depth)
            nn.init.trunc_normal_(self.pos, std=.02)
            nn.init.trunc_normal_(self.cls, std=.02)

        def forward(self, image):
            x = self.patch(image).flatten(2).transpose(1, 2)
            x = torch.cat((self.cls.expand(x.shape[0], -1, -1), x), dim=1)
            return self.encoder(x + self.pos)


    class MemoryLayerNorm(nn.Module):
        """Normalization có memory-conditioned scale/shift cho decoder."""

        def __init__(self, dim: int):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.memory = nn.Linear(dim, dim * 2)

        def forward(self, x, memory):
            scale, shift = self.memory(memory).chunk(2, dim=-1)
            return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


    class ReportDecoderLayer(nn.Module):
        """Một layer decoder tạo report text từ vector hình ảnh và memory."""

        def __init__(self, dim: int = 256, heads: int = 8):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
            self.n1, self.n2, self.n3 = (MemoryLayerNorm(dim) for _ in range(3))

        def forward(self, x, visual, memory, causal_mask):
            q = self.n1(x, memory)
            x = x + self.self_attn(q, q, q, attn_mask=causal_mask, need_weights=False)[0]
            q = self.n2(x, memory)
            x = x + self.cross_attn(q, visual, visual, need_weights=False)[0]
            return x + self.ff(self.n3(x, memory))


    class PureTransformerReportModel(nn.Module):
        """Baseline mô hình pure transformer phục vụ báo cáo X-quang."""

        def __init__(self, vocab_size: int, max_length: int = 256, dim: int = 256):
            super().__init__()
            self.visual = PatchViT(dim=dim)
            self.fuse = nn.Linear(dim * 2, dim)
            self.classifier = nn.Linear(dim, len(LABELS))
            self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
            self.text_pos = nn.Embedding(max_length, dim)
            self.decoder = nn.ModuleList(ReportDecoderLayer(dim) for _ in range(3))
            self.output = nn.Linear(dim, vocab_size)
            self.image_match = nn.Linear(dim, dim)
            self.text_match = nn.Linear(dim, dim)
            self.logit_scale = nn.Parameter(torch.tensor(2.659))

        def forward(self, frontal, lateral, tokens):
            visual, memory, logits = self.encode(frontal, lateral)
            token_logits, text_embed = self.decode(tokens, visual, memory)
            return {"token_logits": token_logits, "label_logits": logits,
                    "image_embed": F.normalize(self.image_match(memory), dim=-1),
                    "text_embed": F.normalize(self.text_match(text_embed), dim=-1),
                    "logit_scale": self.logit_scale.clamp(max=4.6052)}

        def encode(self, frontal, lateral):
            vf, vl = self.visual(frontal), self.visual(lateral)
            visual = torch.cat((vf, vl), dim=1)
            memory = self.fuse(torch.cat((vf[:, 0], vl[:, 0]), dim=-1))
            logits = self.classifier(memory)
            return visual, memory, logits

        def decode(self, tokens, visual, memory):
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            x = self.embedding(tokens) + self.text_pos(positions)[None]
            mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device,
                                         dtype=torch.bool), diagonal=1)
            for layer in self.decoder:
                x = layer(x, visual, memory, mask)
            text_pool = x[:, 0]
            return self.output(x), text_pool


    def multicriteria_loss(outputs, targets, token_targets, idf_weights, pad_id: int,
                           lambda_rg: float = 1., lambda_mlc: float = 2.,
                           lambda_itm: float = 5.):
        """Hàm loss kết hợp: report CE + 14-label BCE + image-text matching."""
        logits = outputs["token_logits"].reshape(-1, outputs["token_logits"].shape[-1])
        gold = token_targets[:, 1:].reshape(-1)
        per_token = F.cross_entropy(logits, gold, ignore_index=pad_id, reduction="none")
        weights = idf_weights[gold].to(per_token.dtype)
        valid = gold.ne(pad_id)
        twrg = (per_token[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1)
        mlc = F.binary_cross_entropy_with_logits(outputs["label_logits"], targets.float())
        image, text = outputs["image_embed"], outputs["text_embed"]
        scale = outputs.get("logit_scale", torch.tensor(1., device=image.device)).exp().clamp(max=100)
        scores = scale * image @ text.T
        truth = torch.arange(scores.shape[0], device=scores.device)
        itm = (F.cross_entropy(scores, truth) + F.cross_entropy(scores.T, truth)) / 2
        total = lambda_rg * twrg + lambda_mlc * mlc + lambda_itm * itm
        return total, {"total": total.detach(), "twRG": twrg.detach(),
                       "MLC": mlc.detach(), "ITM": itm.detach()}


_inference_mode = torch.inference_mode if torch is not None else (lambda fn: fn)


class ModelUnavailable(RuntimeError):
    """Lỗi để báo rằng model chưa sẵn sàng do thiếu checkpoint hoặc dependency."""


class CheckpointInference:
    """Tải checkpoint của mô hình và thực hiện inference an toàn."""

    def __init__(self, checkpoint: str | Path | None = None):
        if torch is None:
            raise ModelUnavailable("Cài dependencies trong requirements-ai.txt để bật AI.")
        self.device = torch.device(os.getenv("MEDAI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        default_checkpoint = Path(__file__).resolve().parent.parent / "data" / "checkpoints" / "medai.pt"
        self.path = Path(checkpoint or os.getenv("MEDAI_CHECKPOINT", str(default_checkpoint)))
        if not self.path.is_file():
            raise ModelUnavailable(f"Chưa có checkpoint đã huấn luyện: {self.path}")
        state = torch.load(self.path, map_location=self.device, weights_only=False)
        self.vocab = state["vocab"]
        self.id_to_token = {int(i): token for token, i in self.vocab.items()}
        self.model = PureTransformerReportModel(len(self.vocab), state.get("max_length", 256))
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()
        self.max_length = state.get("max_length", 256)

    def _image(self, content: bytes):
        """Chuyển raw bytes ảnh sang tensor PyTorch có cùng kích thước mô hình."""
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(content)).convert("L").resize((448, 448))
        tensor = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(1, 1, 448, 448) / 255
        return (tensor - .5) / .5

    @_inference_mode
    def predict(self, frontal: bytes, lateral: bytes) -> dict[str, float]:
        """Dự đoán 14 nhãn bệnh từ 2 ảnh X-quang của bệnh nhân."""
        out = self.model(self._image(frontal).to(self.device), self._image(lateral).to(self.device),
                         torch.tensor([[self.vocab["<bos>"]]], device=self.device))
        scores = torch.sigmoid(out["label_logits"])[0].cpu().tolist()
        return dict(zip(LABELS, scores))

    @_inference_mode
    def generate(self, frontal: bytes, lateral: bytes) -> str:
        """Sinh report text dạng văn bản từ mô hình decoder."""
        bos, eos = self.vocab["<bos>"], self.vocab["<eos>"]
        visual, memory, _ = self.model.encode(self._image(frontal).to(self.device),
                                               self._image(lateral).to(self.device))
        tokens = torch.tensor([[bos]], device=self.device)
        for _ in range(self.max_length - 1):
            logits, _ = self.model.decode(tokens, visual, memory)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            tokens = torch.cat((tokens, nxt), dim=1)
            if nxt.item() == eos:
                break
        words = [self.id_to_token.get(i, "") for i in tokens[0].tolist()[1:]]
        return " ".join(w for w in words if w not in {"<eos>", "<pad>"})

    def draft_report(self, labels, frontal: bytes, lateral: bytes):
        """Tạo findings/impression từ kết quả mô hình."""
        generated = self.generate(frontal, lateral)
        if "<impression>" in generated:
            findings, impression = generated.split("<impression>", 1)
        else:
            findings, impression = generated, ""
        return findings.strip(), impression.strip()
