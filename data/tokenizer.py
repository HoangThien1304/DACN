"""
tokenizer.py — Tokenizer đơn giản cho báo cáo y tế (tiếng Anh).
Dùng word-level regex tokenizer, tự build vocab từ tập train.
Các token đặc biệt: <pad>=0, <bos>=1, <eos>=2, <unk>=3.
"""
import re
import json
from collections import Counter

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL = [PAD, BOS, EOS, UNK]


class SimpleTokenizer:
    def __init__(self, vocab: dict = None):
        self.vocab = vocab or {}
        self.itos = {i: s for s, i in self.vocab.items()}

    @staticmethod
    def tokenize(text: str):
        # Giữ chữ+số và dạng từ như "no'", "patient's"
        return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", str(text).lower())

    def build_vocab(self, reports, min_freq: int = 1):
        counter = Counter()
        for r in reports:
            counter.update(self.tokenize(r))
        self.vocab = {s: i for i, s in enumerate(SPECIAL)}
        for w, c in counter.most_common():
            if c >= min_freq:
                self.vocab[w] = len(self.vocab)
        self.itos = {i: s for s, i in self.vocab.items()}
        return self.vocab

    def encode(self, text: str, max_len: int):
        """Trả về list id đã thêm <bos> ở đầu, <eos> ở cuối, độ dài <= max_len."""
        tokens = self.tokenize(text)[: max_len - 2]
        ids = [self.vocab.get(t, self.vocab[UNK]) for t in tokens]
        return [self.vocab[BOS]] + ids + [self.vocab[EOS]]

    def decode(self, ids):
        words = [self.itos.get(int(i), UNK) for i in ids]
        # Cắt từ <eos>
        if EOS in words:
            words = words[: words.index(EOS)]
        return " ".join(w for w in words if w not in (PAD, BOS))

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.vocab, f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            return cls(json.load(f))

    def __len__(self):
        return len(self.vocab)
