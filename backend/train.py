"""Train the project baseline from a paired-image/report CSV.

CSV columns: frontal,lateral,labels,findings,impression. Paths are relative to
--data-root. labels is JSON mapping the 14 English label names to 0/1 values.
The CSV and images must be obtained and licensed by the user.
"""
import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from .ai_model import LABELS, PureTransformerReportModel, multicriteria_loss

SPECIAL = ["<pad>", "<bos>", "<eos>", "<impression>", "<unk>"]


def tokenize(text):
    return text.lower().split()


class XrayReports(Dataset):
    def __init__(self, csv_path, root, vocab, max_length):
        self.rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig", newline="")))
        self.root, self.vocab, self.max_length = root, vocab, max_length

    def image(self, name):
        image = Image.open(self.root / name).convert("L").resize((448, 448))
        return (torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(1, 448, 448) / 127.5) - 1

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        words = tokenize(row["findings"]) + ["<impression>"] + tokenize(row["impression"])
        ids = [self.vocab["<bos>"]] + [self.vocab.get(w, self.vocab["<unk>"]) for w in words]
        ids = ids[:self.max_length - 1] + [self.vocab["<eos>"]]
        ids += [self.vocab["<pad>"]] * (self.max_length - len(ids))
        raw_labels = json.loads(row["labels"])
        labels = torch.tensor([float(raw_labels.get(name, 0)) for name in LABELS])
        return self.image(row["frontal"]), self.image(row["lateral"]), labels, torch.tensor(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data" / "checkpoints" / "medai.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8-sig", newline="")))
    counter = Counter()
    for row in rows:
        counter.update(tokenize(row["findings"]))
        counter.update(tokenize(row["impression"]))
        counter["<impression>"] += 1
    vocab = {token: idx for idx, token in enumerate(SPECIAL + sorted(w for w in counter if w not in SPECIAL))}
    dataset = XrayReports(args.csv, args.data_root, vocab, args.max_length)
    if len(dataset) < 2:
        raise SystemExit("Need at least two paired studies for train/validation split.")
    n_val = max(1, int(len(dataset) * .1))
    train_set, val_set = random_split(dataset, [len(dataset)-n_val, n_val],
                                      generator=torch.Generator().manual_seed(args.seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    model = PureTransformerReportModel(len(vocab), args.max_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    # Inverse document frequency for term-weighted report generation.
    doc_freq = Counter()
    for row in rows:
        doc_freq.update(set(tokenize(row["findings"]) + ["<impression>"] + tokenize(row["impression"])))
    idf = torch.ones(len(vocab), device=device)
    for term, idx in vocab.items():
        idf[idx] = math.log(1 + len(rows) / (1 + doc_freq[term])) if term in doc_freq else 1.
    best = float("inf")
    for epoch in range(args.epochs):
        model.train(); train_loss = 0.
        for frontal, lateral, labels, tokens in train_loader:
            frontal, lateral, labels, tokens = (x.to(device) for x in (frontal, lateral, labels, tokens))
            out = model(frontal, lateral, tokens[:, :-1])
            loss, _ = multicriteria_loss(out, labels, tokens, idf, vocab["<pad>"])
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_loss += loss.item()
        model.eval(); val_loss = 0.
        with torch.no_grad():
            for frontal, lateral, labels, tokens in val_loader:
                frontal, lateral, labels, tokens = (x.to(device) for x in (frontal, lateral, labels, tokens))
                out = model(frontal, lateral, tokens[:, :-1])
                loss, _ = multicriteria_loss(out, labels, tokens, idf, vocab["<pad>"])
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))
        print(f"epoch {epoch+1}/{args.epochs}: train={train_loss/max(1,len(train_loader)):.4f} val={val_loss:.4f}")
        if val_loss < best:
            best = val_loss; args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "vocab": vocab,
                        "max_length": args.max_length, "labels": LABELS,
                        "config": {"image_size": 448, "patch_size": 32,
                                   "dim": 256, "encoder_layers": 3,
                                   "decoder_layers": 3, "heads": 8}}, args.output)
            print(f"saved {args.output}")


if __name__ == "__main__":
    main()
