"""Tính IDF trên tập train và TF-IDF riêng cho từng vị trí của báo cáo."""
from collections import Counter
import math

import torch


def compute_idf_weights(tokenizer, train_reports):
    """Tính IDF_s = log(K / df_s); các token đặc biệt có IDF bằng 0."""
    report_count = len(train_reports)
    if report_count == 0:
        raise ValueError("Cần ít nhất một báo cáo train để tính IDF.")

    vocab_size = len(tokenizer)
    unk_id = tokenizer.vocab["<unk>"]
    document_frequency = [0] * vocab_size
    for report in train_reports:
        token_ids = {
            tokenizer.vocab.get(token, unk_id)
            for token in tokenizer.tokenize(report)
        }
        for token_id in token_ids:
            document_frequency[token_id] += 1

    idf = torch.zeros(vocab_size, dtype=torch.float32)
    for token_id, frequency in enumerate(document_frequency):
        if frequency > 0:
            idf[token_id] = math.log(report_count / frequency)

    # PAD/BOS/EOS/UNK không phải term y khoa có IDF trong phương trình (6).
    for special in ("<pad>", "<bos>", "<eos>", "<unk>"):
        idf[tokenizer.vocab[special]] = 0.0
    return idf


def tfidf_weights(token_ids, idf, pad_id, bos_id, eos_id, unk_id):
    """Trả trọng số cho từng token theo TF_sk * IDF_s.

    TF được tính trên các term của báo cáo, bao gồm các vị trí được ánh xạ
    sang UNK trong mẫu số. PAD/UNK có trọng số 0; BOS/EOS có trọng số 1.
    """
    lexical_ids = [
        int(token_id)
        for token_id in token_ids
        if int(token_id) not in (pad_id, bos_id, eos_id)
    ]
    counts = Counter(lexical_ids)
    lexical_count = len(lexical_ids)

    weights = []
    for raw_token_id in token_ids:
        token_id = int(raw_token_id)
        if token_id == pad_id or token_id == unk_id:
            weight = 0.0
        elif token_id == bos_id or token_id == eos_id:
            weight = 1.0
        elif lexical_count == 0:
            weight = 0.0
        else:
            weight = (counts[token_id] / lexical_count) * float(idf[token_id])
        weights.append(weight)
    return weights


def encode_report_with_tfidf(report, tokenizer, idf, max_len):
    """Mã hoá báo cáo và tính trọng số trước khi cắt về max_len token."""
    if max_len < 2:
        raise ValueError("max_len phải đủ chỗ cho BOS và EOS.")

    unk_id = tokenizer.vocab["<unk>"]
    lexical_ids = [
        tokenizer.vocab.get(token, unk_id)
        for token in tokenizer.tokenize(report)
    ]
    lexical_weights = tfidf_weights(
        lexical_ids,
        idf,
        pad_id=tokenizer.vocab["<pad>"],
        bos_id=tokenizer.vocab["<bos>"],
        eos_id=tokenizer.vocab["<eos>"],
        unk_id=unk_id,
    )

    kept_ids = lexical_ids[:max_len - 2]
    kept_weights = lexical_weights[:max_len - 2]
    report_ids = [tokenizer.vocab["<bos>"]] + kept_ids + [tokenizer.vocab["<eos>"]]
    report_weights = [1.0] + kept_weights + [1.0]
    return report_ids, report_weights
