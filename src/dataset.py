"""Dataset và bộ gom batch cho PyTorch.

Dataset chỉ giữ văn bản thô, ma trận đặc trưng phong cách đã tính sẵn và nhãn.
Việc token hóa (chậm) được làm theo từng batch trong Collator để tiết kiệm bộ nhớ và
để dùng được padding động (batch ngắn thì đệm ngắn).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .text_utils import prepare_for_lm


class AuthorshipDataset(Dataset):
    def __init__(self, texts: Sequence[str], style: np.ndarray, labels: Sequence[int]) -> None:
        if not (len(texts) == len(style) == len(labels)):
            raise ValueError("texts, style và labels phải cùng độ dài.")
        self.texts = list(texts)
        self.style = torch.as_tensor(style, dtype=torch.float32)
        self.labels = torch.as_tensor(list(labels), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict:
        return {"text": self.texts[i], "style": self.style[i], "label": self.labels[i]}


class Collator:
    """Gom một danh sách mẫu thành batch tensor.

    tokenizer=None dùng cho chế độ style_only (không có mô hình ngôn ngữ).
    """

    def __init__(self, tokenizer, max_length: int, lang: str) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lang = lang

    def __call__(self, items: list[dict]) -> dict:
        batch = {
            "style": torch.stack([it["style"] for it in items]),
            "labels": torch.stack([it["label"] for it in items]),
        }
        if self.tokenizer is not None:
            texts = [prepare_for_lm(it["text"], self.lang) for it in items]
            enc = self.tokenizer(texts, truncation=True, max_length=self.max_length,
                                 padding=True, return_tensors="pt")
            batch["input_ids"] = enc["input_ids"]
            batch["attention_mask"] = enc["attention_mask"]
            # độ dài tương đối (0..1), cổng của gated fusion có thể dùng làm tín hiệu
            batch["length"] = enc["attention_mask"].sum(dim=1).float() / self.max_length
        else:
            batch["length"] = torch.ones(len(items))
        return batch


def make_loader(texts, style, labels, collator: Collator, batch_size: int, shuffle: bool,
                num_workers: int = 0, balanced: bool = False, seed: int = 0) -> DataLoader:
    """balanced=True: rút mẫu có hoàn lại với xác suất tỷ lệ nghịch số mẫu của lớp (oversampling);
    mỗi epoch vẫn gồm len(texts) mẫu nhưng các tác giả xuất hiện với tần suất xấp xỉ bằng nhau."""
    ds = AuthorshipDataset(texts, style, labels)
    if balanced:
        from torch.utils.data import WeightedRandomSampler

        y = np.asarray(list(labels))
        weights = torch.as_tensor(1.0 / np.bincount(y)[y], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(y), replacement=True,
                                        generator=torch.Generator().manual_seed(seed))
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, collate_fn=collator, num_workers=num_workers)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collator,
                      num_workers=num_workers)
