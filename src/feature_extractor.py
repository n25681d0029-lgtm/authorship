"""Trích xuất đặc trưng phong cách (stylometric features).

Các nhóm đặc trưng:
  - Từ vựng : TTR, tỷ lệ hapax, Yule's K, độ dài từ (trung bình, độ lệch chuẩn).
  - Cú pháp : độ dài câu (trung bình, độ lệch chuẩn).
  - Ký tự   : tần suất từng dấu câu, tỷ lệ chữ hoa, tỷ lệ chữ số.
  - Từ chức năng: tần suất các từ phổ biến nhất của KHO HUẤN LUYỆN (không cần danh sách
    từ chức năng riêng cho từng ngôn ngữ, nên dùng được cho cả tiếng Anh lẫn tiếng Việt).

Quy trình: fit() trên tập train để chọn từ chức năng và tính trung bình, độ lệch chuẩn;
transform() trả về ma trận đã chuẩn hóa z-score. Tập val/test chỉ dùng transform()
để không rò rỉ thông tin.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from .text_utils import normalize, split_sentences, tokenize_words
from .utils import load_json, save_json

PUNCT_CHARS = [",", ".", ";", ":", "!", "?", '"', "'", "(", ")", "-", "–", "—", "…", "“", "”", "’"]
BASE_FEATURES = (
    ["ttr", "hapax_ratio", "yule_k", "word_len_mean", "word_len_std",
     "sent_len_mean", "sent_len_std", "upper_ratio", "digit_ratio"]
    + [f"punct[{c}]" for c in PUNCT_CHARS]
)
TTR_WINDOW = 500   # TTR tính trên tối đa 500 từ đầu để bớt phụ thuộc độ dài văn bản
Z_CLIP = 5.0       # cắt giá trị z-score ngoài [-5, 5] để một giá trị cực đoan không át cả vector


def _std(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / n)


class StyleFeatureExtractor:
    def __init__(self, num_function_words: int = 150) -> None:
        self.num_function_words = num_function_words
        self.function_words: list[str] = []
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    # ------------------------------------------------------------------ tên đặc trưng
    @property
    def feature_names(self) -> list[str]:
        return BASE_FEATURES + [f"fw[{w}]" for w in self.function_words]

    @property
    def dim(self) -> int:
        return len(BASE_FEATURES) + len(self.function_words)

    # ------------------------------------------------------------------ đặc trưng thô
    def _raw(self, text: str) -> np.ndarray:
        text = normalize(text)
        tokens = [t.lower() for t in tokenize_words(text)]
        n = len(tokens)
        feats: list[float] = []

        if n == 0:
            feats += [0.0] * 5
        else:
            window = tokens[:TTR_WINDOW]
            ttr = len(set(window)) / len(window)
            counts = Counter(tokens)
            freq_of_freq = Counter(counts.values())          # V_i: số từ xuất hiện đúng i lần
            hapax = freq_of_freq.get(1, 0) / n
            yule_k = 1e4 * (sum(i * i * v for i, v in freq_of_freq.items()) - n) / (n * n)
            lens = [len(t) for t in tokens]
            feats += [ttr, hapax, yule_k, sum(lens) / n, _std(lens)]

        sents = split_sentences(text)
        sent_lens = [len(tokenize_words(s)) for s in sents] or [0]
        feats += [sum(sent_lens) / len(sent_lens), _std(sent_lens)]

        n_chars = max(1, len(text))
        feats += [sum(c.isupper() for c in text) / n_chars, sum(c.isdigit() for c in text) / n_chars]
        feats += [1000.0 * text.count(c) / n_chars for c in PUNCT_CHARS]

        if self.function_words:
            counts = Counter(tokens)
            denom = max(1, n)
            feats += [100.0 * counts.get(w, 0) / denom for w in self.function_words]
        return np.asarray(feats, dtype=np.float64)

    # ------------------------------------------------------------------ fit / transform
    def fit(self, texts: Sequence[str]) -> "StyleFeatureExtractor":
        total = Counter()
        for t in texts:
            total.update(tok.lower() for tok in tokenize_words(normalize(t)))
        self.function_words = [w for w, _ in total.most_common(self.num_function_words)]
        raw = np.stack([self._raw(t) for t in texts])
        self.mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        std[std < 1e-8] = 1.0            # đặc trưng hằng số: tránh chia cho 0
        self.std = std
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("Gọi fit() trước khi transform().")
        raw = np.stack([self._raw(t) for t in texts])
        z = (raw - self.mean) / self.std
        return np.clip(z, -Z_CLIP, Z_CLIP).astype(np.float32)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    # ------------------------------------------------------------------ giải thích
    def explain(self, text: str, top: int = 8) -> list[dict]:
        """Các đặc trưng lệch nhiều nhất so với mức trung bình của kho huấn luyện.
        Dùng để trình bày vì sao mô hình nghiêng về một tác giả (tính giải thích được)."""
        raw = self._raw(text)
        z = np.clip((raw - self.mean) / self.std, -Z_CLIP, Z_CLIP)
        order = np.argsort(-np.abs(z))[:top]
        names = self.feature_names
        return [{"feature": names[i], "value": float(raw[i]), "z": float(z[i])} for i in order]

    # ------------------------------------------------------------------ lưu / nạp
    def state(self) -> dict:
        return {"num_function_words": self.num_function_words, "function_words": self.function_words,
                "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, state: dict) -> "StyleFeatureExtractor":
        obj = cls(state["num_function_words"])
        obj.function_words = list(state["function_words"])
        obj.mean = np.asarray(state["mean"], dtype=np.float64)
        obj.std = np.asarray(state["std"], dtype=np.float64)
        return obj

    def save(self, path: Path) -> None:
        save_json(self.state(), path)

    @classmethod
    def load(cls, path: Path) -> "StyleFeatureExtractor":
        return cls.from_state(load_json(path))
