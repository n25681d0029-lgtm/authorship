"""Hai văn bản có cùng một tác giả không? — so phân bố xác suất tác giả của hai văn bản.

độ giống = hệ số Bhattacharyya Σ √(p_A · p_B) trên các tác giả ứng viên (1: hai văn bản được máy "nhìn" như nhau).
Con số này chưa phải xác suất cùng tác giả, nên được đối chiếu với các cặp đoạn trong tập kiểm tra (đã biết đáp án):
pair_calibration() chia độ giống thành các khoảng và đếm tỷ lệ cặp cùng tác giả ở mỗi khoảng.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

BINS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])


def prob_vectors(pred, texts: Sequence[str], candidates: Optional[Sequence[str]] = None) -> np.ndarray:
    res = pred.predict(list(texts), top_k=len(pred.names), explain=False, candidates=candidates)
    idx = {a: i for i, a in enumerate(pred.names)}
    out = np.zeros((len(res), len(pred.names)))
    for r_i, r in enumerate(res):
        for t in r["top_k"]:
            out[r_i, idx[t["author"]]] = t["prob"]
    return out


def similarity(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None)).sum())


def pair_calibration(probs: np.ndarray, authors: Sequence[str], max_pairs: int = 60000, seed: int = 0) -> list[dict]:
    """probs: xác suất của các đoạn test (đã lọc theo cùng nhóm ứng viên). Trả về [{lo, hi, n, ty_le_cung}] theo khoảng."""
    rng = np.random.default_rng(seed)
    authors = np.asarray(authors)
    n = len(authors)
    i = rng.integers(0, n, max_pairs)
    j = rng.integers(0, n, max_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    # thêm đủ cặp cùng tác giả (ngẫu nhiên thì hiếm) để mọi khoảng có số liệu
    same_i, same_j = [], []
    for a in np.unique(authors):
        ids = np.flatnonzero(authors == a)
        if len(ids) >= 2:
            k = min(len(ids) * 4, 400)
            same_i += list(rng.choice(ids, k)); same_j += list(rng.choice(ids, k))
    i = np.concatenate([i, same_i]).astype(int)
    j = np.concatenate([j, same_j]).astype(int)
    keep = i != j
    i, j = i[keep], j[keep]
    s = np.sqrt(np.clip(probs[i], 0, None) * np.clip(probs[j], 0, None)).sum(1)
    same = authors[i] == authors[j]
    # trọng số đưa số cặp cùng / khác tác giả về 50:50 — tức là giả định trước khi so, khả năng cùng tác giả là
    # một nửa (người dùng so hai văn bản thường vì nghi ngờ chúng cùng tác giả); tỷ lệ ra là "nếu ban đầu 50/50"
    w = np.ones(len(i))
    base_rate = 0.5
    obs_rate = same.mean()
    w[same] *= base_rate / max(obs_rate, 1e-9)
    w[~same] *= (1 - base_rate) / max(1 - obs_rate, 1e-9)
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (s >= lo) & (s < hi)
        if m.sum() >= 20:
            out.append({"lo": float(lo), "hi": float(min(hi, 1.0)), "n": int(m.sum()),
                        "ty_le_cung": float((w[m] * same[m]).sum() / w[m].sum())})
    return out


def lookup(calib: list[dict], s: float) -> Optional[dict]:
    for b in calib:
        if b["lo"] <= s < b["hi"] or (s >= 1.0 and b["hi"] >= 1.0):
            return b
    return None
