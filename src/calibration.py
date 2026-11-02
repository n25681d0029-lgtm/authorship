"""Hiệu chỉnh xác suất (temperature scaling) và chọn ngưỡng từ chối "unknown".

Mạng neural thường quá tự tin: xác suất cao nhất 0,9 nhưng chỉ đúng khoảng 70% số lần. Temperature scaling
(Guo và cộng sự, 2017) chia logits cho một hệ số T > 0 học trên tập val sao cho log-loss nhỏ nhất; T không đổi
nhãn dự đoán (argmax) nên không ảnh hưởng accuracy, chỉ làm xác suất phản ánh đúng độ tin cậy.

Ngưỡng từ chối được chọn trên tập val (chỉ có tác giả đã biết) sao cho tỷ lệ accept_rate số đoạn được chấp nhận;
khi gặp văn bản của người lạ, xác suất cao nhất thường thấp hơn và bị trả về "unknown".
"""
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> float:
    p = softmax(logits, temperature)[np.arange(len(y)), y]
    return float(-np.log(np.clip(p, 1e-12, None)).mean())


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error: chênh lệch trung bình giữa độ tin cậy và tỷ lệ đúng theo từng khoảng tin cậy."""
    conf, correct = probs.max(axis=1), probs.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    grid = np.exp(np.linspace(np.log(0.25), np.log(20.0), 400))
    return float(grid[int(np.argmin([nll(logits, y, t) for t in grid]))])


def calibrate(logits: np.ndarray, y: np.ndarray, accept_rate: float = 0.95) -> dict:
    t = fit_temperature(logits, y)
    before, after = softmax(logits), softmax(logits, t)
    return {
        "temperature": t,
        "reject_threshold": float(np.quantile(after.max(axis=1), 1.0 - accept_rate)),
        "accept_rate_val": accept_rate,
        "val_nll_before": nll(logits, y), "val_nll_after": nll(logits, y, t),
        "val_ece_before": ece(before, y), "val_ece_after": ece(after, y),
        "n_val": int(len(y)),
    }


def fit_class_bias(logits: np.ndarray, y: np.ndarray, n_passes: int = 3,
                   grid: np.ndarray | None = None) -> np.ndarray:
    """Độ ưu tiên b_c cộng vào logit của từng tác giả để macro-F1 trên tập val lớn nhất.

    Mô hình thường thiên về tác giả nhiều dữ liệu; cộng b_c > 0 làm tác giả c được đoán nhiều hơn (tăng recall,
    giảm precision). Tìm lần lượt từng tác giả trên một lưới giá trị (coordinate ascent), lặp n_passes vòng;
    chỉ nhận thay đổi khi macro-F1 tăng thật sự, hòa thì giữ giá trị gần 0 hơn để tránh khớp quá mức tập val.
    Chỉ tác giả có mặt trong y mới được chỉnh.
    """
    from sklearn.metrics import f1_score

    grid = np.arange(-3.0, 3.01, 0.25) if grid is None else grid
    z = np.asarray(logits, dtype=np.float64)
    bias = np.zeros(z.shape[1])
    present = sorted(set(np.asarray(y).tolist()))

    def score(b):
        return f1_score(y, (z + b).argmax(axis=1), labels=present, average="macro", zero_division=0)

    best = score(bias)
    for _ in range(n_passes):
        changed = False
        for c in present:
            for v in sorted(grid, key=abs):
                cand = bias.copy()
                cand[c] = v
                s = score(cand)
                if s > best + 1e-6:
                    best, bias, changed = s, cand, True
        if not changed:
            break
    return bias
