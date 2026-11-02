"""Thước đo đánh giá, ma trận nhầm lẫn và hàm chạy mô hình trên một DataLoader."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .utils import save_json


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], label_names: Sequence[str]) -> dict:
    """Accuracy, Macro-F1 và chi tiết theo từng tác giả.

    Macro-F1 lấy trung bình trên các tác giả có mặt trong nhãn thật hoặc nhãn đoán (như mặc định của
    scikit-learn): tác giả vắng hẳn khỏi tập đánh giá không bị tính là F1 = 0."""
    labels = list(range(len(label_names)))
    present = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=present, average="weighted", zero_division=0)),
        "n": int(len(y_true)),
        "per_class": classification_report(y_true, y_pred, labels=labels, target_names=list(label_names),
                                           output_dict=True, zero_division=0),
    }


def top_confusions(cm: np.ndarray, label_names: Sequence[str], k: int = 5) -> list[dict]:
    """Các cặp (thật -> đoán) bị nhầm nhiều nhất, để biết hai tác giả nào dễ lẫn."""
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": label_names[i], "predicted": label_names[j], "count": int(cm[i, j])})
    return sorted(pairs, key=lambda p: -p["count"])[:k]


def save_evaluation(y_true, y_pred, label_names: Sequence[str], out_dir: Path, prefix: str) -> dict:
    """Lưu metrics (JSON), báo cáo văn bản, ma trận nhầm lẫn (CSV, và PNG nếu có matplotlib)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(y_true, y_pred, label_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    metrics["top_confusions"] = top_confusions(cm, label_names)
    save_json(metrics, out_dir / f"{prefix}_metrics.json")
    pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(out_dir / f"{prefix}_confusion.csv",
                                                                    encoding="utf-8")
    (out_dir / f"{prefix}_report.txt").write_text(
        classification_report(y_true, y_pred, labels=list(range(len(label_names))),
                              target_names=list(label_names), zero_division=0), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        size = min(14, max(4, 0.4 * len(label_names) + 3))
        fig, ax = plt.subplots(figsize=(size, size))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(label_names)))
        ax.set_yticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=90, fontsize=7)
        ax.set_yticklabels(label_names, fontsize=7)
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        ax.set_title(f"Ma trận nhầm lẫn ({prefix})")
        if len(label_names) <= 15:
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_confusion.png", dpi=130)
        plt.close(fig)
    except ImportError:
        pass
    return metrics


def collect_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Chạy mô hình ở chế độ suy luận. Trả về (nhãn thật, logits chưa qua softmax)."""
    import torch

    model.eval()
    ys, zs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(style=batch["style"], input_ids=batch.get("input_ids"),
                           attention_mask=batch.get("attention_mask"), length=batch.get("length"))
            zs.append(logits.float().cpu().numpy())
            ys.append(batch["labels"].cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.zeros(0, dtype=int)
    return y_true, (np.concatenate(zs) if zs else np.zeros((0, 0)))


def run_model(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chạy mô hình ở chế độ suy luận. Trả về (nhãn thật, nhãn đoán, xác suất chưa hiệu chỉnh)."""
    from .calibration import softmax

    y_true, logits = collect_logits(model, loader, device)
    if not len(logits):
        return y_true, y_true, logits
    probs = softmax(logits)
    return y_true, probs.argmax(axis=1), probs
