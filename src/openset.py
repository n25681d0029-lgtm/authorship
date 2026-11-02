"""Đánh giá chức năng từ chối "unknown" (tập mở đơn giản).

Tập "đã biết": các đoạn test của chính checkpoint (cùng cách chia lúc huấn luyện).
Tập "lạ": văn bản của tác giả KHÔNG có trong kho (ví dụ Lan Khai, bị loại vì thiếu dữ liệu), cắt thành đoạn
cùng cỡ dữ liệu huấn luyện.

Chỉ số:
  AUROC            : xác suất một đoạn đã biết có độ tin cậy cao hơn một đoạn lạ (0,5 = không phân biệt được).
  accept_known     : tỷ lệ đoạn đã biết được chấp nhận ở ngưỡng t.
  acc_accepted     : accuracy trên các đoạn đã biết được chấp nhận.
  reject_unknown   : tỷ lệ đoạn lạ bị trả về "unknown" ở ngưỡng t.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from config import TrainConfig
from .calibration import softmax
from .corpus_builder import pack_passages, split_paragraphs
from .data_loader import load_corpus, make_splits, read_text_file
from .predict import AuthorPredictor, resolve_checkpoint
from .utils import load_json, save_json

DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def unknown_passages(src: Path) -> list[str]:
    """Đọc mọi file .txt trong thư mục (hoặc một file) và cắt thành đoạn 50-200 âm tiết."""
    src = Path(src)
    files = [src] if src.is_file() else sorted(src.rglob("*.txt"))
    out = []
    for f in files:
        out += pack_passages(split_paragraphs(read_text_file(f)), 50, 200, 150)
    if not out:
        raise ValueError(f"Không có đoạn văn nào đủ 50 âm tiết trong {src}")
    return out


def _table(conf_known, correct, conf_unknown, thresholds) -> list[dict]:
    rows = []
    for t in thresholds:
        acc = conf_known >= t
        rows.append({"threshold": float(t), "accept_known": float(acc.mean()),
                     "acc_accepted": float(correct[acc].mean()) if acc.any() else float("nan"),
                     "reject_unknown": float((conf_unknown < t).mean())})
    return rows


def evaluate_open_set(ckpt: str | Path, unknown_src: Path, device: str = "auto",
                      thresholds: Sequence[float] = DEFAULT_THRESHOLDS, data_path: Optional[Path] = None) -> dict:
    ckpt = resolve_checkpoint(ckpt)
    meta = load_json(ckpt / "meta.json")
    info, cfg = meta["data"], TrainConfig.from_dict(meta["config"])
    df = load_corpus(Path(data_path) if data_path else Path(info["path"]), fmt=info["fmt"],
                     text_col=info["text_col"], author_col=info["author_col"])
    df = make_splits(df, cfg.val_size, cfg.test_size, cfg.seed, info.get("test_topics"), cfg.topic_holdout or None)
    test = df[(df["split"] == "test") & df["author"].isin(meta["label_map"])]
    unk = unknown_passages(unknown_src)

    pred = AuthorPredictor.from_checkpoint(ckpt, device)
    y = test["author"].map(meta["label_map"]).to_numpy()
    z_known, _ = pred.predict_logits(test["text"].tolist())
    z_unknown, _ = pred.predict_logits(unk)
    stored = pred.reject_threshold
    result = {"checkpoint": str(ckpt), "unknown_source": str(unknown_src), "n_known": int(len(y)),
              "n_unknown": len(unk), "temperature": pred.temperature, "stored_threshold": stored}
    for name, t_scale in (("uncalibrated", 1.0), ("calibrated", pred.temperature)):
        pk, pu = softmax(z_known, t_scale), softmax(z_unknown, t_scale)
        ck, cu = pk.max(axis=1), pu.max(axis=1)
        correct = pk.argmax(axis=1) == y
        ths = sorted(set(thresholds) | ({stored} if stored is not None and name == "calibrated" else set()))
        result[name] = {
            "auroc": float(roc_auc_score(np.r_[np.ones(len(ck)), np.zeros(len(cu))], np.r_[ck, cu])),
            "mean_conf_known": float(ck.mean()), "mean_conf_unknown": float(cu.mean()),
            "unknown_assigned_to": dict(Counter(pred.names[i] for i in pu.argmax(axis=1)).most_common()),
            "by_threshold": _table(ck, correct, cu, ths),
        }
    save_json(result, ckpt / "reports" / "openset.json")
    return result
