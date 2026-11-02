#!/usr/bin/env python3
"""So sánh các checkpoint trên cùng tập test, tập trung vào tác giả ít dữ liệu, và mô phỏng bộ lọc thể loại.

Chạy: python scripts/compare_low_resource.py --ckpts best_model v4_cw_bias v4_os_bias --out outputs/low_resource_v4.md
Mỗi checkpoint dùng --class-bias được báo cáo hai dòng: trước (test_nobias) và sau khi cộng độ ưu tiên.
Macro-F1 tính như src/evaluate.py: trung bình trên tác giả có mặt trong nhãn thật hoặc nhãn đoán.
Bộ lọc thể loại được mô phỏng lạc quan: người dùng chọn đúng (các) thể loại của tác giả thật theo data/author_groups.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHECKPOINT_DIR, TrainConfig  # noqa: E402
from src.data_loader import load_corpus, make_splits  # noqa: E402
from src.utils import load_json  # noqa: E402

LOW = 160


def md(df: pd.DataFrame, index: bool = False) -> str:
    """Bảng markdown (không cần thư viện tabulate)."""
    d = df.reset_index() if index else df
    fmt = lambda v: f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)  # noqa: E731
    out = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    out += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from src.predict import AuthorPredictor

    meta0 = load_json(CHECKPOINT_DIR / args.ckpts[0] / "meta.json")
    cfg = TrainConfig.from_dict(meta0["config"])
    df = make_splits(load_corpus(meta0["data"]["path"]), cfg.val_size, cfg.test_size, cfg.seed)
    test = df[df["split"] == "test"]
    counts = df["author"].value_counts()
    groups = json.loads((Path(__file__).resolve().parent.parent / "data" / "author_groups.json").read_text(encoding="utf-8"))
    low = sorted(a for a in counts.index if counts[a] < LOW and a in set(test["author"]))

    rows, per_author = [], {}
    for ck in args.ckpts:
        pred = AuthorPredictor.from_checkpoint(ck)
        names = pred.names
        y = test["author"].map({a: i for i, a in enumerate(names)}).to_numpy()
        z, _ = pred.predict_logits(test["text"].tolist())          # đã cộng độ ưu tiên (nếu có)
        variants = {ck: z}
        if np.any(pred.class_bias):
            variants[f"{ck} (chưa cộng độ ưu tiên)"] = z - pred.class_bias
        for name, logits in variants.items():
            yp = logits.argmax(axis=1)
            # bộ lọc thể loại: chỉ giữ tác giả có chung thể loại với tác giả thật
            allowed = np.array([[bool(set(groups["tac_gia"].get(b, [])) & set(groups["tac_gia"].get(a, [])))
                                 for b in names] for a in test["author"]])
            yp_f = np.where(allowed, logits, -np.inf).argmax(axis=1)
            present = sorted(set(y))
            low_idx = [names.index(a) for a in low]
            rec = recall_score(y, yp, labels=present, average=None, zero_division=0)
            per_author[name] = dict(zip([names[i] for i in present], rec))
            rows.append({"mô hình": name, "accuracy": accuracy_score(y, yp),
                         "macro-F1": f1_score(y, yp, labels=sorted(set(y) | set(yp)), average="macro", zero_division=0),
                         "recall TB tác giả ít dữ liệu": float(np.mean([per_author[name][names[i]] for i in low_idx])),
                         "accuracy + lọc thể loại": accuracy_score(y, yp_f),
                         "macro-F1 + lọc thể loại": f1_score(y, yp_f, labels=sorted(set(y) | set(yp_f)), average="macro",
                                                             zero_division=0)})
    t = pd.DataFrame(rows)
    pa = pd.DataFrame(per_author).loc[low]
    pa.insert(0, "số đoạn trong kho", counts[low].values)
    lines = [f"# So sánh trên tập test ({len(test)} đoạn, {test['author'].nunique()} tác giả có mặt)", "",
             f"Tác giả ít dữ liệu (< {LOW} đoạn và có mặt ở test): {', '.join(low)}", "",
             md(t), "", "## Recall từng tác giả ít dữ liệu", "",
             md(pa, index=True), ""]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
