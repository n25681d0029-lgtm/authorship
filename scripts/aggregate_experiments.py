#!/usr/bin/env python3
"""Tổng hợp nhiều lần chạy `experiment` (các seed khác nhau): trung bình ± độ lệch chuẩn.

Chạy:  python scripts/aggregate_experiments.py --names vi_wikisource_v3 vi_wikisource_v3_s43 vi_wikisource_v3_s44 \
           --out vi_wikisource_v3_3seeds
Ghi:   outputs/experiments/<out>.md và <out>.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EXPERIMENT_DIR  # noqa: E402
from src.experiment import BASELINES, NEURAL_ORDER  # noqa: E402

HARD_AUTHORS = ("pham_quynh", "vu_trong_phung")


def ms(x) -> str:
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.4f} ± {x.std(ddof=1):.4f}" if len(x) > 1 else f"{x.mean():.4f}"


def load(names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    res, per_author, calib = [], [], []
    for n in names:
        d = EXPERIMENT_DIR / n
        r = pd.read_csv(d / "results.csv")
        r["run"] = n
        res.append(r)
        for sc in ("random", "cross_topic"):
            det = json.loads((d / sc / "details.json").read_text(encoding="utf-8"))
            for model, c in det.get("calibration", {}).items():
                calib.append({"run": n, "scenario": sc, "model": model, **{k: c.get(k) for k in
                              ("temperature", "test_ece_before", "test_ece_after", "reject_threshold")}})
            for f in (d / sc / "confusion").glob("*_metrics.json"):
                model = f.name[: -len("_metrics.json")]
                pc = json.loads(f.read_text(encoding="utf-8"))["per_class"]
                for a in HARD_AUTHORS:
                    if a in pc:
                        per_author.append({"run": n, "scenario": sc, "model": model, "author": a,
                                           "recall": pc[a]["recall"], "f1": pc[a]["f1-score"]})
    return pd.concat(res, ignore_index=True), pd.DataFrame(per_author), pd.DataFrame(calib)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res, pa, cal = load(args.names)
    order = [m for m in (*BASELINES, *NEURAL_ORDER) if m in set(res["model"])]
    lines = [f"# Tổng hợp {len(args.names)} lần chạy: {', '.join(args.names)}", "",
             "Giá trị: trung bình ± độ lệch chuẩn (ddof=1) qua các seed. Mỗi seed đổi cả cách chia dữ liệu lẫn khởi tạo.", ""]
    rows = []
    for metric in ("macro_f1", "accuracy"):
        lines += [f"## {metric} (đoạn test đầy đủ)", "", "| mô hình | random | cross_topic |", "|---|---|---|"]
        for m in order:
            cells = []
            for sc in ("random", "cross_topic"):
                v = res[(res.model == m) & (res.scenario == sc) & (res.variant == "full")][metric]
                cells.append(ms(v))
                rows.append({"metric": metric, "model": m, "scenario": sc, "variant": "full",
                             "mean": v.mean(), "std": v.std(ddof=1), "n": len(v)})
            lines.append(f"| {m} | {cells[0]} | {cells[1]} |")
        lines.append("")
    variants = [v for v in ("first_10w", "first_25w", "first_50w") if v in set(res["variant"])]
    for sc in ("random", "cross_topic"):
        lines += [f"## macro_f1 theo độ dài đoạn test ({sc})", "", "| mô hình | " + " | ".join(variants) + " |",
                  "|---" * (len(variants) + 1) + "|"]
        for m in order:
            cells = [ms(res[(res.model == m) & (res.scenario == sc) & (res.variant == v)]["macro_f1"]) for v in variants]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
    if not pa.empty:
        for a in HARD_AUTHORS:
            lines += [f"## Recall riêng {a}", "", "| mô hình | random | cross_topic |", "|---|---|---|"]
            for m in order:
                cells = [ms(pa[(pa.model == m) & (pa.scenario == sc) & (pa.author == a)]["recall"])
                         for sc in ("random", "cross_topic")]
                lines.append(f"| {m} | {cells[0]} | {cells[1]} |")
            lines.append("")
    # so sánh từng cặp có / không có n-gram trên cùng seed
    lines += ["## Cùng seed: có n-gram trừ không có n-gram (macro-F1, đoạn đầy đủ)", "",
              "| cặp | random: từng seed | cross_topic: từng seed |", "|---|---|---|"]
    full = res[res.variant == "full"].pivot_table(index=["run", "scenario"], columns="model", values="macro_f1")
    for a, b in (("style_only+ng", "style_only"), ("concat+ng", "concat"), ("gated+ng", "gated"),
                 ("gated+ng", "char_ngram_svm"), ("concat", "lm_only"), ("gated", "lm_only")):
        if a in full.columns and b in full.columns:
            cells = []
            for sc in ("random", "cross_topic"):
                d = (full[a] - full[b]).xs(sc, level="scenario")
                cells.append(", ".join(f"{x:+.4f}" for x in d) + f" (thắng {int((d > 0).sum())}/{len(d)})")
            lines.append(f"| {a} − {b} | {cells[0]} | {cells[1]} |")
    lines.append("")
    if not cal.empty:
        lines += ["## Hiệu chỉnh xác suất (ECE trên test, trung bình các seed)", "",
                  "| mô hình | kịch bản | T | ECE trước | ECE sau |", "|---|---|---|---|---|"]
        for (m, sc), g in cal.groupby(["model", "scenario"]):
            lines.append(f"| {m} | {sc} | {g.temperature.mean():.2f} | {g.test_ece_before.mean():.4f} | "
                         f"{g.test_ece_after.mean():.4f} |")
        lines.append("")
    (EXPERIMENT_DIR / f"{args.out}.md").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(rows).to_csv(EXPERIMENT_DIR / f"{args.out}.csv", index=False, encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
