"""Thí nghiệm đánh giá đầy đủ, mọi mô hình dùng CHUNG một cách chia dữ liệu trong mỗi kịch bản.

Kịch bản:
  random      : chia theo bài (doc_id), phân tầng theo tác giả. Cùng một tác phẩm có thể có chương
                ở train và chương khác ở test, nên tên nhân vật, bối cảnh có thể giúp mô hình "đoán".
  cross_topic : với mỗi tác giả, giữ lại trọn một số tác phẩm (topic) làm test; train không thấy
                tác phẩm đó. Đây là phép thử mô hình nhận ra phong cách hay chỉ nhận ra chủ đề.

Trong mỗi kịch bản:
  - 3 baseline (Burrows' Delta, char n-gram + SVM, stylometry + LogReg) và các chế độ mạng lai
    (style_only, lm_only, concat, gated) để làm ablation.
  - accuracy, macro-F1 trên tập test; ma trận nhầm lẫn cho từng mô hình.
  - văn bản ngắn: cắt mỗi đoạn test còn N từ đầu (mặc định 10, 25, 50) rồi đánh giá lại.
  - độ chính xác theo nhóm độ dài của đoạn test đầy đủ.

Kết quả ghi vào outputs/experiments/<tên>/ (results.csv, summary.md và thư mục con cho từng kịch bản).
"""
from __future__ import annotations

import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from config import EXPERIMENT_DIR, TrainConfig
from .baselines import fit_baselines
from .data_loader import build_label_map, check_group_leak, load_corpus, make_splits
from .evaluate import compute_metrics, save_evaluation
from .utils import get_logger, save_json

log = get_logger()

BASELINES = ("burrows_delta", "char_ngram_svm", "stylometry_logreg")
NEURAL_ORDER = ("style_only", "lm_only", "concat", "gated", "style_only+ng", "concat+ng", "gated+ng")
LENGTH_BINS = [0, 100, 150, 10_000]
LENGTH_LABELS = ["<100 từ", "100-149 từ", ">=150 từ"]


def truncate_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def split_stats(df: pd.DataFrame) -> dict:
    """Số đoạn mỗi tác giả trong từng tập, và tỷ lệ đoạn test có tác phẩm đã xuất hiện ở train."""
    counts = df.pivot_table(index="author", columns="split", values="text", aggfunc="size", fill_value=0)
    out = {"sizes": df["split"].value_counts().to_dict(),
           "per_author": counts.reset_index().to_dict(orient="records"),
           "authors_missing_in_test": sorted(set(df["author"]) - set(df.loc[df["split"] == "test", "author"]))}
    if out["authors_missing_in_test"]:
        log.warning("Tác giả không có đoạn nào trong test: %s", out["authors_missing_in_test"])
    if "topic" in df.columns:
        train_topics = set(df.loc[df["split"] == "train", "topic"].astype(str))
        test = df[df["split"] == "test"]
        out["test_topics"] = int(test["topic"].nunique())
        out["test_passages_with_seen_topic"] = float(test["topic"].astype(str).isin(train_topics).mean())
    if "doc_id" in df.columns:
        out["docs"] = {s: int(df.loc[df["split"] == s, "doc_id"].nunique()) for s in ("train", "val", "test")}
    return out


def parse_model_name(name: str) -> tuple[str, bool]:
    """'gated+ng' -> ('gated', True): chế độ hợp nhất và có ghép n-gram ký tự vào luồng phong cách hay không."""
    fusion, _, flag = name.partition("+")
    return fusion, flag == "ng"


def _neural_predictions(cfg: TrainConfig, df: pd.DataFrame, data_path: Path, variants: dict, y_test: np.ndarray,
                        ckpt_dir: Path, device_pref: str, keep_checkpoint: bool) -> tuple[dict, dict, float, dict]:
    from .calibration import ece, softmax
    from .predict import AuthorPredictor
    from .train import run_training

    t0 = time.time()
    res = run_training(cfg, data_path, out_dir=ckpt_dir, device_pref=device_pref, df=df)
    seconds = time.time() - t0
    pred = AuthorPredictor.from_checkpoint(ckpt_dir, device_pref)
    preds, gates, calib = {}, {}, dict(pred.calibration)
    for v, texts in variants.items():
        logits, gate = pred.predict_logits(texts, batch_size=cfg.batch_size * 2)
        preds[v] = logits.argmax(axis=1)
        if gate is not None:
            gates[v] = float(gate.mean())
        if v == "full":
            calib["test_ece_before"] = ece(softmax(logits), y_test)
            calib["test_ece_after"] = ece(softmax(logits, pred.temperature), y_test)
    del pred
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    if not keep_checkpoint:
        # trọng số PhoBERT ~540MB mỗi chế độ; thí nghiệm chỉ cần số liệu nên xóa trọng số, giữ báo cáo
        for name in ("model.pt", "ngram.joblib"):
            (ckpt_dir / name).unlink(missing_ok=True)
        shutil.rmtree(ckpt_dir / "tokenizer", ignore_errors=True)
    log.info("%s%s: best val macro-F1 %.4f, huấn luyện %.0f giây", cfg.fusion, "+ng" if cfg.char_ngram else "",
             res["best_val_macro_f1"], seconds)
    return preds, gates, seconds, calib


def run_scenario(df_all: pd.DataFrame, data_path: Path, base: TrainConfig, scenario: str, out_dir: Path,
                 fusions: Sequence[str], short_lengths: Sequence[int], topic_holdout: float,
                 device_pref: str, keep_checkpoints: bool) -> pd.DataFrame:
    holdout = topic_holdout if scenario == "cross_topic" else None
    df = make_splits(df_all, base.val_size, base.test_size, base.seed, topic_holdout=holdout)
    leaked = check_group_leak(df)
    if leaked:
        raise ValueError(f"Chia sai: {len(leaked)} bài nằm ở nhiều tập")
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ("author", "topic", "doc_id", "n_words", "split") if c in df.columns]
    df[cols].to_csv(out_dir / "split.csv", encoding="utf-8")
    stats = split_stats(df)
    save_json(stats, out_dir / "split_stats.json")
    log.info("[%s] chia: %s | đoạn test có tác phẩm đã thấy ở train: %.1f%%", scenario, stats["sizes"],
             100 * stats.get("test_passages_with_seen_topic", float("nan")))

    label_map = build_label_map(df)
    names = [a for a, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    y_test = test["author"].map(label_map).to_numpy()
    variants = {"full": test["text"].tolist()}
    for n in short_lengths:
        variants[f"first_{n}w"] = [truncate_words(t, n) for t in test["text"]]

    preds, gates, seconds, calibration = {}, {}, {}, {}
    t0 = time.time()
    for name, fn in fit_baselines(train["text"].to_numpy(), train["author"].map(label_map).to_numpy(),
                                  base.num_function_words, base.num_function_words).items():
        preds[name] = {v: fn(np.asarray(texts, dtype=object)) for v, texts in variants.items()}
    seconds["baselines"] = time.time() - t0
    for model_name in fusions:
        fusion, use_ngram = parse_model_name(model_name)
        cfg = replace(base, fusion=fusion, char_ngram=use_ngram, topic_holdout=holdout or 0.0)
        preds[model_name], g, seconds[model_name], calibration[model_name] = _neural_predictions(
            cfg, df, data_path, variants, y_test, out_dir / "checkpoints" / model_name.replace("+", "_"),
            device_pref, keep_checkpoints)
        if g:
            gates[model_name] = g

    rows, len_rows, confusions = [], [], {}
    lengths = test["n_words"].to_numpy() if "n_words" in test.columns else np.array([len(t.split()) for t in test["text"]])
    bucket = pd.cut(lengths, LENGTH_BINS, right=False, labels=LENGTH_LABELS)
    for model, by_variant in preds.items():
        for v, yp in by_variant.items():
            m = compute_metrics(y_test, yp, names)
            rows.append({"scenario": scenario, "model": model, "variant": v, "n": m["n"],
                         "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
        full = save_evaluation(y_test, by_variant["full"], names, out_dir / "confusion", model)
        confusions[model] = full["top_confusions"]
        for b in LENGTH_LABELS:
            mask = np.asarray(bucket == b)
            if mask.any():
                len_rows.append({"scenario": scenario, "model": model, "bucket": b, "n": int(mask.sum()),
                                 "accuracy": float((by_variant["full"][mask] == y_test[mask]).mean())})
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "results.csv", index=False, encoding="utf-8")
    # dự đoán từng đoạn test (đoạn đầy đủ) để phân tích lỗi theo tác phẩm, theo nguồn
    per_passage = test[[c for c in ("author", "topic", "doc_id", "n_words") if c in test.columns]].copy()
    for model, by_variant in preds.items():
        per_passage[model] = [names[int(k)] for k in by_variant["full"]]
    per_passage.to_csv(out_dir / "predictions.csv", encoding="utf-8")
    pd.DataFrame(len_rows).to_csv(out_dir / "by_length.csv", index=False, encoding="utf-8")
    save_json({"top_confusions": confusions, "gate_mean_lm_weight": gates, "seconds": seconds,
               "calibration": calibration, "label_names": names}, out_dir / "details.json")
    return res


def _markdown(res: pd.DataFrame, value: str) -> str:
    known = (*BASELINES, *NEURAL_ORDER)
    order = [m for m in known if m in set(res["model"])] + sorted(set(res["model"]) - set(known))
    t = res.pivot_table(index="model", columns=["scenario", "variant"], values=value).reindex(order)
    t.columns = [f"{s} / {v}" for s, v in t.columns]
    lines = ["| model | " + " | ".join(t.columns) + " |", "|---" * (len(t.columns) + 1) + "|"]
    for model, row in t.iterrows():
        lines.append(f"| {model} | " + " | ".join(f"{x:.4f}" for x in row.values) + " |")
    return "\n".join(lines)


def run_experiment(data_path: Path, name: str, base: TrainConfig, scenarios: Sequence[str] = ("random", "cross_topic"),
                   fusions: Sequence[str] = ("style_only", "lm_only", "concat", "gated"),
                   short_lengths: Sequence[int] = (10, 25, 50), topic_holdout: float = 0.3, fmt: str = "auto",
                   device_pref: str = "auto", keep_checkpoints: bool = False,
                   out_root: Optional[Path] = None) -> pd.DataFrame:
    out_root = Path(out_root) if out_root else EXPERIMENT_DIR / name
    df_all = load_corpus(data_path, fmt=fmt)
    all_res = []
    for sc in scenarios:
        all_res.append(run_scenario(df_all, data_path, base, sc, out_root / sc, fusions, short_lengths,
                                    topic_holdout, device_pref, keep_checkpoints))
    res = pd.concat(all_res, ignore_index=True)
    res.to_csv(out_root / "results.csv", index=False, encoding="utf-8")
    summary = [f"# Thí nghiệm {name}", "", f"Dữ liệu: `{data_path}`", "",
               f"Cấu hình: backbone={base.backbone}, epochs={base.epochs}, batch={base.batch_size}, "
               f"max_length={base.max_length}, seed={base.seed}, topic_holdout={topic_holdout}", "",
               "## Accuracy", "", _markdown(res, "accuracy"), "", "## Macro-F1", "", _markdown(res, "macro_f1"), ""]
    (out_root / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    log.info("Đã ghi kết quả vào %s", out_root)
    return res
