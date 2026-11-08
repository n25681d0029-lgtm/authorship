"""Quy trình huấn luyện đầy đủ: đọc dữ liệu -> đặc trưng -> mô hình -> vòng lặp -> lưu checkpoint.

Checkpoint là một thư mục tự chứa, chép sang máy khác dùng được ngay (không cần tải lại backbone):
    model.pt          trọng số toàn mô hình
    meta.json         cấu hình, ánh xạ nhãn, thông tin dữ liệu để đánh giá lại
    features.json     trạng thái bộ trích đặc trưng phong cách (từ chức năng, mean, std)
    ngram.joblib      bộ phân loại n-gram ký tự (chỉ khi char_ngram=True)
    backbone_config/  cấu hình kiến trúc của mô hình ngôn ngữ
    tokenizer/        tokenizer của mô hình ngôn ngữ
meta.json có thêm mục calibration: nhiệt độ T và ngưỡng từ chối học trên tập val.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from config import CHECKPOINT_DIR, REPORT_DIR, TrainConfig
from .data_loader import build_label_map, check_group_leak, load_corpus, make_splits
from .dataset import Collator, make_loader
from .calibration import calibrate, fit_class_bias
from .evaluate import collect_logits, run_model, save_evaluation
from .feature_extractor import StyleFeatureExtractor
from .models import HybridAuthorModel
from .ngram_features import CharNgramStacker, stacking_groups
from .utils import count_parameters, get_device, get_logger, progress, save_json, set_seed

log = get_logger()


def _linear_schedule(optimizer, total_steps: int, warmup_ratio: float):
    """Tăng tuyến tính learning rate trong giai đoạn khởi động rồi giảm tuyến tính về 0."""
    warmup = max(1, int(total_steps * warmup_ratio))

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def _param_groups(model: HybridAuthorModel, cfg: TrainConfig) -> list[dict]:
    """Backbone dùng lr nhỏ (tinh chỉnh nhẹ), phần đầu mới dùng lr lớn. Bias/LayerNorm không weight decay."""
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = name.startswith("backbone.")
        decay = p.ndim > 1
        key = (is_backbone, decay)
        groups.setdefault(key, {"params": [],
                                "lr": cfg.lr_backbone if is_backbone else cfg.lr_head,
                                "weight_decay": cfg.weight_decay if decay else 0.0})["params"].append(p)
    return list(groups.values())


def _move(batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def _forward(model, batch):
    return model(style=batch["style"], input_ids=batch.get("input_ids"),
                 attention_mask=batch.get("attention_mask"), length=batch.get("length"))


def save_checkpoint(out_dir: Path, model, cfg: TrainConfig, extractor: StyleFeatureExtractor,
                    label_map: dict, tokenizer, data_info: dict, style_dim: Optional[int] = None,
                    stacker: Optional[CharNgramStacker] = None, calibration: Optional[dict] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out_dir / "model.pt")
    extractor.save(out_dir / "features.json")
    if stacker is not None:
        stacker.save(out_dir / "ngram.joblib")
    save_json({"config": cfg.to_dict(), "label_map": label_map, "style_dim": style_dim or extractor.dim,
               "data": data_info, "calibration": calibration or {}}, out_dir / "meta.json")
    if model.backbone is not None:
        model.backbone.config.save_pretrained(out_dir / "backbone_config")
        if tokenizer is not None:
            tokenizer.save_pretrained(out_dir / "tokenizer")


def run_training(cfg: TrainConfig, data_path: Path, out_dir: Optional[Path] = None, fmt: str = "auto",
                 device_pref: str = "auto", test_topics: Optional[list[str]] = None,
                 text_col: str = "text", author_col: str = "author", name: Optional[str] = None,
                 df: Optional["pd.DataFrame"] = None) -> dict:
    """df: DataFrame đã có cột split (do experiment truyền vào để mọi mô hình dùng chung một cách chia).
    Nếu không có thì tự đọc data_path và chia theo cấu hình (cùng seed nên vẫn ra cùng kết quả)."""
    set_seed(cfg.seed)
    device = get_device(device_pref)
    log.info("Thiết bị: %s | fusion=%s | backbone=%s", device, cfg.fusion,
             cfg.backbone if cfg.fusion != "style_only" else "(không dùng)")

    # ---- 1. dữ liệu
    if df is None:
        df = load_corpus(data_path, fmt=fmt, text_col=text_col, author_col=author_col)
        if "genre" in df.columns and not test_topics and not cfg.topic_holdout:
            from .genre_experiment import genre_splits   # kho theo thể loại: phân tầng theo cặp thể loại|tác giả

            df = genre_splits(df, cfg)
        else:
            df = make_splits(df, cfg.val_size, cfg.test_size, cfg.data_seed, test_topics, cfg.topic_holdout or None)
    label_map = build_label_map(df)
    names = [a for a, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    sizes = df["split"].value_counts().to_dict()
    log.info("Dữ liệu: %d văn bản, %d tác giả, chia %s", len(df), len(label_map), sizes)
    if "train" not in sizes or "test" not in sizes:
        raise ValueError("Cần có cả tập train và test sau khi chia dữ liệu.")
    leaked = check_group_leak(df)
    if leaked:
        raise ValueError(f"{len(leaked)} bài có đoạn nằm ở nhiều tập, ví dụ {leaked[:3]}.")

    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}

    # ---- 2. đặc trưng phong cách (fit chỉ trên train)
    extractor = StyleFeatureExtractor(cfg.num_function_words)
    style = {"train": extractor.fit_transform(parts["train"]["text"].tolist())}
    for s in ("val", "test"):
        style[s] = extractor.transform(parts[s]["text"].tolist()) if len(parts[s]) else np.zeros((0, extractor.dim), np.float32)
    stacker = None
    if cfg.char_ngram:
        stacker = CharNgramStacker(len(label_map))
        tr = parts["train"]
        ng = {"train": stacker.fit_transform(tr["text"].tolist(), tr["author"].map(label_map).to_numpy(),
                                             stacking_groups(tr))}
        for s in ("val", "test"):
            ng[s] = stacker.transform(parts[s]["text"].tolist()) if len(parts[s]) else np.zeros((0, len(label_map)), np.float32)
        style = {s: np.hstack([style[s], ng[s]]).astype(np.float32) for s in style}
        oof_acc = float((ng["train"].argmax(axis=1) == tr["author"].map(label_map).to_numpy()).mean())
        log.info("N-gram ký tự: thêm %d chiều xác suất; accuracy out-of-fold trên train %.4f", len(label_map), oof_acc)
    style_dim = style["train"].shape[1]
    log.info("Đặc trưng phong cách: %d chiều", style_dim)

    # ---- 3. tokenizer và mô hình
    tokenizer = None
    if cfg.fusion != "style_only":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    collator = Collator(tokenizer, cfg.max_length, cfg.lang)
    labels = {s: parts[s]["author"].map(label_map).tolist() for s in parts}
    loaders = {
        s: make_loader(parts[s]["text"].tolist(), style[s], labels[s], collator, cfg.batch_size,
                       shuffle=(s == "train"), num_workers=cfg.num_workers,
                       balanced=(s == "train" and cfg.oversample), seed=cfg.seed)
        for s in parts if len(parts[s])
    }

    model = HybridAuthorModel(
        num_labels=len(label_map), style_dim=style_dim, fusion=cfg.fusion,
        backbone_name=cfg.backbone if cfg.fusion != "style_only" else None,
        freeze_backbone=cfg.freeze_backbone, style_hidden=cfg.style_hidden, style_out=cfg.style_out,
        fused_dim=cfg.fused_dim, dropout=cfg.dropout,
    ).to(device)
    total, trainable = count_parameters(model)
    log.info("Tham số: %s (huấn luyện: %s)", f"{total:,}", f"{trainable:,}")

    # ---- 4. tối ưu
    optimizer = torch.optim.AdamW(_param_groups(model, cfg))
    total_steps = cfg.epochs * len(loaders["train"])
    scheduler = _linear_schedule(optimizer, total_steps, cfg.warmup_ratio)
    weight = None
    if cfg.class_weight:
        counts = np.bincount(np.asarray(labels["train"]), minlength=len(label_map)).astype(np.float64)
        weight = torch.tensor(counts.sum() / (len(label_map) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
        log.info("Trọng số lớp: %.2f – %.2f", float(weight.min()), float(weight.max()))
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)

    best_f1, best_state, bad_epochs, history = -1.0, None, 0, []
    eval_split = "val" if "val" in loaders else "test"
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for batch in progress(loaders["train"], desc=f"epoch {epoch}/{cfg.epochs}", leave=False):
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(_forward(model, batch), batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(batch["labels"])
            seen += len(batch["labels"])

        y_true, y_pred, _ = run_model(model, loaders[eval_split], device)
        from sklearn.metrics import accuracy_score, f1_score

        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": running / max(1, seen),
                        f"{eval_split}_acc": acc, f"{eval_split}_macro_f1": f1})
        log.info("epoch %d | loss %.4f | %s acc %.4f | %s macro-F1 %.4f", epoch, running / max(1, seen),
                 eval_split, acc, eval_split, f1)

        if f1 > best_f1:
            best_f1, bad_epochs = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                log.info("Dừng sớm: %d epoch liên tiếp không cải thiện.", bad_epochs)
                break
    log.info("Huấn luyện xong sau %.1f giây", time.time() - t0)

    # ---- 5. đánh giá trên test bằng trọng số tốt nhất
    if best_state is not None:
        model.load_state_dict(best_state)
    out_dir = Path(out_dir) if out_dir else CHECKPOINT_DIR / (name or f"{cfg.fusion}_{time.strftime('%Y%m%d_%H%M%S')}")

    # ---- 6. hiệu chỉnh xác suất và ngưỡng từ chối trên tập val (không đổi nhãn dự đoán)
    calibration, bias = {}, np.zeros(len(label_map))
    if "val" in loaders:
        y_val, z_val = collect_logits(model, loaders["val"], device)
        if cfg.class_bias:
            bias = fit_class_bias(z_val, y_val)
            log.info("Độ ưu tiên từng tác giả (val): %s", {names[i]: round(float(b), 2) for i, b in enumerate(bias) if b})
        calibration = calibrate(z_val + bias, y_val, cfg.accept_rate)
        calibration["class_bias"] = [float(b) for b in bias]
        log.info("Hiệu chỉnh: T = %.2f | ECE val %.4f -> %.4f | ngưỡng từ chối %.3f (chấp nhận %.0f%% đoạn val)",
                 calibration["temperature"], calibration["val_ece_before"], calibration["val_ece_after"],
                 calibration["reject_threshold"], 100 * cfg.accept_rate)

    y_true, z_test = collect_logits(model, loaders["test"], device)
    if cfg.class_bias:   # lưu thêm kết quả khi chưa cộng độ ưu tiên để so sánh
        m0 = save_evaluation(y_true, z_test.argmax(axis=1), names, out_dir / "reports", "test_nobias")
        log.info("TEST (chưa cộng độ ưu tiên) | accuracy %.4f | macro-F1 %.4f", m0["accuracy"], m0["macro_f1"])
    y_pred = (z_test + bias).argmax(axis=1)
    metrics = save_evaluation(y_true, y_pred, names, out_dir / "reports", "test")
    save_json(history, out_dir / "reports" / "history.json")
    log.info("TEST | accuracy %.4f | macro-F1 %.4f", metrics["accuracy"], metrics["macro_f1"])

    data_info = {"path": str(data_path), "fmt": fmt, "text_col": text_col, "author_col": author_col,
                 "test_topics": test_topics, "n_docs": int(len(df)), "split_sizes": {k: int(v) for k, v in sizes.items()}}
    if "genre" in df.columns:   # để giao diện lọc tác giả theo thể loại người dùng chọn
        data_info["genre_authors"] = {g: sorted(a) for g, a in df.groupby("genre")["author"].unique().items()}
    save_checkpoint(out_dir, model, cfg, extractor, label_map, tokenizer, data_info, style_dim, stacker, calibration)
    log.info("Đã lưu checkpoint tại: %s", out_dir)
    return {"checkpoint": str(out_dir), "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            "best_val_macro_f1": best_f1, "history": history}
