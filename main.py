#!/usr/bin/env python3
"""Điểm vào duy nhất của hệ thống định danh tác giả (Stylometry + mô hình ngôn ngữ).

Ví dụ:
    python main.py check
    python main.py demo-data --lang en
    python main.py baseline --data demo_en.csv
    python main.py train --data demo_en.csv --fusion concat --epochs 5
    python main.py predict --ckpt concat_demo --text "Câu cần kiểm tra ..."
Chạy từ thư mục nào cũng được: file này tự thêm thư mục dự án vào sys.path.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

# Console Windows cũ mặc định không phải UTF-8 nên in tiếng Việt dễ lỗi; ép UTF-8 nếu có thể.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (CHECKPOINT_DIR, DEFAULT_BACKBONES, FUSION_MODES, OUTPUT_DIR, RAW_DATA_DIR, REPORT_DIR,  # noqa: E402
                    TrainConfig, ensure_dirs)


def resolve_data(arg: str) -> Path:
    """Đường dẫn dữ liệu: ưu tiên đường dẫn người dùng đưa, nếu không có thì tìm trong data/raw."""
    p = Path(arg).expanduser()
    if p.exists():
        return p.resolve()
    q = RAW_DATA_DIR / arg
    if q.exists():
        return q.resolve()
    raise FileNotFoundError(f"Không tìm thấy dữ liệu '{arg}' (đã thử {p} và {q}). "
                            f"Tạo dữ liệu thử bằng: python main.py demo-data")


def cmd_check(args) -> int:
    print(f"Python {platform.python_version()} trên {platform.system()} {platform.machine()}")
    print(f"Thư mục dự án : {ROOT}")
    print(f"Dữ liệu (raw) : {RAW_DATA_DIR}")
    print(f"Đầu ra        : {OUTPUT_DIR}")
    ok = True
    for mod, required in [("numpy", True), ("pandas", True), ("sklearn", True), ("torch", True),
                          ("transformers", True), ("tokenizers", False), ("tqdm", False),
                          ("matplotlib", False), ("pyvi", False)]:
        try:
            m = __import__(mod)
            print(f"  [ok]      {mod:<13}{getattr(m, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            print(f"  [{'THIẾU' if required else 'tùy chọn'}] {mod:<13}{type(e).__name__}")
            ok = ok and not required
    try:
        from src.utils import get_device
        print(f"Thiết bị tính toán: {get_device()}")
    except Exception:  # noqa: BLE001
        pass
    return 0 if ok else 1


def cmd_demo_data(args) -> int:
    from src.demo_data import generate_demo_corpus
    ensure_dirs()
    path = generate_demo_corpus(Path(args.out) if args.out else RAW_DATA_DIR, lang=args.lang,
                                n_authors=args.authors, docs_per_author=args.docs, seed=args.seed,
                                holdout_topic=args.holdout_topic)
    print(f"Đã tạo dữ liệu thử: {path}")
    return 0


def cmd_tiny_backbone(args) -> int:
    from src.data_loader import load_corpus
    from src.tiny_backbone import build_tiny_backbone
    ensure_dirs()
    df = load_corpus(resolve_data(args.data))
    out = Path(args.out) if args.out else OUTPUT_DIR / "tiny_backbone"
    build_tiny_backbone(df["text"].tolist(), out)
    print(f"Đã tạo backbone nhỏ (ngẫu nhiên, chỉ để chạy thử) tại: {out}")
    print(f"Dùng bằng: python main.py train --data {args.data} --backbone \"{out}\"")
    return 0


def _config_from_args(args) -> TrainConfig:
    return TrainConfig(
        lang=args.lang, backbone=args.backbone or "", fusion=args.fusion, max_length=args.max_length,
        batch_size=args.batch_size, epochs=args.epochs, lr_backbone=args.lr_backbone, lr_head=args.lr_head,
        dropout=args.dropout, freeze_backbone=args.freeze_backbone, num_function_words=args.function_words,
        patience=args.patience, seed=args.seed, val_size=args.val_size, test_size=args.test_size,
        label_smoothing=args.label_smoothing, topic_holdout=args.topic_holdout or 0.0,
        char_ngram=args.char_ngram, accept_rate=args.accept_rate, class_weight=args.class_weight,
        oversample=args.oversample, class_bias=args.class_bias,
        split_seed=getattr(args, "split_seed", None))


def cmd_build_corpus(args) -> int:
    import pandas as pd
    from src.corpus_builder import build_corpus
    from src.utils import save_json
    ensure_dirs()
    df, stats = build_corpus(resolve_data(args.src), args.min_words, args.max_words, args.target_words,
                             args.per_author, args.min_per_author, args.seed, args.authors, args.max_doc_passages)
    out = Path(args.out) if args.out else RAW_DATA_DIR / "corpus.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    save_json(stats, out.with_suffix(".stats.json"))
    print(pd.DataFrame(stats["per_author"]).to_string(index=False))
    if stats["dropped_authors"]:
        print(f"Loại vì ít hơn {args.min_per_author} đoạn: {stats['dropped_authors']}")
    print(f"{stats['n_passages']} đoạn, {stats['n_authors']} tác giả -> {out}")
    return 0


def cmd_build_genre_corpus(args) -> int:
    import pandas as pd
    from config import DATA_DIR
    from src.corpus_builder import build_genre_corpus, load_genres
    from src.utils import save_json
    ensure_dirs()
    genres = load_genres(args.genres or DATA_DIR / "genres.json")
    mins = dict(kv.split("=", 1) for kv in (args.min_per_author or []))
    df, stats = build_genre_corpus(
        Path(args.src) if args.src else DATA_DIR / "v5_genre" / "raw", genres,
        per_author=args.per_author, min_per_author={k: int(v) for k, v in mins.items()},
        min_works=args.min_works, min_authors=args.min_authors, max_invalid_ratio=args.max_invalid_ratio,
        max_doc_passages=args.max_doc_passages, seed=args.seed)
    out = Path(args.out) if args.out else DATA_DIR / "v5_corpus.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    save_json(stats, out.with_suffix(".stats.json"))
    for genre, g in stats["per_genre"].items():
        print(f"\n== {genre}: {g['n_authors']} tác giả, {g['n_passages']} đoạn")
        print(pd.DataFrame(g["per_author"]).to_string(index=False))
    if stats["dropped_pairs"]:
        print("\nCặp (thể loại/tác giả) bị loại:")
        for k, v in stats["dropped_pairs"].items():
            print(f"  {k}: {v}")
    if stats["dropped_genres"]:
        print(f"Thể loại bị loại: {stats['dropped_genres']}")
    if stats["noisy_passages_removed"]:
        print(f"Đoạn bị loại vì nhiều âm tiết sai (OCR): {stats['noisy_passages_removed']}")
    print(f"\n{stats['n_passages']} đoạn, {stats['n_genres']} thể loại, {stats['n_authors']} tác giả -> {out}")
    return 0


def cmd_experiment(args) -> int:
    from src.experiment import run_experiment
    ensure_dirs()
    res = run_experiment(resolve_data(args.data), args.name, _config_from_args(args), args.scenarios, args.fusions,
                         args.short_lengths, args.topic_holdout or 0.3, args.fmt, args.device, args.keep_checkpoints)
    full = res[res["variant"] == "full"].pivot_table(index="model", columns="scenario", values=["accuracy", "macro_f1"])
    print(full.to_string(float_format=lambda x: f"{x:.4f}"))
    return 0


def cmd_genre_experiment(args) -> int:
    from src.genre_experiment import run_genre_experiment
    ensure_dirs()
    res = run_genre_experiment(resolve_data(args.data), args.name, _config_from_args(args), args.seeds, args.fusions,
                               args.short_lengths, args.device, args.keep_checkpoints)
    full = res[(res["variant"] == "full")].pivot_table(index=["approach", "model"], columns="genre", values="macro_f1")
    print(full.to_string(float_format=lambda x: f"{x:.4f}"))
    return 0


def cmd_baseline(args) -> int:
    from src.baselines import run_baselines
    from src.data_loader import build_label_map, load_corpus, make_splits
    from src.utils import save_json
    ensure_dirs()
    df = load_corpus(resolve_data(args.data), fmt=args.fmt)
    df = make_splits(df, args.val_size, args.test_size, args.seed, args.test_topics, args.topic_holdout)
    res = run_baselines(df, build_label_map(df), args.function_words, args.function_words)
    print(f"{'phương pháp':<20}{'val acc':>10}{'val F1':>10}{'test acc':>10}{'test F1':>10}")
    for name, r in res.items():
        v, t = r.get("val", {}), r.get("test", {})
        print(f"{name:<20}{v.get('accuracy', float('nan')):>10.4f}{v.get('macro_f1', float('nan')):>10.4f}"
              f"{t.get('accuracy', float('nan')):>10.4f}{t.get('macro_f1', float('nan')):>10.4f}")
    save_json(res, REPORT_DIR / "baselines.json")
    return 0


def cmd_train(args) -> int:
    from src.train import run_training
    ensure_dirs()
    res = run_training(_config_from_args(args), resolve_data(args.data), fmt=args.fmt,
                       device_pref=args.device, test_topics=args.test_topics, name=args.name)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, ensure_ascii=False, indent=2))
    return 0


def cmd_ablation(args) -> int:
    """Chạy lần lượt các chế độ fusion trên cùng dữ liệu và in bảng so sánh."""
    import pandas as pd
    from src.train import run_training
    ensure_dirs()
    rows = []
    for fusion in args.fusions:
        cfg = _config_from_args(args)
        cfg.fusion = fusion
        res = run_training(cfg, resolve_data(args.data), fmt=args.fmt, device_pref=args.device,
                           test_topics=args.test_topics, name=f"{args.name or 'ablation'}_{fusion}")
        rows.append({"fusion": fusion, "accuracy": res["accuracy"], "macro_f1": res["macro_f1"]})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    table.to_csv(REPORT_DIR / f"{args.name or 'ablation'}.csv", index=False)
    return 0


def cmd_ensemble(args) -> int:
    from src.ensemble import study
    res = study(args.ckpt, args.extra, resolve_data(args.data) if args.data else None, save=args.save)
    out = CHECKPOINT_DIR / args.ckpt / "reports" / "ensemble_study.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, r in res["ket_qua"].items():
        d = r.get("chenh_acc_loc_so_voi_hybrid")
        print(f"{name:40s} không lọc acc {r['khong_loc']['accuracy']:.3f} F1 {r['khong_loc']['macro_f1']:.3f} | "
              f"lọc thể loại acc {r['loc_the_loai']['accuracy']:.3f} F1 {r['loc_the_loai']['macro_f1']:.3f}"
              + (f" | chênh acc {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" if d else "") + f" | w {r['weights']}")
    print("Đã ghi", out, "| bộ gộp:", res.get("da_luu", "(không lưu, thêm --save)"))
    return 0


def cmd_evaluate(args) -> int:
    from src.predict import evaluate_checkpoint
    m = evaluate_checkpoint(args.ckpt, resolve_data(args.data) if args.data else None, args.split, args.device)
    print(f"{args.split}: accuracy {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} | n={m['n']}")
    for c in m["top_confusions"]:
        print(f"  nhầm {c['true']} -> {c['predicted']}: {c['count']} lần")
    return 0


def cmd_predict(args) -> int:
    from src.data_loader import read_text_file
    from src.predict import AuthorPredictor
    if bool(args.text) == bool(args.file):
        print("Cần đúng một trong hai: --text hoặc --file", file=sys.stderr)
        return 2
    text = args.text if args.text else read_text_file(Path(args.file).expanduser())
    pred = AuthorPredictor.from_checkpoint(args.ckpt, args.device)
    if args.threshold == "auto":
        threshold = pred.reject_threshold          # ngưỡng học trên tập val lúc huấn luyện (None với checkpoint cũ)
    else:
        threshold = float(args.threshold) or None  # --threshold 0: không bao giờ từ chối
    r = pred.predict([text], top_k=args.top_k, reject_below=threshold)[0]
    print(f"Tác giả dự đoán: {r['author']}  (độ tin cậy {r['confidence']:.1%}"
          + (f", ngưỡng từ chối {threshold:.1%})" if threshold else ")"))
    for t in r["top_k"]:
        print(f"  {t['author']:<24}{t['prob']:.1%}")
    if r["n_chunks"] > 1:
        print(f"Văn bản dài nên được cắt thành {r['n_chunks']} đoạn; xác suất trên là trung bình các đoạn:")
        for c in r["chunks"]:
            print(f"  đoạn {c['n_words']:>3} âm tiết -> {c['author']:<20}{c['prob']:.1%}")
    if "lm_weight" in r:
        print(f"Trọng số luồng ngôn ngữ trong cổng: {r['lm_weight']:.2f} (còn lại là luồng phong cách)")
    print("Đặc trưng phong cách nổi bật (z so với kho huấn luyện):")
    for e in r["style_evidence"]:
        print(f"  {e['feature']:<18} giá trị {e['value']:>9.3f}   z = {e['z']:+.2f}")
    return 0


def cmd_openset(args) -> int:
    from src.openset import evaluate_open_set
    r = evaluate_open_set(args.ckpt, Path(args.unknown), args.device,
                          data_path=resolve_data(args.data) if args.data else None)
    print(f"{r['n_known']} đoạn test đã biết, {r['n_unknown']} đoạn lạ | T = {r['temperature']:.2f}"
          + (f" | ngưỡng lưu trong checkpoint {r['stored_threshold']:.3f}" if r["stored_threshold"] else ""))
    for name in ("uncalibrated", "calibrated"):
        c = r[name]
        print(f"[{name}] AUROC {c['auroc']:.3f} | tin cậy TB đã biết {c['mean_conf_known']:.3f}, lạ "
              f"{c['mean_conf_unknown']:.3f} | đoạn lạ bị gán cho: {c['unknown_assigned_to']}")
        print(f"  {'ngưỡng':>7}{'chấp nhận đã biết':>19}{'acc khi chấp nhận':>19}{'từ chối đoạn lạ':>17}")
        for row in c["by_threshold"]:
            print(f"  {row['threshold']:>7.3f}{row['accept_known']:>19.1%}{row['acc_accepted']:>19.1%}"
                  f"{row['reject_unknown']:>17.1%}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Định danh tác giả: Stylometry + mô hình ngôn ngữ")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Kiểm tra môi trường và thư viện").set_defaults(fn=cmd_check)

    p = sub.add_parser("demo-data", help="Sinh dữ liệu nhân tạo để chạy thử")
    p.add_argument("--lang", choices=list(DEFAULT_BACKBONES), default="en")
    p.add_argument("--authors", type=int, default=5)
    p.add_argument("--docs", type=int, default=40, help="số văn bản mỗi tác giả")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--holdout-topic", action="store_true", help="chủ đề cuối chỉ xuất hiện ở test")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_demo_data)

    p = sub.add_parser("tiny-backbone", help="Tạo mô hình BERT nhỏ ngẫu nhiên để chạy thử offline")
    p.add_argument("--data", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_tiny_backbone)

    def add_common(p):
        p.add_argument("--data", required=True, help="CSV, thư mục tác giả, hoặc thư mục C50")
        p.add_argument("--fmt", choices=["auto", "csv", "c50", "folder"], default="auto")
        p.add_argument("--lang", choices=list(DEFAULT_BACKBONES), default="en")
        p.add_argument("--val-size", type=float, default=0.15)
        p.add_argument("--test-size", type=float, default=0.15)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--split-seed", type=int, default=None,
                       help="seed cho cách chia dữ liệu (mặc định = --seed); dùng khi huấn luyện thêm mô hình để gộp")
        p.add_argument("--function-words", type=int, default=150)
        p.add_argument("--test-topics", nargs="*", default=None, help="chủ đề dành riêng cho tập test")
        p.add_argument("--topic-holdout", type=float, default=None,
                       help="mỗi tác giả giữ lại các chủ đề chiếm tỷ lệ này làm test (chéo chủ đề), ví dụ 0.3")

    def add_train_args(p):
        add_common(p)
        p.add_argument("--fusion", choices=FUSION_MODES, default="concat")
        p.add_argument("--backbone", default="", help="tên trên Hugging Face hoặc đường dẫn thư mục cục bộ")
        p.add_argument("--max-length", type=int, default=256)
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--epochs", type=int, default=5)
        p.add_argument("--lr-backbone", type=float, default=2e-5)
        p.add_argument("--lr-head", type=float, default=1e-3)
        p.add_argument("--dropout", type=float, default=0.3)
        p.add_argument("--label-smoothing", type=float, default=0.0)
        p.add_argument("--patience", type=int, default=2)
        p.add_argument("--freeze-backbone", action="store_true")
        p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
        p.add_argument("--name", default=None, help="tên thư mục checkpoint")
        p.add_argument("--char-ngram", action="store_true",
                       help="ghép xác suất n-gram ký tự (out-of-fold) vào luồng phong cách")
        p.add_argument("--oversample", action="store_true",
                       help="rút đoạn train cân bằng giữa các tác giả (tác giả ít dữ liệu xuất hiện nhiều hơn)")
        p.add_argument("--class-bias", action="store_true",
                       help="học độ ưu tiên từng tác giả trên tập val để tăng macro-F1 (tăng độ nhạy tác giả bị bỏ sót)")
        p.add_argument("--class-weight", action="store_true",
                       help="trọng số lớp tỷ lệ nghịch với số đoạn (khi số đoạn giữa tác giả chênh lệch)")
        p.add_argument("--accept-rate", type=float, default=0.95,
                       help="ngưỡng từ chối chọn sao cho tỷ lệ này số đoạn val được chấp nhận")

    p = sub.add_parser("baseline", help="Chạy Burrows' Delta, char n-gram + SVM, stylometry + LogReg")
    add_common(p)
    p.set_defaults(fn=cmd_baseline)

    p = sub.add_parser("train", help="Huấn luyện mô hình lai")
    add_train_args(p)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("ablation", help="Chạy nhiều chế độ fusion và so sánh")
    add_train_args(p)
    p.add_argument("--fusions", nargs="+", choices=FUSION_MODES, default=["style_only", "lm_only", "concat", "gated"])
    p.set_defaults(fn=cmd_ablation)

    p = sub.add_parser("experiment", help="Baseline + ablation trên cùng cách chia, hai kịch bản, đoạn ngắn")
    add_train_args(p)
    p.add_argument("--fusions", nargs="*", choices=list(FUSION_MODES) + [f"{f}+ng" for f in FUSION_MODES],
                   default=["style_only", "lm_only", "concat", "gated"],
                   help="thêm hậu tố +ng để ghép n-gram ký tự vào luồng phong cách, ví dụ gated+ng")
    p.add_argument("--scenarios", nargs="+", choices=["random", "cross_topic"], default=["random", "cross_topic"])
    p.add_argument("--short-lengths", nargs="*", type=int, default=[10, 25, 50],
                   help="cắt đoạn test còn N từ đầu để đánh giá văn bản ngắn")
    p.add_argument("--keep-checkpoints", action="store_true", help="giữ trọng số mô hình (mặc định xóa cho nhẹ)")
    p.set_defaults(fn=cmd_experiment, name="experiment")

    p = sub.add_parser("genre-experiment",
                       help="Kho v5: mô hình riêng từng thể loại vs mô hình chung lọc theo thể loại, chọn sai thể loại, "
                            "chéo thể loại; nhiều seed")
    add_train_args(p)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--fusions", nargs="*", choices=FUSION_MODES, default=["concat"])
    p.add_argument("--short-lengths", nargs="*", type=int, default=[25, 50])
    p.add_argument("--keep-checkpoints", action="store_true")
    p.set_defaults(fn=cmd_genre_experiment, name="v5_the_loai", lang="vi")

    p = sub.add_parser("build-corpus", help="Dựng CSV đoạn văn 50-200 từ từ thư mục <tác giả>/<tác phẩm>/*.txt")
    p.add_argument("--src", required=True, help="ví dụ data/raw/wikisource")
    p.add_argument("--out", default=None, help="mặc định data/raw/corpus.csv")
    p.add_argument("--min-words", type=int, default=50)
    p.add_argument("--max-words", type=int, default=200)
    p.add_argument("--target-words", type=int, default=150)
    p.add_argument("--per-author", type=int, default=300, help="số đoạn tối đa mỗi tác giả")
    p.add_argument("--min-per-author", type=int, default=100, help="ít hơn thì loại tác giả")
    p.add_argument("--authors", nargs="*", default=None, help="chỉ lấy các tác giả này")
    p.add_argument("--max-doc-passages", type=int, default=20,
                   help="trang dài hơn số đoạn này được chia thành nhiều bài (doc_id) liên tiếp")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_build_corpus)

    p = sub.add_parser("build-genre-corpus",
                       help="Dựng kho v5 theo thể loại từ data/v5_genre/raw/<thể loại>/<tác giả>/<tác phẩm>/*.txt")
    p.add_argument("--src", default=None, help="mặc định data/v5_genre/raw")
    p.add_argument("--out", default=None, help="mặc định data/v5_corpus.csv (+ .stats.json)")
    p.add_argument("--genres", default=None, help="mặc định data/genres.json")
    p.add_argument("--per-author", type=int, default=200, help="số đoạn tối đa mỗi cặp (thể loại, tác giả)")
    p.add_argument("--min-per-author", nargs="*", default=None,
                   help="ghi đè ngưỡng số đoạn theo thể loại, ví dụ: tho=30 ky=30")
    p.add_argument("--min-works", type=int, default=None, help="số tác phẩm tối thiểu mỗi cặp (mặc định theo genres.json)")
    p.add_argument("--min-authors", type=int, default=None, help="số tác giả tối thiểu mỗi thể loại")
    p.add_argument("--max-invalid-ratio", type=float, default=0.03,
                   help="bỏ đoạn có tỷ lệ âm tiết sai cấu trúc lớn hơn (lọc bản OCR chưa soát)")
    p.add_argument("--max-doc-passages", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_build_genre_corpus)

    p = sub.add_parser("ensemble", help="Gộp mô hình lai (+ các lần huấn luyện khác) với SVM n-gram, học trọng số trên val")
    p.add_argument("--ckpt", default="best_model")
    p.add_argument("--extra", nargs="*", default=[], help="checkpoint mô hình lai phụ cùng cách chia (--split-seed)")
    p.add_argument("--data", default=None)
    p.add_argument("--save", action="store_true", help="ghi <ckpt>/ensemble.joblib để predict/app dùng")
    p.set_defaults(fn=cmd_ensemble)

    p = sub.add_parser("evaluate", help="Đánh giá lại một checkpoint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default=None, help="nếu dữ liệu đã chuyển sang vị trí khác")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("predict", help="Dự đoán tác giả cho văn bản mới")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--text", default=None)
    p.add_argument("--file", default=None)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--threshold", default="auto",
                   help="dưới ngưỡng này trả về 'unknown'; auto = ngưỡng học trên tập val; 0 = không từ chối")
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("openset", help="Đánh giá từ chối 'unknown' với văn bản của tác giả ngoài kho")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--unknown", required=True, help="thư mục hoặc file .txt của tác giả KHÔNG có trong kho")
    p.add_argument("--data", default=None, help="nếu dữ liệu huấn luyện đã chuyển sang vị trí khác")
    p.add_argument("--device", default="auto")
    p.set_defaults(fn=cmd_openset)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"Lỗi: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
