"""Thí nghiệm theo thể loại trên kho v5 (cột genre): người dùng chọn thể loại trước khi định danh.

Mỗi seed chia kho MỘT lần (theo bài doc_id, phân tầng theo cặp thể loại|tác giả) rồi so sánh trên cùng các đoạn test:
  rieng_the_loai     : mỗi thể loại một mô hình, chỉ học tác giả của thể loại đó.
  chung_loc_the_loai : một mô hình chung học mọi tác giả, mọi thể loại; lúc dự đoán chỉ xét tác giả của thể loại đã chọn.
  chung_khong_loc    : mô hình chung, không lọc (người dùng không chọn thể loại).
  chung_chon_sai     : mô hình chung, người dùng chọn SAI thể loại (lọc theo tác giả của một thể loại khác),
                       gộp mọi cặp (đoạn, thể loại sai). Tác giả đúng thường không có trong danh sách nên gần như chắc sai.
Thêm phép thử chéo thể loại: mô hình riêng của thể loại X đoán các đoạn thể loại Y của những tác giả viết cả X và Y
(Tản Đà thơ -> văn xuôi...). Mô hình chưa thấy thể loại Y của tác giả đó, nên đây là mức tụt khi đổi thể loại.

Mô hình: 3 baseline + các chế độ mạng lai trong `fusions`. Kết quả: outputs/experiments/<tên>/seed_<s>/ và summary.md
(trung bình ± độ lệch chuẩn qua các seed). Seed đã có results.csv thì không chạy lại.
"""
from __future__ import annotations

import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from config import EXPERIMENT_DIR, TrainConfig
from .baselines import fit_baseline_scorers
from .data_loader import build_label_map, check_group_leak, load_corpus, make_splits
from .experiment import truncate_words
from .utils import get_logger, save_json

log = get_logger()

APPROACHES = ("rieng_the_loai", "chung_loc_the_loai", "chung_khong_loc", "chung_chon_sai")
ALL = "tat_ca"     # dòng gộp mọi thể loại


def genre_splits(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    """Chia train/val/test theo bài, phân tầng theo cặp thể loại|tác giả để mỗi cặp có mặt ở test nếu đủ bài."""
    keyed = make_splits(df.assign(author=df["genre"] + "|" + df["author"], split=None), cfg.val_size,
                        cfg.test_size, cfg.seed)
    out = df.copy()
    out["split"] = keyed["split"]
    leaked = check_group_leak(out)
    if leaked:
        raise ValueError(f"Chia sai: {len(leaked)} bài nằm ở nhiều tập")
    return out


def _metrics(true: Sequence[str], pred: Sequence[str]) -> dict:
    """Accuracy và macro-F1 trên các nhãn có mặt (như evaluate.compute_metrics), nhãn là tên tác giả."""
    true, pred = list(true), list(pred)
    labels = sorted(set(true) | set(pred))
    return {"n": len(true), "accuracy": float(accuracy_score(true, pred)) if true else float("nan"),
            "macro_f1": float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)) if true else float("nan")}


def _restricted_argmax(scores: np.ndarray, names: Sequence[str], allowed: set) -> list[str]:
    mask = np.array([n in allowed for n in names])
    s = np.where(mask[None, :], scores, -np.inf)
    return [names[i] for i in s.argmax(axis=1)]


def fit_and_score(df: pd.DataFrame, data_path: Path, base: TrainConfig, fusions: Sequence[str],
                  eval_sets: dict[str, list[str]], ckpt_root: Path, device_pref: str,
                  keep_checkpoints: bool) -> tuple[list[str], dict, dict]:
    """Huấn luyện 3 baseline + mạng lai trên df (đã có cột split) và trả về (tên tác giả theo chỉ số,
    {mô hình: {tên tập: ma trận điểm}}, {mô hình: giây huấn luyện})."""
    label_map = build_label_map(df)
    names = [a for a, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    train = df[df["split"] == "train"]
    scores, seconds = {}, {}
    t0 = time.time()
    scorers = fit_baseline_scorers(train["text"].to_numpy(), train["author"].map(label_map).to_numpy(),
                                   base.num_function_words, base.num_function_words, n_labels=len(names))
    for name, fn in scorers.items():
        scores[name] = {k: fn(np.asarray(v, dtype=object)) for k, v in eval_sets.items() if v}
    seconds["baselines"] = time.time() - t0

    for fusion in fusions:
        from .predict import AuthorPredictor
        from .train import run_training

        ckpt = ckpt_root / fusion
        t0 = time.time()
        run_training(replace(base, fusion=fusion), data_path, out_dir=ckpt, device_pref=device_pref, df=df)
        seconds[fusion] = time.time() - t0
        pred = AuthorPredictor.from_checkpoint(ckpt, device_pref)
        if pred.names != names:
            raise RuntimeError("Thứ tự nhãn của checkpoint khác build_label_map")
        scores[fusion] = {k: pred.predict_logits(v, batch_size=base.batch_size * 2)[0]
                          for k, v in eval_sets.items() if v}
        del pred
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        if not keep_checkpoints:     # chỉ cần số liệu; trọng số PhoBERT ~540MB mỗi lần
            for f in ("model.pt", "ngram.joblib"):
                (ckpt / f).unlink(missing_ok=True)
            shutil.rmtree(ckpt / "tokenizer", ignore_errors=True)
    return names, scores, seconds


def run_seed(df_all: pd.DataFrame, data_path: Path, base: TrainConfig, fusions: Sequence[str],
             short_lengths: Sequence[int], out_dir: Path, device_pref: str,
             keep_checkpoints: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = genre_splits(df_all, base)
    df[["genre", "author", "topic", "doc_id", "split"]].to_csv(out_dir / "split.csv", encoding="utf-8")
    genres = sorted(df["genre"].unique())
    authors_of = {g: set(df.loc[df["genre"] == g, "author"]) for g in genres}
    test = df[df["split"] == "test"]
    sizes = df.groupby(["genre", "split"]).size().unstack(fill_value=0)
    log.info("Seed %d — số đoạn:\n%s", base.seed, sizes.to_string())
    variants = {"full": lambda t: t, **{f"first_{n}w": (lambda t, n=n: truncate_words(t, n)) for n in short_lengths}}

    rows, preds = [], test[["genre", "author", "topic", "doc_id", "n_words"]].copy()

    # ---- mô hình chung: mọi thể loại, mọi tác giả
    sets = {v: [f(t) for t in test["text"]] for v, f in variants.items()}
    names, scores, sec_global = fit_and_score(df, data_path, base, fusions, sets, out_dir / "checkpoints" / "chung",
                                              device_pref, keep_checkpoints)
    genre_arr = test["genre"].to_numpy()
    for model, by_v in scores.items():
        for v, s in by_v.items():
            unf = [names[i] for i in s.argmax(axis=1)]
            filt = [None] * len(test)
            for g in genres:
                idx = np.flatnonzero(genre_arr == g)
                for i, p in zip(idx, _restricted_argmax(s[idx], names, authors_of[g])):
                    filt[i] = p
            if v == "full":
                preds[f"chung_khong_loc:{model}"] = unf
                preds[f"chung_loc_the_loai:{model}"] = filt
            for g in genres + [ALL]:
                m = (genre_arr == g) if g != ALL else np.ones(len(test), bool)
                true = test["author"].to_numpy()[m]
                for appr, p in (("chung_khong_loc", unf), ("chung_loc_the_loai", filt)):
                    pk = np.asarray(p, dtype=object)[m]
                    # gộp mọi thể loại: nhãn là cặp thể loại|tác giả để Tản Đà thơ và Tản Đà tự sự là hai lớp
                    r = _metrics(true, pk) if g != ALL else _metrics(genre_arr + "|" + true, genre_arr + "|" + pk)
                    rows.append({"seed": base.seed, "approach": appr, "model": model, "genre": g, "variant": v, **r})
                # chọn sai thể loại: mỗi đoạn được lọc theo từng thể loại khác
                wt, wp = [], []
                for g_true in (genres if g == ALL else [g]):
                    idx = np.flatnonzero(genre_arr == g_true)
                    for g_wrong in genres:
                        if g_wrong != g_true and len(idx):
                            wt += list(test["author"].to_numpy()[idx])
                            wp += _restricted_argmax(s[idx], names, authors_of[g_wrong])
                rows.append({"seed": base.seed, "approach": "chung_chon_sai", "model": model, "genre": g,
                             "variant": v, **_metrics(wt, wp)})

    # ---- mô hình riêng từng thể loại (+ phép thử chéo thể loại)
    cross_rows, sec_genre = [], {}
    per_genre_pred = {}     # (model, variant) -> list tên dự đoán theo thứ tự test
    for g in genres:
        sub = df[df["genre"] == g]
        sub_test = sub[sub["split"] == "test"]
        others = df[(df["genre"] != g) & df["author"].isin(authors_of[g])]   # cùng tác giả, thể loại khác
        sets = {v: [f(t) for t in sub_test["text"]] for v, f in variants.items()}
        sets["cheo"] = others["text"].tolist()
        g_names, g_scores, sec_genre[g] = fit_and_score(sub, data_path, base, fusions, sets,
                                                        out_dir / "checkpoints" / g, device_pref, keep_checkpoints)
        for model, by_v in g_scores.items():
            for v in variants:
                p = [g_names[i] for i in by_v[v].argmax(axis=1)]
                per_genre_pred.setdefault((model, v), {}).update(dict(zip(sub_test.index, p)))
                rows.append({"seed": base.seed, "approach": "rieng_the_loai", "model": model, "genre": g, "variant": v,
                             **_metrics(sub_test["author"], p)})
            if "cheo" in by_v:
                own = dict(zip(sub_test.index, [g_names[i] for i in by_v["full"].argmax(axis=1)]))
                cp = np.asarray([g_names[i] for i in by_v["cheo"].argmax(axis=1)], dtype=object)
                for (g2, a), part in others.groupby(["genre", "author"]):
                    pos = others.index.get_indexer(part.index)
                    own_idx = sub_test.index[sub_test["author"] == a]
                    cross_rows.append({
                        "seed": base.seed, "model": model, "train_genre": g, "test_genre": g2, "author": a,
                        "n": len(part), "accuracy": float((cp[pos] == a).mean()),
                        "n_in_genre": len(own_idx),
                        "accuracy_in_genre": float(np.mean([own[i] == a for i in own_idx])) if len(own_idx) else float("nan"),
                        "top_predicted": pd.Series(cp[pos]).value_counts().index[0]})
    for (model, v), d in per_genre_pred.items():
        if v == "full":
            preds[f"rieng_the_loai:{model}"] = [d[i] for i in test.index]
        p = np.asarray([d[i] for i in test.index], dtype=object)
        true = test["author"].to_numpy()
        rows.append({"seed": base.seed, "approach": "rieng_the_loai", "model": model, "genre": ALL, "variant": v,
                     **_metrics(genre_arr + "|" + true, genre_arr + "|" + p)})

    res, cross = pd.DataFrame(rows), pd.DataFrame(cross_rows)
    preds.to_csv(out_dir / "predictions.csv", encoding="utf-8")
    cross.to_csv(out_dir / "cross_genre.csv", index=False, encoding="utf-8")
    save_json({"split_sizes": sizes.to_dict(), "seconds": {"chung": sec_global, **sec_genre},
               "authors_of_genre": {g: sorted(a) for g, a in authors_of.items()}}, out_dir / "details.json")
    res.to_csv(out_dir / "results.csv", index=False, encoding="utf-8")   # ghi sau cùng: có file này = seed đã xong
    return res, cross


def _mean_std(df: pd.DataFrame, index, columns, value: str) -> pd.DataFrame:
    g = df.groupby(index + columns)[value]
    return pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1), "k": g.size()}).reset_index()


def _table(res: pd.DataFrame, value: str, variant: str, genres: Sequence[str]) -> str:
    sub = res[res["variant"] == variant]
    ms = _mean_std(sub, ["approach", "model"], ["genre"], value)
    cols = list(genres) + [ALL]
    lines = ["| cách làm | mô hình | " + " | ".join(cols) + " | trung bình các thể loại |",
             "|---" * (len(cols) + 3) + "|"]
    models = list(dict.fromkeys(sub["model"]))
    for appr in APPROACHES:
        for model in models:
            cells, per_genre = [], []
            for g in cols:
                r = ms[(ms.approach == appr) & (ms.model == model) & (ms.genre == g)]
                if r.empty:
                    cells.append("")
                    continue
                m, s = float(r["mean"].iloc[0]), r["std"].iloc[0]
                cells.append(f"{m:.4f}" + (f" ± {s:.4f}" if pd.notna(s) else ""))
                if g != ALL:
                    per_genre.append(m)
            if any(cells):
                avg = f"{np.mean(per_genre):.4f}" if per_genre else ""
                lines.append(f"| {appr} | {model} | " + " | ".join(cells) + f" | {avg} |")
    return "\n".join(lines)


def write_summary(out_root: Path, name: str, base: TrainConfig, seeds: Sequence[int], data_path: Path) -> None:
    res = pd.concat([pd.read_csv(out_root / f"seed_{s}" / "results.csv") for s in seeds
                     if (out_root / f"seed_{s}" / "results.csv").exists()], ignore_index=True)
    cross = pd.concat([pd.read_csv(out_root / f"seed_{s}" / "cross_genre.csv") for s in seeds
                       if (out_root / f"seed_{s}" / "cross_genre.csv").exists()], ignore_index=True)
    res.to_csv(out_root / "results_all_seeds.csv", index=False, encoding="utf-8")
    cross.to_csv(out_root / "cross_genre_all_seeds.csv", index=False, encoding="utf-8")
    genres = sorted(set(res["genre"]) - {ALL})
    done = sorted(res["seed"].unique())
    lines = [f"# Thí nghiệm theo thể loại: {name}", "", f"Dữ liệu: `{data_path}`", "",
             f"Seed: {', '.join(map(str, done))} (trung bình ± độ lệch chuẩn giữa các seed). Cấu hình mạng lai: "
             f"backbone={base.backbone}, epochs={base.epochs}, patience={base.patience}, batch={base.batch_size}, "
             f"max_length={base.max_length}, oversample={base.oversample}, class_bias={base.class_bias}, "
             f"char_ngram={base.char_ngram}.", "",
             "Cột `tat_ca` gộp mọi thể loại (nhãn = thể loại|tác giả). `chung_chon_sai` gộp mọi cặp (đoạn, thể loại sai).", ""]
    for value, title in (("macro_f1", "Macro-F1"), ("accuracy", "Accuracy")):
        lines += [f"## {title} — đoạn đầy đủ", "", _table(res, value, "full", genres), ""]
    for v in sorted(set(res["variant"]) - {"full"}, key=lambda s: int(s.split("_")[1][:-1])):
        lines += [f"## Macro-F1 — đoạn ngắn ({v})", "", _table(res, "macro_f1", v, genres), ""]
    if not cross.empty:
        c = cross.groupby(["model", "train_genre", "test_genre", "author"]).agg(
            n=("n", "first"), acc_cheo=("accuracy", "mean"), acc_cheo_std=("accuracy", "std"),
            acc_cung_the_loai=("accuracy_in_genre", "mean"),
            doan_nhieu_nhat=("top_predicted", lambda s: s.value_counts().index[0])).reset_index()
        lines += ["## Chéo thể loại — mô hình riêng của thể loại train đoán đoạn thể loại khác của cùng tác giả", "",
                  "`acc_cung_the_loai`: accuracy của cùng mô hình trên đoạn test cùng thể loại của tác giả đó (mốc so sánh).",
                  "", "| mô hình | train → test | tác giả | số đoạn | acc chéo | acc cùng thể loại | đoán nhiều nhất |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in c.iterrows():
            sd = f" ± {r.acc_cheo_std:.3f}" if pd.notna(r.acc_cheo_std) else ""
            lines.append(f"| {r.model} | {r.train_genre} → {r.test_genre} | {r.author} | {r.n} | {r.acc_cheo:.3f}{sd} | "
                         f"{r.acc_cung_the_loai:.3f} | {r.doan_nhieu_nhat} |")
        lines.append("")
    (out_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_genre_experiment(data_path: Path, name: str, base: TrainConfig, seeds: Sequence[int] = (42, 43, 44),
                         fusions: Sequence[str] = ("concat",), short_lengths: Sequence[int] = (25, 50),
                         device_pref: str = "auto", keep_checkpoints: bool = False,
                         out_root: Optional[Path] = None) -> pd.DataFrame:
    out_root = Path(out_root) if out_root else EXPERIMENT_DIR / name
    df_all = load_corpus(data_path)
    if "genre" not in df_all.columns:
        raise ValueError("Dữ liệu không có cột genre (dùng kho dựng bằng build-genre-corpus).")
    for s in seeds:
        seed_dir = out_root / f"seed_{s}"
        if (seed_dir / "results.csv").exists():
            log.info("Seed %d đã có kết quả, bỏ qua", s)
            continue
        t0 = time.time()
        run_seed(df_all, data_path, replace(base, seed=s), fusions, short_lengths, seed_dir, device_pref,
                 keep_checkpoints)
        log.info("Xong seed %d sau %.1f phút", s, (time.time() - t0) / 60)
        write_summary(out_root, name, base, seeds, data_path)
    write_summary(out_root, name, base, seeds, data_path)
    log.info("Đã ghi %s", out_root / "summary.md")
    return pd.read_csv(out_root / "results_all_seeds.csv")
