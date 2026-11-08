"""Kết hợp mô hình lai với các bộ phân loại n-gram (SVM) — học trọng số trên val, chấm trên test.

Ý tưởng: thí nghiệm theo thể loại cho thấy char n-gram + SVM và mô hình lai ngang nhau nhưng nhìn văn bản theo cách
khác nhau (chuỗi ký tự vs ngữ cảnh PhoBERT + phong cách) nên sai ở những chỗ khác nhau → cộng log-xác suất có trọng
số thường tốt hơn từng mô hình. Điểm SVM được đổi thành xác suất bằng temperature scaling trên val.

  NgramMember : một bộ TF-IDF (ký tự 2-4 hoặc từ 1-2) + LinearSVC, kèm nhiệt độ T
  EnsembleHead: danh sách thành viên + trọng số; lưu cạnh checkpoint (ensemble.joblib) để AuthorPredictor dùng
  study()     : thí nghiệm — so mô hình lai / SVM / kết hợp trên test, có lọc và không lọc thể loại,
                khoảng tin cậy bootstrap theo bài (doc_id) cho chênh lệch accuracy
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .calibration import fit_temperature, softmax

NGRAM_KINDS = {
    "char_svm": dict(analyzer="char", ngram_range=(2, 4), sublinear_tf=True, max_features=200_000),
    "word_svm": dict(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, max_features=200_000,
                     token_pattern=r"(?u)\b\w+\b"),
}


def log_softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return np.log(np.clip(softmax(z, temperature), 1e-12, None))


@dataclass
class NgramMember:
    kind: str
    vec: object = None
    svm: object = None
    temperature: float = 1.0
    n_labels: int = 0

    def fit(self, texts: Sequence[str], y: np.ndarray, n_labels: int) -> "NgramMember":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC

        self.n_labels = n_labels
        self.vec = TfidfVectorizer(**NGRAM_KINDS[self.kind])
        self.svm = LinearSVC(C=1.0).fit(self.vec.fit_transform(list(texts)), y)
        return self

    def scores(self, texts: Sequence[str]) -> np.ndarray:
        s = self.svm.decision_function(self.vec.transform(list(texts)))
        if s.ndim == 1:
            s = np.stack([-s, s], axis=1)
        out = np.full((s.shape[0], self.n_labels), -30.0)     # tác giả không có trong train của SVM: điểm rất thấp
        out[:, self.svm.classes_.astype(int)] = s
        return out

    def logp(self, texts: Sequence[str]) -> np.ndarray:
        return log_softmax(self.scores(texts), self.temperature)


@dataclass
class EnsembleHead:
    """Lưu trong <checkpoint>/ensemble.joblib. AuthorPredictor tự nạp nếu có file này.

    log-xác suất kết hợp z = w_hybrid · trung bình_i log softmax(logit_i / T_i)   (mô hình chính + các mô hình phụ)
                           + Σ_k w_k · log softmax(điểm SVM_k / T_k)
    xác suất cuối = softmax(z / temperature); temperature và reject_threshold học lại trên val.
    """
    members: list = field(default_factory=list)          # NgramMember
    weights: dict = field(default_factory=dict)          # {"hybrid": w, "char_svm": w, ...}, cộng lại = 1
    hybrids: list = field(default_factory=list)          # tên checkpoint các mô hình lai phụ (cùng cách chia dữ liệu)
    temperature: float = 1.0
    reject_threshold: Optional[float] = None
    report: dict = field(default_factory=dict)

    def combine(self, hybrid_logp: np.ndarray, texts: Sequence[str]) -> np.ndarray:
        return self.combine_parts(hybrid_logp, self.member_logps(texts))

    def member_logps(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        return {m.kind: m.logp(texts) for m in self.members if self.weights.get(m.kind, 0.0)}

    def combine_parts(self, hybrid_logp: np.ndarray, parts: dict[str, np.ndarray]) -> np.ndarray:
        z = self.weights.get("hybrid", 1.0) * hybrid_logp
        for k, lp in parts.items():
            z = z + self.weights.get(k, 0.0) * lp
        return z

    def save(self, path: Path) -> None:
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "EnsembleHead":
        import joblib
        return joblib.load(path)


def _grid(names: list[str], step: float = 0.05):
    """Mọi bộ trọng số không âm, bước step, cộng lại bằng 1."""
    k = int(round(1 / step))
    for combo in itertools.product(range(k + 1), repeat=len(names) - 1):
        if sum(combo) <= k:
            yield dict(zip(names, [c * step for c in combo] + [(k - sum(combo)) * step]))


def fit_weights(logps: dict[str, np.ndarray], y: np.ndarray, step: float = 0.05) -> tuple[dict, float]:
    """Chọn trọng số làm log-loss trên val nhỏ nhất (ổn định hơn chọn theo accuracy khi val nhỏ)."""
    best, best_loss = None, np.inf
    names = list(logps)
    for w in _grid(names, step):
        z = sum(w[n] * logps[n] for n in names)
        p = softmax(z)[np.arange(len(y)), y]
        loss = float(-np.log(np.clip(p, 1e-12, None)).mean())
        if loss < best_loss - 1e-9:
            best, best_loss = w, loss
    return best, best_loss


def _restrict(z: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    return np.where(allowed, z, -np.inf)


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    from sklearn.metrics import f1_score
    labels = np.unique(np.concatenate([y, pred]))
    return {"accuracy": float((y == pred).mean()),
            "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0))}


def bootstrap_diff(y: np.ndarray, a: np.ndarray, b: np.ndarray, groups: np.ndarray, n: int = 2000,
                   seed: int = 0) -> tuple[float, float, float]:
    """Chênh lệch accuracy (b - a) và khoảng tin cậy 95% bootstrap lấy mẫu lại theo bài (cùng bài thì cùng ra/vào)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    da = np.bincount(inv, weights=(a == y).astype(float))
    db = np.bincount(inv, weights=(b == y).astype(float))
    cnt = np.bincount(inv)
    diffs = []
    for _ in range(n):
        s = rng.integers(0, len(uniq), len(uniq))
        diffs.append((db[s].sum() - da[s].sum()) / cnt[s].sum())
    return float((b == y).mean() - (a == y).mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def study(ckpt: str = "best_model", extra: Sequence[str] = (), data_path: Optional[Path] = None,
          kinds: Sequence[str] = ("char_svm", "word_svm"), save: bool = False, device: str = "auto") -> dict:
    """So mô hình lai (1 hoặc nhiều lần huấn luyện), từng SVM và các cách kết hợp trên đúng tập test của checkpoint
    (chia lại bằng genre_splits). Trọng số và nhiệt độ học trên val; save=True ghi <ckpt>/ensemble.joblib với tổ hợp
    đầy đủ (trọng số học trên val có thể bằng 0 nên tổ hợp đầy đủ không kém tổ hợp con trên val)."""
    from config import TrainConfig
    from .calibration import calibrate
    from .data_loader import load_corpus
    from .genre_experiment import genre_splits
    from .predict import AuthorPredictor, resolve_checkpoint
    from .utils import load_json

    path_ck = resolve_checkpoint(ckpt)
    meta = load_json(path_ck / "meta.json")
    info = meta["data"]
    cfg = TrainConfig.from_dict(meta["config"])
    df = load_corpus(Path(data_path) if data_path else Path(info["path"]), fmt=info["fmt"],
                     text_col=info["text_col"], author_col=info["author_col"])
    df = genre_splits(df, cfg)
    lm = meta["label_map"]
    df = df[df["author"].isin(lm)]
    part = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    y = {s: part[s]["author"].map(lm).to_numpy() for s in part}
    txt = {s: part[s]["text"].tolist() for s in part}
    L = len(lm)

    single = {}
    for name in [ckpt, *extra]:
        p = AuthorPredictor.from_checkpoint(name, device, use_ensemble=False)
        m2 = load_json(resolve_checkpoint(name) / "meta.json")
        if m2["label_map"] != lm or TrainConfig.from_dict(m2["config"]).data_seed != cfg.data_seed:
            raise ValueError(f"{name} khác danh sách tác giả hoặc cách chia dữ liệu với {ckpt}")
        single[name] = {s: log_softmax(p.predict_logits(txt[s])[0], p.temperature) for s in ("val", "test")}
        del p
    logps = {"hybrid": single[ckpt]}
    if extra:
        logps[f"hybrid_x{len(single)}"] = {s: np.mean([single[n][s] for n in single], axis=0) for s in ("val", "test")}
    hyb = f"hybrid_x{len(single)}" if extra else "hybrid"
    members = []
    for k in kinds:
        m = NgramMember(k).fit(txt["train"], y["train"], L)
        m.temperature = fit_temperature(m.scores(txt["val"]), y["val"])
        members.append(m)
        logps[k] = {s: m.logp(txt[s]) for s in ("val", "test")}

    combos = {n: [n] for n in logps}
    for h in dict.fromkeys(["hybrid", hyb]):
        for k in kinds:
            combos[f"{h}+{k}"] = [h, k]
        if len(kinds) > 1:
            combos[f"{h}+" + "+".join(kinds)] = [h, *kinds]
    full = f"{hyb}+" + "+".join(kinds)

    names = [a for a, _ in sorted(lm.items(), key=lambda kv: kv[1])]
    ga = info.get("genre_authors") or {}
    allowed = np.array([[n in ga.get(g, names) for n in names] for g in part["test"]["genre"]])
    groups = part["test"]["doc_id"].to_numpy() if "doc_id" in part["test"] else np.arange(len(part["test"]))
    rows, preds, weights, zval = {}, {}, {}, {}
    for cname, parts in combos.items():
        w = {parts[0]: 1.0} if len(parts) == 1 else fit_weights({p: logps[p]["val"] for p in parts}, y["val"])[0]
        weights[cname] = w
        zval[cname] = sum(w[p] * logps[p]["val"] for p in parts)
        z = sum(w[p] * logps[p]["test"] for p in parts)
        p_all, p_genre = z.argmax(1), _restrict(z, allowed).argmax(1)
        preds[cname] = p_genre
        rows[cname] = {"weights": {k: round(v, 2) for k, v in w.items()},
                       "khong_loc": _metrics(y["test"], p_all), "loc_the_loai": _metrics(y["test"], p_genre)}
        for g in sorted(set(part["test"]["genre"])):
            m = (part["test"]["genre"] == g).to_numpy()
            rows[cname][f"loc/{g}"] = _metrics(y["test"][m], p_genre[m])
    for cname in combos:
        if cname != "hybrid":
            rows[cname]["chenh_acc_loc_so_voi_hybrid"] = bootstrap_diff(y["test"], preds["hybrid"], preds[cname], groups)
    result = {"n_test": int(len(y["test"])), "n_val": int(len(y["val"])), "ket_qua": rows, "tong_hop_luu": full}

    if save:
        w = weights[full]
        cal = calibrate(zval[full], y["val"], cfg.accept_rate)
        head = EnsembleHead(members=[m for m in members if w.get(m.kind, 0.0)],
                            weights={("hybrid" if k == hyb else k): v for k, v in w.items()},
                            hybrids=list(extra), temperature=cal["temperature"],
                            reject_threshold=cal["reject_threshold"],
                            report={"tong_hop": full, **rows[full], "calibration": cal})
        head.save(path_ck / "ensemble.joblib")
        result["da_luu"] = str(path_ck / "ensemble.joblib")
    return result
