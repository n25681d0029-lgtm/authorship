"""Các baseline truyền thống để so sánh với mô hình lai.

  1. Burrows' Delta          : z-score tần suất các từ phổ biến nhất, khoảng cách Manhattan.
  2. Char n-gram + LinearSVC : TF-IDF n-gram ký tự (2 đến 4), baseline rất mạnh cho định danh tác giả.
  3. Stylometry + Logistic   : chỉ dùng vector đặc trưng phong cách, không có mô hình ngôn ngữ.

Cả ba chạy trên CPU trong vài giây đến vài phút, không cần torch.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .evaluate import compute_metrics
from .feature_extractor import StyleFeatureExtractor
from .text_utils import normalize, tokenize_words


class BurrowsDelta:
    def __init__(self, n_words: int = 150) -> None:
        self.n_words = n_words
        self.vocab: list[str] = []
        self.mu = self.sigma = None
        self.profiles: dict[int, np.ndarray] = {}

    def _freqs(self, text: str) -> np.ndarray:
        toks = [t.lower() for t in tokenize_words(normalize(text))]
        c, n = Counter(toks), max(1, len(toks))
        return np.array([c.get(w, 0) / n for w in self.vocab])

    def fit(self, texts, labels) -> "BurrowsDelta":
        total = Counter()
        for t in texts:
            total.update(w.lower() for w in tokenize_words(normalize(t)))
        self.vocab = [w for w, _ in total.most_common(self.n_words)]
        f = np.stack([self._freqs(t) for t in texts])
        self.mu = f.mean(axis=0)
        self.sigma = f.std(axis=0)
        self.sigma[self.sigma < 1e-12] = 1.0
        z = (f - self.mu) / self.sigma
        labels = np.asarray(labels)
        # hồ sơ của mỗi tác giả = trung bình z-score các văn bản của tác giả đó
        self.profiles = {int(a): z[labels == a].mean(axis=0) for a in np.unique(labels)}
        return self

    def scores(self, texts) -> np.ndarray:
        """Điểm (càng lớn càng giống) theo thứ tự sorted(profiles) = -Delta."""
        z = (np.stack([self._freqs(t) for t in texts]) - self.mu) / self.sigma
        # Delta = trung bình |z_văn_bản - z_hồ_sơ|; tác giả có Delta nhỏ nhất là gần nhất
        return -np.stack([np.abs(z - self.profiles[a]).mean(axis=1) for a in sorted(self.profiles)], axis=1)

    def predict(self, texts) -> np.ndarray:
        return np.asarray(sorted(self.profiles))[self.scores(texts).argmax(axis=1)]


def _expand(scores: np.ndarray, classes, n_labels: int) -> np.ndarray:
    """Đưa ma trận điểm theo classes (nhãn có trong train) về đủ n_labels cột; nhãn vắng nhận -inf."""
    out = np.full((scores.shape[0], n_labels), -np.inf)
    out[:, np.asarray(classes, dtype=int)] = scores
    return out


def fit_baseline_scorers(texts, labels, n_words: int = 150, num_function_words: int = 150,
                         n_labels: int | None = None) -> dict:
    """Như fit_baselines nhưng trả về {tên: hàm cho ma trận điểm (số văn bản x n_labels)}; argmax của điểm
    trùng với nhãn dự đoán. Cần điểm để lọc theo danh sách tác giả ứng viên (ví dụ tác giả của một thể loại)."""
    texts, labels = np.asarray(texts), np.asarray(labels)
    n_labels = n_labels or int(labels.max()) + 1
    delta = BurrowsDelta(n_words).fit(texts, labels)

    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True, max_features=200_000)
    svm = LinearSVC(C=1.0).fit(vec.fit_transform(texts), labels)

    ext = StyleFeatureExtractor(num_function_words)
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(ext.fit_transform(texts), labels)

    def svm_scores(t):
        s = svm.decision_function(vec.transform(t))
        if s.ndim == 1:                      # chỉ 2 tác giả: một cột, dương là classes_[1]
            s = np.stack([-s, s], axis=1)
        return _expand(s, svm.classes_, n_labels)

    return {
        "burrows_delta": lambda t: _expand(delta.scores(t), sorted(delta.profiles), n_labels),
        "char_ngram_svm": svm_scores,
        "stylometry_logreg": lambda t: _expand(lr.predict_log_proba(ext.transform(t)), lr.classes_, n_labels),
    }


def fit_baselines(texts, labels, n_words: int = 150, num_function_words: int = 150) -> dict:
    """Huấn luyện cả ba baseline trên (texts, labels). Trả về {tên: hàm dự đoán nhãn số}."""
    scorers = fit_baseline_scorers(texts, labels, n_words, num_function_words)
    return {name: (lambda t, f=f: f(t).argmax(axis=1)) for name, f in scorers.items()}


def run_baselines(df: pd.DataFrame, label_map: dict[str, int], n_words: int = 150,
                  num_function_words: int = 150) -> dict:
    """Huấn luyện trên split=train, báo cáo trên val và test. Trả về {tên: {split: metrics}}."""
    names = [a for a, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    y = df["author"].map(label_map).to_numpy()
    parts = {s: (df["split"] == s).to_numpy() for s in ("train", "val", "test")}
    text = df["text"].to_numpy()
    results: dict[str, dict] = {}
    for name, predict_fn in fit_baselines(text[parts["train"]], y[parts["train"]], n_words,
                                          num_function_words).items():
        results[name] = {}
        for s in ("val", "test"):
            if parts[s].any():
                m = compute_metrics(y[parts[s]], predict_fn(text[parts[s]]), names)
                results[name][s] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "n": m["n"]}
    return results
