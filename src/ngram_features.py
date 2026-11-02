"""Đặc trưng n-gram ký tự cho luồng phong cách (stacking).

Một bộ phân loại n-gram ký tự (TF-IDF 2-4 ký tự + hồi quy logistic) cho ra C xác suất tác giả; C số này được
ghép vào vector phong cách trước khi đưa vào StyleEncoder.

Nếu dùng chính dự đoán của bộ phân loại trên tập train thì các xác suất gần như luôn đúng (bộ phân loại đã thấy
văn bản đó), mạng lai sẽ học tin tuyệt đối vào chúng. Vì vậy trên tập train dùng dự đoán out-of-fold: mỗi đoạn
được dự đoán bởi một bộ phân loại không thấy nhóm của nó khi học. Nhóm là tác phẩm (topic) để mô phỏng việc gặp
tác phẩm mới; tác giả có ít hơn 3 tác phẩm trong train thì dùng bài (doc_id) làm nhóm, vì giữ lại trọn tác phẩm
sẽ làm bộ phân loại của fold đó không còn mẫu nào của tác giả.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

MIN_TOPICS_FOR_TOPIC_GROUPS = 3


def stacking_groups(df: pd.DataFrame) -> np.ndarray:
    """Nhãn nhóm cho dự đoán out-of-fold: tác phẩm nếu tác giả có đủ tác phẩm, nếu không thì bài, nếu không có
    cả hai cột thì từng đoạn là một nhóm."""
    if "topic" not in df.columns:
        key = df["doc_id"].astype(str) if "doc_id" in df.columns else pd.Series(df.index.astype(str), index=df.index)
        return ("d:" + key).to_numpy()
    n_topics = df.groupby("author")["topic"].transform("nunique")
    fine = df["doc_id"].astype(str) if "doc_id" in df.columns else pd.Series(df.index.astype(str), index=df.index)
    groups = np.where(n_topics >= MIN_TOPICS_FOR_TOPIC_GROUPS, "t:" + df["topic"].astype(str), "d:" + fine)
    return groups


class CharNgramStacker:
    def __init__(self, n_labels: int, n_folds: int = 5, C: float = 10.0, max_features: int = 100_000) -> None:
        self.n_labels = n_labels
        self.n_folds = n_folds
        self.C = C
        self.max_features = max_features
        self.vec: Optional[TfidfVectorizer] = None
        self.clf: Optional[LogisticRegression] = None

    def _new(self) -> tuple[TfidfVectorizer, LogisticRegression]:
        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True, max_features=self.max_features)
        return vec, LogisticRegression(C=self.C, max_iter=2000)

    def _proba(self, vec, clf, texts) -> np.ndarray:
        """Xác suất đủ C cột (lớp không có trong dữ liệu học của fold thì xác suất 0)."""
        out = np.zeros((len(texts), self.n_labels), dtype=np.float32)
        out[:, clf.classes_] = clf.predict_proba(vec.transform(texts))
        return out

    def fit_transform(self, texts: Sequence[str], labels: Sequence[int], groups: Sequence[str]) -> np.ndarray:
        texts, labels, groups = np.asarray(texts, dtype=object), np.asarray(labels), np.asarray(groups)
        oof = np.zeros((len(texts), self.n_labels), dtype=np.float32)
        n_folds = min(self.n_folds, len(set(groups)))
        for tr, te in GroupKFold(n_splits=n_folds).split(texts, labels, groups):
            vec, clf = self._new()
            clf.fit(vec.fit_transform(texts[tr]), labels[tr])
            oof[te] = self._proba(vec, clf, texts[te])
        self.vec, self.clf = self._new()
        self.clf.fit(self.vec.fit_transform(texts), labels)
        return oof

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Gọi fit_transform() trước khi transform().")
        return self._proba(self.vec, self.clf, np.asarray(texts, dtype=object))

    def save(self, path: Path) -> None:
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "CharNgramStacker":
        import joblib

        return joblib.load(path)
