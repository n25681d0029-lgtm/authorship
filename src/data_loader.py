"""Đọc kho văn bản và chia train/val/test.

Ba kiểu dữ liệu được hỗ trợ, tự nhận diện khi fmt="auto":

1. csv     : một file .csv có cột text, author (tùy chọn: split, topic, doc_id).
2. c50     : thư mục Reuters C50 gốc, có C50train/<tác giả>/*.txt và C50test/<tác giả>/*.txt.
3. folder  : data/<tác giả>/*.txt, hoặc có chia sẵn data/train/<tác giả>/*.txt, data/test/...

Kết quả luôn là một DataFrame có các cột: text, author, split, và topic (nếu có).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from .text_utils import normalize

SPLIT_NAMES = {"train": "train", "training": "train", "val": "val", "valid": "val",
               "validation": "val", "dev": "val", "test": "test", "testing": "test"}


def read_text_file(path: Path) -> str:
    """Đọc file văn bản, thử vài bảng mã phổ biến để tránh lỗi trên dữ liệu từ nhiều nguồn."""
    raw = Path(path).read_bytes()
    for enc in ("utf-8", "utf-8-sig", "utf-16", "cp1258", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def _read_author_dirs(root: Path, split: Optional[str]) -> list[dict]:
    rows = []
    for author_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(author_dir.glob("*.txt")):
            rows.append({"text": read_text_file(f), "author": author_dir.name,
                         "split": split, "source": str(f.relative_to(root.parent))})
    return rows


def _find_child(root: Path, name: str) -> Optional[Path]:
    for p in root.iterdir():
        if p.is_dir() and p.name.lower() == name.lower():
            return p
    return None


def detect_format(path: Path) -> str:
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() == ".csv":
            return "csv"
        raise ValueError(f"Không nhận diện được định dạng của file {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Không tìm thấy đường dẫn dữ liệu: {path}")
    if _find_child(path, "C50train") is not None and _find_child(path, "C50test") is not None:
        return "c50"
    return "folder"


def load_corpus(path: Path | str, fmt: str = "auto", text_col: str = "text",
                author_col: str = "author", topic_col: str = "topic",
                min_chars: int = 20) -> pd.DataFrame:
    """Đọc dữ liệu thành DataFrame chuẩn (text, author, split, [topic])."""
    path = Path(path).expanduser()
    if fmt == "auto":
        fmt = detect_format(path)

    if fmt == "csv":
        df = pd.read_csv(path, encoding="utf-8")
        missing = [c for c in (text_col, author_col) if c not in df.columns]
        if missing:
            raise ValueError(f"File CSV thiếu cột {missing}. Các cột hiện có: {list(df.columns)}")
        df = df.rename(columns={text_col: "text", author_col: "author"})
        if topic_col in df.columns and topic_col != "topic":
            df = df.rename(columns={topic_col: "topic"})
        if "split" in df.columns:
            df["split"] = df["split"].map(lambda s: SPLIT_NAMES.get(str(s).lower(), None))
        else:
            df["split"] = None
    elif fmt == "c50":
        rows = _read_author_dirs(_find_child(path, "C50train"), "train")
        rows += _read_author_dirs(_find_child(path, "C50test"), "test")
        df = pd.DataFrame(rows)
    elif fmt == "folder":
        split_dirs = {SPLIT_NAMES[p.name.lower()]: p for p in path.iterdir()
                      if p.is_dir() and p.name.lower() in SPLIT_NAMES}
        rows = []
        if split_dirs:
            for split, d in split_dirs.items():
                rows += _read_author_dirs(d, split)
        else:
            rows = _read_author_dirs(path, None)
        if not rows:
            raise ValueError(f"Không có file .txt nào trong {path}. Cấu trúc mong đợi: {path}/<tác giả>/*.txt")
        df = pd.DataFrame(rows)
    else:
        raise ValueError(f"fmt không hợp lệ: {fmt}")

    df["text"] = df["text"].astype(str).map(normalize)
    df["author"] = df["author"].astype(str)
    before = len(df)
    df = df[df["text"].str.strip().str.len() >= min_chars].reset_index(drop=True)
    if len(df) < before:
        warnings.warn(f"Bỏ {before - len(df)} văn bản rỗng hoặc quá ngắn (< {min_chars} ký tự).", stacklevel=2)
    if df["author"].nunique() < 2:
        raise ValueError("Cần ít nhất 2 tác giả để huấn luyện bộ phân loại.")
    return df


GROUP_COL = "doc_id"   # các đoạn cùng một bài (chương, truyện ngắn) luôn nằm chung một tập


def _safe_split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tách (phần còn lại, phần test) có phân tầng theo tác giả.

    Nếu dữ liệu có cột doc_id thì chia theo nhóm: mọi đoạn của cùng một bài vào cùng một phía,
    để câu chữ, tên nhân vật của một bài không xuất hiện ở cả train lẫn test.
    """
    labels = df["author"]
    if GROUP_COL in df.columns and df[GROUP_COL].nunique() < len(df):
        from sklearn.model_selection import StratifiedGroupKFold

        n_splits = max(2, round(1.0 / test_size))
        try:
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            rest_idx, test_idx = next(sgkf.split(df, labels, groups=df[GROUP_COL].astype(str)))
            return df.iloc[rest_idx], df.iloc[test_idx]
        except ValueError:
            warnings.warn("Không chia phân tầng theo nhóm được, chuyển sang chia theo nhóm không phân tầng.",
                          stacklevel=2)
            from sklearn.model_selection import GroupShuffleSplit

            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            rest_idx, test_idx = next(gss.split(df, labels, groups=df[GROUP_COL].astype(str)))
            return df.iloc[rest_idx], df.iloc[test_idx]
    try:
        a, b = train_test_split(df, test_size=test_size, random_state=seed, stratify=labels)
    except ValueError:
        warnings.warn("Có tác giả quá ít văn bản, chia ngẫu nhiên không phân tầng.", stacklevel=2)
        a, b = train_test_split(df, test_size=test_size, random_state=seed)
    return a, b


def _holdout_topics(df: pd.DataFrame, fraction: float, seed: int) -> pd.Series:
    """Với từng tác giả, chọn ngẫu nhiên một số chủ đề (tác phẩm) làm test cho tới khi đủ
    khoảng fraction số đoạn của tác giả đó. Luôn để lại ít nhất một chủ đề cho train."""
    import random

    rng = random.Random(seed)
    is_test = pd.Series(False, index=df.index)
    for author, part in df.groupby("author"):
        sizes = part["topic"].astype(str).value_counts()
        topics = sorted(sizes.index)
        if len(topics) < 2:
            warnings.warn(f"Tác giả {author} chỉ có 1 chủ đề, không có mặt trong tập test.", stacklevel=3)
            continue
        rng.shuffle(topics)
        chosen, n = [], 0
        for t in topics[:-1]:
            if n >= fraction * len(part):
                break
            chosen.append(t)
            n += sizes[t]
        is_test |= df.index.isin(part.index[part["topic"].astype(str).isin(chosen)])
    return is_test


def make_splits(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42,
                test_topics: Optional[Iterable[str]] = None, topic_holdout: Optional[float] = None) -> pd.DataFrame:
    """Gán cột split cho mọi dòng. Quy tắc theo thứ tự ưu tiên:

    - test_topics: các văn bản thuộc chủ đề này làm tập test (kiểm tra đổi chủ đề).
    - topic_holdout: với mỗi tác giả giữ lại khoảng tỷ lệ này số đoạn, gồm trọn các chủ đề
      (tác phẩm) mà train không thấy, làm tập test (kiểm tra chéo chủ đề).
    - split có sẵn (ví dụ C50): giữ nguyên, nếu thiếu val thì tách từ train.
    - còn lại: chia ngẫu nhiên có phân tầng theo tác giả (theo nhóm doc_id nếu có).
    """
    df = df.copy()
    random_test = False
    if test_topics or topic_holdout:
        if "topic" not in df.columns:
            raise ValueError("Dữ liệu không có cột topic nên không dùng được test_topics/topic_holdout.")
        if test_topics:
            wanted = {str(t) for t in test_topics}
            is_test = df["topic"].astype(str).isin(wanted)
            if not is_test.any():
                raise ValueError(f"Không có văn bản nào thuộc chủ đề {sorted(wanted)}.")
        else:
            is_test = _holdout_topics(df, float(topic_holdout), seed)
        df["split"] = "train"
        df.loc[is_test, "split"] = "test"
    elif df["split"].notna().all() and "test" in set(df["split"]):
        pass  # đã có sẵn train/test
    else:
        _, test = _safe_split(df, test_size, seed)
        df["split"] = "train"
        df.loc[test.index, "split"] = "test"
        random_test = True

    if "val" not in set(df["split"]):
        # val_size được hiểu là tỷ lệ so với toàn bộ dữ liệu khi test cũng được chia ngẫu nhiên
        share = val_size / (1.0 - test_size) if random_test else val_size
        share = min(max(share, 0.05), 0.5)
        _, val = _safe_split(df[df["split"] == "train"], share, seed)
        df.loc[val.index, "split"] = "val"
    return df


def check_group_leak(df: pd.DataFrame) -> list[str]:
    """Các doc_id xuất hiện ở nhiều hơn một tập (rỗng nghĩa là chia đúng)."""
    if GROUP_COL not in df.columns:
        return []
    n = df.groupby(GROUP_COL)["split"].nunique()
    return sorted(n[n > 1].index.astype(str))


def build_label_map(df: pd.DataFrame) -> dict[str, int]:
    """Gán số cho từng tác giả theo thứ tự chữ cái để ổn định giữa các lần chạy."""
    return {a: i for i, a in enumerate(sorted(df["author"].unique()))}
