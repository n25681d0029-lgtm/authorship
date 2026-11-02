"""Dựng kho đoạn văn (CSV) từ thư mục văn bản thô của nhiều tác giả.

Cấu trúc đầu vào (tự nhận diện):
    <src>/<tác giả>/<tác phẩm>/*.txt     topic = tác phẩm, mỗi file là một "bài" (chương, truyện ngắn)
    <src>/<tác giả>/*.txt                topic = tên file, mỗi file là một bài
Nếu có <src>/manifest.csv (do scripts/fetch_wikisource.py tạo) thì lấy thêm URL nguồn.

Quy trình:
  1. Tách mỗi bài thành các đoạn văn tự nhiên (cách nhau bởi dòng trống).
  2. Bỏ đoạn văn trùng lặp trong cùng một tác giả (tập truyện hay chép lại truyện lẻ).
  3. Ghép các đoạn văn liền nhau thành "đoạn mẫu" dài min_words..max_words từ; đoạn văn quá dài
     được cắt theo ranh giới câu. Không ghép xuyên bài, nên mỗi đoạn mẫu thuộc đúng một bài (doc_id).
     Trang quá dài (Wikisource có trang chứa nhiều chương) được chia thành các phần liên tiếp tối đa
     max_doc_passages đoạn, mỗi phần là một doc_id, để việc chia train/test theo bài không bị kẹt
     vì một tác giả chỉ có vài trang rất dài.
  4. Cân bằng: mỗi tác giả lấy tối đa per_author đoạn, chia đều cho các tác phẩm để kho không bị
     một cuốn tiểu thuyết dài chi phối; tác giả có ít hơn min_per_author đoạn bị loại.

Đầu ra có các cột: text, author, topic, doc_id, n_words, source.
Số "từ" ở đây là số token tách theo khoảng trắng (với tiếng Việt là số âm tiết).
"""
from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

from .data_loader import read_text_file
from .text_utils import normalize, split_sentences


def split_paragraphs(text: str) -> list[str]:
    text = normalize(text).strip()
    blocks = [b.strip() for b in text.split("\n\n")]
    if len(blocks) <= 1:
        blocks = [b.strip() for b in text.split("\n")]
    return [" ".join(b.split()) for b in blocks if b.strip()]


def _cut_long(paragraph: str, max_words: int) -> list[str]:
    """Cắt đoạn văn dài hơn max_words theo ranh giới câu; câu nào vẫn quá dài thì cắt theo từ."""
    pieces, buf, n = [], [], 0
    for sent in split_sentences(paragraph):
        words = sent.split()
        while len(words) > max_words:
            if buf:
                pieces.append(" ".join(buf))
                buf, n = [], 0
            pieces.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if n + len(words) > max_words and buf:
            pieces.append(" ".join(buf))
            buf, n = [], 0
        buf += words
        n += len(words)
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def pack_passages(paragraphs: list[str], min_words: int = 50, max_words: int = 200,
                  target_words: int = 150) -> list[str]:
    """Ghép đoạn văn liền nhau thành đoạn mẫu dài trong [min_words, max_words].

    Đủ target_words thì chốt đoạn; giữ ranh giới đoạn văn bằng ký tự xuống dòng.
    Phần thừa ngắn hơn min_words (thường là vài câu thoại) bị bỏ.
    """
    units = []
    for p in paragraphs:
        units += [p] if len(p.split()) <= max_words else _cut_long(p, max_words)
    out, buf, n = [], [], 0
    for u in units:
        k = len(u.split())
        if buf and n + k > max_words:
            if n >= min_words:
                out.append("\n".join(buf))
            buf, n = [], 0
        buf.append(u)
        n += k
        if n >= target_words:
            out.append("\n".join(buf))
            buf, n = [], 0
    if buf and n >= min_words:
        out.append("\n".join(buf))
    return out


def _allocate(counts: dict[str, int], cap: int) -> dict[str, int]:
    """Chia cap đoạn cho các tác phẩm càng đều càng tốt (tác phẩm ít đoạn thì nhường phần dư)."""
    alloc = {t: 0 for t in counts}
    active = [t for t in sorted(counts) if counts[t] > 0]
    remaining = cap
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        for t in list(active):
            give = min(share, counts[t] - alloc[t], remaining)
            alloc[t] += give
            remaining -= give
            if alloc[t] >= counts[t]:
                active.remove(t)
            if remaining == 0:
                break
    return alloc


def _read_manifest(src: Path) -> dict[str, str]:
    p = src / "manifest.csv"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8", newline="") as f:
        return {r["file"]: r["url"] for r in csv.DictReader(f)}


def _iter_docs(src: Path):
    """Sinh (tác giả, tác phẩm, doc_id, đường dẫn) cho cả hai kiểu cấu trúc thư mục."""
    for adir in sorted(p for p in src.iterdir() if p.is_dir()):
        for f in sorted(adir.glob("*.txt")):
            yield adir.name, f.stem, f"{adir.name}/{f.stem}", f
        for wdir in sorted(p for p in adir.iterdir() if p.is_dir()):
            for f in sorted(wdir.glob("*.txt")):
                yield adir.name, wdir.name, f"{adir.name}/{wdir.name}/{f.stem}", f


def build_corpus(src: Path | str, min_words: int = 50, max_words: int = 200, target_words: int = 150,
                 per_author: int = 300, min_per_author: int = 100, seed: int = 42,
                 authors: Optional[list[str]] = None, max_doc_passages: int = 20) -> tuple[pd.DataFrame, dict]:
    """Trả về (DataFrame đoạn mẫu, thống kê). Xem docstring của module."""
    src = Path(src)
    if not src.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục {src}")
    urls = _read_manifest(src)
    rows: dict[str, list[dict]] = defaultdict(list)
    seen: dict[str, set] = defaultdict(set)
    dup = 0
    for author, topic, doc_id, path in _iter_docs(src):
        if authors and author not in authors:
            continue
        paras = []
        for p in split_paragraphs(read_text_file(path)):
            h = hashlib.md5(p.lower().encode("utf-8")).hexdigest()
            if h in seen[author]:
                dup += 1
                continue
            seen[author].add(h)
            paras.append(p)
        rel = str(path.relative_to(src)).replace("\\", "/")
        passages = pack_passages(paras, min_words, max_words, target_words)
        for i, text in enumerate(passages):
            # trang quá dài (cả cuốn hoặc nhiều chương trên một trang) được chia thành phần liên tiếp cỡ một chương
            doc = doc_id if len(passages) <= max_doc_passages else f"{doc_id}#{i // max_doc_passages}"
            rows[author].append({"text": text, "author": author, "topic": topic, "doc_id": doc,
                                 "n_words": len(text.split()), "source": urls.get(rel, rel)})

    kept, dropped, available = [], {}, {}
    for author in sorted(rows):
        # mỗi tác giả một bộ sinh riêng: thêm dữ liệu cho một tác giả không làm đổi đoạn được chọn của tác giả khác
        rng = random.Random(f"{seed}-{author}")
        items = rows[author]
        available[author] = len(items)
        if len(items) < min_per_author:
            dropped[author] = len(items)
            continue
        by_topic = defaultdict(list)
        for it in items:
            by_topic[it["topic"]].append(it)
        alloc = _allocate({t: len(v) for t, v in by_topic.items()}, min(per_author, len(items)))
        for t, v in sorted(by_topic.items()):
            kept += rng.sample(v, alloc[t])
    if not kept:
        raise ValueError("Không còn tác giả nào đủ số đoạn tối thiểu.")
    df = pd.DataFrame(kept).sort_values(["author", "doc_id"]).reset_index(drop=True)
    return df, _stats(df, dup, available, dropped, {
        "min_words": min_words, "max_words": max_words, "target_words": target_words, "per_author": per_author,
        "min_per_author": min_per_author, "seed": seed, "max_doc_passages": max_doc_passages})


def _stats(df: pd.DataFrame, dup: int, available: dict, dropped: dict, params: dict) -> dict:
    return {
        "n_passages": int(len(df)), "n_authors": int(df["author"].nunique()),
        "duplicate_paragraphs_removed": dup, "available_per_author": available, "dropped_authors": dropped,
        "per_author": df.groupby("author").agg(passages=("text", "size"), topics=("topic", "nunique"),
                                               docs=("doc_id", "nunique"), words_mean=("n_words", "mean"),
                                               words_min=("n_words", "min"), words_max=("n_words", "max"))
                        .round(1).reset_index().to_dict(orient="records"),
        "params": params,
    }


# ====================================================================== kho theo thể loại (v5)
POEM_SEPARATOR = "***"   # dòng chỉ chứa *** ngăn các bài thơ khác nhau trong cùng một file


def split_poems(text: str) -> list[list[str]]:
    """Tách nội dung file thơ thành các bài; mỗi bài là danh sách dòng thơ (đã bỏ dòng trống giữa các khổ).

    Khổ thơ không được giữ: cách chia khổ khác nhau giữa các bản chép, giữ lại thì mô hình học nguồn chép."""
    text = normalize(text)
    poems, cur = [], []
    for raw in text.split("\n"):
        line = " ".join(raw.split())
        if line == POEM_SEPARATOR:
            if cur:
                poems.append(cur)
            cur = []
        elif line:
            cur.append(line)
    if cur:
        poems.append(cur)
    return poems


def pack_poem_lines(poems: list[list[str]], min_words: int = 40, max_words: int = 160,
                    target_words: int = 120, labels: Optional[list] = None) -> list:
    """Ghép dòng thơ thành đoạn mẫu [min_words, max_words] âm tiết, giữ ký tự xuống dòng giữa các dòng.

    Bài dài được cắt thành nhiều đoạn; bài ngắn (tứ tuyệt 28 âm tiết) được ghép với bài kế tiếp trong danh sách
    để đủ độ dài. Dòng dài hơn max_words (phú, thơ văn xuôi) được cắt theo từ.
    labels: nhãn của từng bài (vd. tác phẩm); nếu có thì trả về [(đoạn, nhãn của bài đầu tiên trong đoạn)]."""
    out, buf, n, first = [], [], 0, None

    def flush():
        nonlocal buf, n, first
        if buf and n >= min_words:
            out.append(("\n".join(buf), first))
        buf, n, first = [], 0, None

    for k, poem in enumerate(poems):
        lab = labels[k] if labels is not None else None
        for line in poem:
            words = line.split()
            while len(words) > max_words:
                flush()
                out.append((" ".join(words[:max_words]), lab))
                words = words[max_words:]
            if not words:
                continue
            if buf and n + len(words) > max_words:
                flush()
            if not buf:
                first = lab
            buf.append(" ".join(words))
            n += len(words)
        if n >= target_words:            # hết bài và đã đủ dài: chốt, không ghép sang bài sau
            flush()
    flush()
    return out if labels is not None else [t for t, _ in out]


def load_genres(path: Path | str) -> dict:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_manifest_v5(root: Path) -> dict[str, str]:
    p = root / "manifest.csv"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8", newline="") as f:
        return {r["file"]: r["url"] for r in csv.DictReader(f)}


def build_genre_corpus(root: Path | str, genres_cfg: dict, min_words: int = 50, max_words: int = 200,
                       target_words: int = 150, poem_min_words: int = 40, poem_max_words: int = 160,
                       poem_target_words: int = 120, per_author: int = 200,
                       min_per_author: Optional[dict[str, int]] = None, min_works: Optional[int] = None,
                       min_authors: Optional[int] = None, max_invalid_ratio: float = 0.03,
                       max_doc_passages: int = 20, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Dựng kho đoạn mẫu từ <root>/<thể loại>/<tác giả>/<tác phẩm>/*.txt (do scripts/fetch_v5.py tạo).

    Khác build_corpus:
      - có cột genre; cân bằng và ngưỡng tính theo từng cặp (thể loại, tác giả);
      - thơ cắt theo dòng thơ (pack_poem_lines), doc_id là cụm liên tiếp trong một tác phẩm (bài thơ quá ngắn
        để làm nhóm riêng); văn xuôi giữ cách cắt cũ;
      - bỏ đoạn có tỷ lệ âm tiết sai cấu trúc > max_invalid_ratio (bản chép OCR chưa soát);
      - cặp (thể loại, tác giả) thiếu số đoạn hoặc số tác phẩm bị loại; thể loại còn ít hơn min_authors tác giả
        bị loại cả thể loại (không đủ để định danh có ý nghĩa).
    Cột tác giả giữ slug chung giữa các thể loại để so sánh được mô hình riêng và mô hình chung.
    """
    from .text_utils import invalid_syllable_ratio

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục {root}")
    thr = genres_cfg.get("nguong_toi_thieu", {})
    min_per_author = {**thr.get("so_doan_toi_thieu_moi_tac_gia", {}), **(min_per_author or {})}
    min_works = thr.get("so_tac_pham_toi_thieu_moi_tac_gia", 2) if min_works is None else min_works
    min_authors = thr.get("so_tac_gia_toi_thieu_moi_the_loai", 5) if min_authors is None else min_authors
    known = list(genres_cfg["the_loai"])
    urls = _read_manifest_v5(root)

    rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    seen: dict[tuple[str, str], set] = defaultdict(set)
    dup, noisy = 0, defaultdict(int)
    poem_queue: dict[tuple[str, str], list] = defaultdict(list)   # thơ gom theo (thể loại, tác giả), xử lý sau

    def _add(key, text, topic, doc_id, src):
        bad = invalid_syllable_ratio(text)
        if bad > max_invalid_ratio:
            noisy[f"{key[0]}/{key[1]}"] += 1
            return
        rows[key].append({"text": text, "author": key[1], "genre": key[0], "topic": topic, "doc_id": doc_id,
                          "n_words": len(text.split()), "source": src, "invalid_ratio": round(bad, 4)})

    for gdir in sorted(p for p in root.iterdir() if p.is_dir()):
        genre = gdir.name
        if genre not in known:
            continue
        for adir in sorted(p for p in gdir.iterdir() if p.is_dir()):
            author, key = adir.name, (genre, adir.name)
            for wdir in sorted(p for p in adir.iterdir() if p.is_dir()):
                files = sorted(wdir.glob("*.txt"))
                if genre == "tho":
                    for f in files:
                        for poem in split_poems(read_text_file(f)):
                            h = hashlib.md5("\n".join(poem).lower().encode("utf-8")).hexdigest()
                            if h in seen[key]:
                                dup += 1
                                continue
                            seen[key].add(h)
                            rel = str(f.relative_to(root)).replace("\\", "/")
                            poem_queue[key].append((poem, wdir.name, urls.get(rel, rel)))
                    continue
                items = []
                for f in files:
                    paras = []
                    for p in split_paragraphs(read_text_file(f)):
                        h = hashlib.md5(p.lower().encode("utf-8")).hexdigest()
                        if h in seen[key]:
                            dup += 1
                            continue
                        seen[key].add(h)
                        paras.append(p)
                    passages = pack_passages(paras, min_words, max_words, target_words)
                    doc_id = f"{genre}/{author}/{wdir.name}/{f.stem}"
                    rel = str(f.relative_to(root)).replace("\\", "/")
                    items += [(t, doc_id if len(passages) <= max_doc_passages
                               else f"{doc_id}#{i // max_doc_passages}", urls.get(rel, rel))
                              for i, t in enumerate(passages)]
                for text, doc_id, src in items:
                    _add(key, text, wdir.name, doc_id, src)

    # Thơ: bài ngắn (thường mỗi bài một trang) được ghép liên tiếp theo thứ tự tác phẩm của cùng tác giả.
    # doc_id = cụm max_doc_passages đoạn liên tiếp, nên các đoạn ghép từ những bài liền nhau luôn chung một tập.
    for key, queue in sorted(poem_queue.items()):
        packed = pack_poem_lines([q[0] for q in queue], poem_min_words, poem_max_words, poem_target_words,
                                 labels=list(range(len(queue))))
        for i, (text, k) in enumerate(packed):
            _add(key, text, queue[k][1], f"{key[0]}/{key[1]}#{i // max_doc_passages}", queue[k][2])

    kept, dropped, available = [], {}, {}
    for (genre, author) in sorted(rows):
        items = rows[(genre, author)]
        name = f"{genre}/{author}"
        n_works = len({it["topic"] for it in items})
        available[name] = {"passages": len(items), "works": n_works}
        need = int(min_per_author.get(genre, 40))
        if len(items) < need or n_works < min_works:
            dropped[name] = f"{len(items)} đoạn, {n_works} tác phẩm (cần ≥{need} đoạn, ≥{min_works} tác phẩm)"
            continue
        rng = random.Random(f"{seed}-{genre}-{author}")
        by_topic = defaultdict(list)
        for it in items:
            by_topic[it["topic"]].append(it)
        alloc = _allocate({t: len(v) for t, v in by_topic.items()}, min(per_author, len(items)))
        for t, v in sorted(by_topic.items()):
            kept += rng.sample(v, alloc[t])
    if not kept:
        raise ValueError("Không còn cặp (thể loại, tác giả) nào đủ ngưỡng.")
    df = pd.DataFrame(kept)
    n_auth = df.groupby("genre")["author"].nunique()
    small = sorted(g for g, n in n_auth.items() if n < min_authors)
    dropped_genres = {g: f"chỉ {int(n_auth[g])} tác giả (cần ≥{min_authors})" for g in small}
    df = df[~df["genre"].isin(small)].sort_values(["genre", "author", "doc_id"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"Không thể loại nào đủ {min_authors} tác giả: {dropped_genres}")

    per_genre = {}
    for genre, part in df.groupby("genre"):
        per_genre[genre] = {
            "n_passages": int(len(part)), "n_authors": int(part["author"].nunique()),
            "per_author": part.groupby("author").agg(passages=("text", "size"), topics=("topic", "nunique"),
                                                     docs=("doc_id", "nunique"), words_mean=("n_words", "mean"),
                                                     invalid_mean=("invalid_ratio", "mean"))
                              .round(3).reset_index().to_dict(orient="records")}
    stats = {
        "n_passages": int(len(df)), "n_genres": int(df["genre"].nunique()), "n_authors": int(df["author"].nunique()),
        "authors_in_several_genres": sorted(a for a, n in df.groupby("author")["genre"].nunique().items() if n > 1),
        "duplicate_units_removed": dup, "noisy_passages_removed": dict(noisy),
        "available": available, "dropped_pairs": dropped, "dropped_genres": dropped_genres, "per_genre": per_genre,
        "params": {"min_words": min_words, "max_words": max_words, "target_words": target_words,
                   "poem_min_words": poem_min_words, "poem_max_words": poem_max_words,
                   "poem_target_words": poem_target_words, "per_author": per_author,
                   "min_per_author": min_per_author, "min_works": min_works, "min_authors": min_authors,
                   "max_invalid_ratio": max_invalid_ratio, "max_doc_passages": max_doc_passages, "seed": seed},
    }
    return df, stats
