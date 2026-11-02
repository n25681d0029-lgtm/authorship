#!/usr/bin/env python3
"""Hỗ trợ duyệt data/v5_works.csv: tính thêm đặc điểm của từng trang từ bộ đệm API của bước survey (không gọi mạng
nếu đã có đệm) để phát hiện thơ chữ Hán và bản dịch mà header không ghi (phiên âm Hán Việt, dịch nghĩa, dịch thơ
của người khác). Ghi data/v5_genre/review_features.csv; KHÔNG sửa v5_works.csv.

Chạy: python scripts/review_v5_features.py
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_v5 import V5, WORKS_CSV, Api, extract_content, read_csv  # noqa: E402

CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
MARKERS = ["phiên âm", "dịch nghĩa", "dịch thơ", "nguyên văn chữ hán", "bản dịch", "người dịch", "dịch giả",
           "nguyên tác", "chữ hán", "chữ nôm"]


def features(html: str) -> dict:
    from bs4 import BeautifulSoup

    text = BeautifulSoup(html, "html.parser").get_text(" ")
    low = text.lower()
    c = extract_content(html)
    return {"cjk": len(CJK_RE.findall(text)),
            "markers": ";".join(m for m in MARKERS if m in low),
            "words_poems": sum(len(x.split()) for x in c["poems"]),
            "words_paras": sum(len(x.split()) for x in c["paras"]),
            "first": (c["poems"][0] if c["poems"] else (c["paras"][0] if c["paras"] else ""))[:120].replace("\n", " / ")}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    api = Api(sleep=1.5)
    rows = read_csv(WORKS_CSV)
    out = []
    for i, r in enumerate(rows):
        if r["source"] != "ws":
            continue
        p = api.parse(r["work"])
        if p is None:
            continue
        f = features(p["html"])
        out.append({"author": r["author"], "work": r["work"], "de_xuat": r["de_xuat"], "genre": r["genre"],
                    "ok": r["ok"], "poem_ratio": r["poem_ratio"], "n_subpages": r["n_subpages"], **f})
        if i % 200 == 0:
            print(i, flush=True)
    path = V5 / "review_features.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("ghi", path, len(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
