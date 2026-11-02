#!/usr/bin/env python3
"""Đồng bộ dữ liệu đã tải (data/v5_genre/raw) với nhãn hiện tại trong data/v5_works.csv.

Khi đổi nhãn một tác phẩm SAU lúc đã chạy `fetch_v5.py fetch` (vd. phát hiện là bản dịch), file đã tải vẫn nằm trong
raw/<thể loại cũ>/ và build-genre-corpus sẽ đọc nó. Script này bỏ các dòng manifest (và file .txt tương ứng) mà cặp
(thể loại, tác giả, tác phẩm) không còn là nhãn hợp lệ. Nguồn `local` (văn bản tự đặt vào) không bị đụng tới.

Chạy: python scripts/sync_v5_manifest.py            (xem trước)
      python scripts/sync_v5_manifest.py --apply    (thực hiện)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_v5 import GENRES, MANIFEST_FIELDS, RAW, WORKS_CSV, read_csv, write_csv  # noqa: E402


def stale_entries(manifest: list[dict], works: list[dict]) -> list[dict]:
    """Các dòng manifest không còn khớp nhãn hợp lệ (genre ∈ 4 thể loại, ok = 1) của đúng tác phẩm đó."""
    valid = {(w["genre"], w["author"], w["work"]) for w in works
             if w.get("genre") in GENRES and str(w.get("ok")) == "1"}
    return [m for m in manifest if m.get("source") != "local" and (m["genre"], m["author"], m["work"]) not in valid]


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    manifest = read_csv(RAW / "manifest.csv")
    stale = stale_entries(manifest, read_csv(WORKS_CSV))
    for m in stale:
        print(f"{'XÓA' if args.apply else 'sẽ xóa'}: {m['genre']}/{m['author']} | {m['work']} | {m['file']} ({m['n_words']} âm tiết)")
    if args.apply and stale:
        drop = {id(m) for m in stale}
        for m in stale:
            (RAW / m["file"]).unlink(missing_ok=True)
        write_csv(RAW / "manifest.csv", [m for m in manifest if id(m) not in drop], MANIFEST_FIELDS)
    print(f"{len(stale)} dòng không còn khớp nhãn" + ("" if args.apply else " (thêm --apply để xóa)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
