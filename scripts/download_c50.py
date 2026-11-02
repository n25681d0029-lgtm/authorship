#!/usr/bin/env python3
"""Tải bộ Reuters C50 về data/raw/ và giải nén (cần Internet).

Chạy:  python scripts/download_c50.py
Nếu mạng chặn hoặc đường dẫn thay đổi, tải tay rồi giải nén sao cho có hai thư mục
    data/raw/C50/C50train/<tác giả>/*.txt
    data/raw/C50/C50test/<tác giả>/*.txt
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_DIR  # noqa: E402

URLS = [
    "https://archive.ics.uci.edu/static/public/217/reuter+50+50.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00217/C50.zip",
]


def main() -> int:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DATA_DIR / "C50"
    zip_path = RAW_DATA_DIR / "C50.zip"
    for url in URLS:
        try:
            print(f"Đang tải {url} ...")
            urllib.request.urlretrieve(url, zip_path)
            break
        except Exception as e:  # noqa: BLE001
            print(f"  không được: {e}")
    else:
        print("Không tải được. Hãy tải tay (UCI: 'Reuter_50_50') rồi giải nén vào", target)
        return 1
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target)
    zip_path.unlink()
    print("Đã giải nén vào", target)
    print("Kiểm tra cấu trúc rồi chạy:  python main.py train --data", target, "--fmt c50")
    return 0


if __name__ == "__main__":
    sys.exit(main())
