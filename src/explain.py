"""Chuyển các đặc trưng phong cách lệch nhiều nhất (z-score) thành câu tiếng Việt dễ đọc.

Ví dụ: {"feature": "fw[thì]", "value": 1.42, "z": 2.8}
    -> "Dùng từ “thì” nhiều hơn mức trung bình của kho huấn luyện 2,8 độ lệch chuẩn (1,42 lần/100 từ)."
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

BASE_NAMES = {
    "ttr": ("độ đa dạng từ vựng (tỷ lệ từ khác nhau)", ""),
    "hapax_ratio": ("tỷ lệ từ chỉ xuất hiện một lần", ""),
    "yule_k": ("mức lặp từ (chỉ số Yule’s K)", ""),
    "word_len_mean": ("độ dài trung bình của từ", " ký tự"),
    "word_len_std": ("độ chênh lệch độ dài giữa các từ", " ký tự"),
    "sent_len_mean": ("độ dài trung bình của câu", " từ/câu"),
    "sent_len_std": ("độ chênh lệch độ dài giữa các câu", " từ"),
    "upper_ratio": ("tỷ lệ chữ viết hoa", ""),
    "digit_ratio": ("tỷ lệ chữ số", ""),
}
PUNCT_NAMES = {",": "dấu phẩy", ".": "dấu chấm", ";": "dấu chấm phẩy", ":": "dấu hai chấm", "!": "dấu chấm than",
               "?": "dấu hỏi", '"': "dấu ngoặc kép thẳng", "'": "dấu nháy đơn", "(": "dấu mở ngoặc", ")": "dấu đóng ngoặc",
               "-": "dấu gạch nối", "–": "dấu gạch ngang ngắn", "—": "dấu gạch ngang (đầu dòng thoại, chèn ý)",
               "…": "dấu ba chấm", "“": "dấu mở ngoặc kép", "”": "dấu đóng ngoặc kép", "’": "dấu nháy cong"}


def _num(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def describe(ev: dict) -> str:
    """Một câu tiếng Việt cho một đặc trưng (đầu ra của StyleFeatureExtractor.explain)."""
    name, value, z = ev["feature"], ev["value"], ev["z"]
    more = "nhiều hơn" if z > 0 else "ít hơn"
    higher = "cao hơn" if z > 0 else "thấp hơn"
    sd = f"{_num(abs(z), 1)} độ lệch chuẩn"
    m = re.fullmatch(r"fw\[(.+)\]", name)
    if m:
        return f"Dùng từ “{m.group(1)}” {more} mức trung bình của kho huấn luyện {sd} ({_num(value)} lần/100 từ)."
    m = re.fullmatch(r"punct\[(.+)\]", name)
    if m:
        label = PUNCT_NAMES.get(m.group(1), f"dấu “{m.group(1)}”")
        return f"Dùng {label} {more} mức trung bình {sd} ({_num(value)} lần/1000 ký tự)."
    label, unit = BASE_NAMES.get(name, (name, ""))
    return f"{label[0].upper() + label[1:]} {higher} mức trung bình {sd} (giá trị {_num(value)}{unit})."


def feature_label(name: str) -> str:
    """Tên dễ đọc của một đặc trưng phong cách, kèm đơn vị: 'fw[thì]' -> 'từ “thì” (lần/100 từ)'."""
    m = re.fullmatch(r"fw\[(.+)\]", name)
    if m:
        return f"từ “{m.group(1)}” (lần/100 từ)"
    m = re.fullmatch(r"punct\[(.+)\]", name)
    if m:
        return f"{PUNCT_NAMES.get(m.group(1), 'dấu “' + m.group(1) + '”')} (lần/1000 ký tự)"
    m = re.fullmatch(r"ngram\[(.+)\]", name)
    if m:
        return f"xác suất n-gram ký tự cho {m.group(1)}"
    label, unit = BASE_NAMES.get(name, (name, ""))
    return label + (f" ({unit.strip()})" if unit else "")


def author_display_names(manifest: Path) -> dict[str, str]:
    """slug -> tên tác giả có dấu, lấy từ file CSV có cột author, author_name (data/v5_works.csv)."""
    names = {}
    if Path(manifest).exists():
        with open(manifest, encoding="utf-8-sig", newline="") as f:   # file Excel lưu có BOM
            for r in csv.DictReader(f):
                names.setdefault(r["author"], r["author_name"])
    return names
