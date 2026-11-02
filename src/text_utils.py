"""Xử lý văn bản cơ bản: chuẩn hóa Unicode, tách từ, tách câu, tách từ tiếng Việt (tùy chọn)."""
from __future__ import annotations

import functools
import re
import unicodedata
import warnings

# Từ gồm chữ/số (Unicode, nên giữ nguyên chữ có dấu tiếng Việt), cho phép dấu nháy trong từ: don't
TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)
# Dấu kết câu theo sau bởi khoảng trắng
SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
# Gạch nối giữa hai chữ cái (chính tả cũ "quốc-ngữ"); không đụng tới số như 1934-1935
INTRA_WORD_HYPHEN_RE = re.compile(r"(?<=[^\W\d_])-(?=[^\W\d_])")
# Gạch dùng làm dấu câu (đầu dòng thoại, chèn ý): "-" đứng riêng, "–", "—" đều đưa về "—"
DASH_RE = re.compile(r"(?:(?<=\s)|^)-(?=\s)|[–—]", re.MULTILINE)
# Gạch nối ở cuối dòng của bản chép từ sách in ("sai-\nbiệt"): chữ tiếng Việt viết theo âm tiết nên chỗ ngắt
# luôn là ranh giới âm tiết, thay bằng dấu cách như gạch nối giữa âm tiết
LINE_END_HYPHEN_RE = re.compile(r"(?<=[^\W\d_])-[ \t]*\n[ \t]*(?=[^\W\d_])")
# Ký tự vô hình hay lẫn vào văn bản sao chép từ PDF/web: zero-width, soft hyphen, BOM
INVISIBLE_RE = re.compile("[​‌‍⁠­﻿]")
# Chữ Kirin/Hy Lạp trông giống hệt chữ Latin (OCR và bộ gõ hay sinh ra); để nguyên thì n-gram ký tự coi là chữ khác
HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "Х": "X", "І": "I", "ο": "o", "Ο": "O", "α": "a", "ν": "v", "Α": "A", "Β": "B", "Ε": "E", "Ι": "I",
    "Κ": "K", "Μ": "M", "Ν": "N", "Ρ": "P", "Τ": "T", "Χ": "X", "Υ": "Y", "Ζ": "Z",
})

# Dấu cách trước dấu câu / sau ngoặc mở theo lối in kiểu Pháp ("lối : Một", "chăng ?", "“ văn ”"): có bản chép giữ,
# có bản bỏ, tần suất khác nhau giữa các nguồn (0 đến > 1 lần/100 âm tiết trong kho v5) nên là dấu vết của nguồn chép
SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t\u00a0\u202f]+(?=[:;?!,.)\]”…])")
SPACE_AFTER_OPEN_RE = re.compile(r"(?<=[(\[“])[ \t\u00a0\u202f]+")

_warned_vi_segmenter = False


def normalize(text: str) -> str:
    """Chuẩn hóa NFC và các quy ước trình bày của bản chép.

    NFC quan trọng với tiếng Việt: cùng một chữ có thể được mã hóa dạng tổ hợp hoặc dựng sẵn.
    Các quy ước trình bày (gạch nối giữa âm tiết kiểu "văn-minh", loại gạch đầu dòng thoại, ngoặc « »)
    do bản in và người số hóa quyết định chứ không phải thói quen của tác giả; nếu giữ nguyên thì mô
    hình học nhận ra nguồn chép thay vì phong cách, nên đưa về một dạng chung. Tương tự với dấu cách trước
    dấu câu kiểu Pháp và các dấu vết
    của khâu số hóa: ký tự vô hình, chữ Kirin giả chữ Latin, gạch nối khi ngắt dòng."""
    text = INVISIBLE_RE.sub("", text.translate(HOMOGLYPHS))
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = LINE_END_HYPHEN_RE.sub(" ", text)
    text = INTRA_WORD_HYPHEN_RE.sub(" ", text)
    text = DASH_RE.sub("—", text)
    text = text.replace("«", "“").replace("»", "”")
    return SPACE_AFTER_OPEN_RE.sub("", SPACE_BEFORE_PUNCT_RE.sub("", text))


# ---------------------------------------------------------------- kiểm tra âm tiết (lọc lỗi OCR)
_TONE_MARKS = {"̀", "́", "̃", "̉", "̣"}   # huyền, sắc, ngã, hỏi, nặng
_ONSET = r"(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|b|c|d|đ|g|h|k|l|m|n|p|r|s|t|v|x)?"
# Vần mở (không có âm cuối) và nhân vần có thể đi với âm cuối, theo chính tả quốc ngữ (gồm cả lối viết cũ)
_OPEN_RIMES = ("a ă â e ê i o ô ơ u ư y ai ao au ay âu ây eo êu ia iu oa oe oi ôi ơi ua uâ uê ui uơ uy ưa ưi ưu "
               "iêu yêu oai oay oeo uây uôi ươi ươu uya uyu uyê").split()
_CLOSED_NUCLEI = "a ă â e ê i o ô ơ u ư y iê yê uô ươ oa oă oe uâ uê uy uyê oo".split()
_CODA = r"(?:ch|ng|nh|c|m|n|p|t)"
_SYLLABLE_RE = re.compile(
    "^" + _ONSET + "(?:(?:" + "|".join(sorted(_OPEN_RIMES, key=len, reverse=True)) + ")"
    + "|(?:" + "|".join(sorted(_CLOSED_NUCLEI, key=len, reverse=True)) + ")" + _CODA + ")$")
_ALPHA_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def is_valid_syllable(token: str) -> bool:
    """Âm tiết có đúng cấu trúc chính tả tiếng Việt không (âm đầu + vần + tối đa một dấu thanh).

    Chỉ kiểm tra HÌNH THỨC: "đữ" (OCR sai của "đã") vẫn hợp lệ, còn "nưưc", "liọc", chữ lẫn ký tự Kirin thì không.
    Tên riêng, từ nước ngoài (Voltaire) cũng bị tính là không hợp lệ, nên chỉ dùng như tỷ lệ trên cả đoạn."""
    decomposed = unicodedata.normalize("NFD", token.lower())
    if sum(c in _TONE_MARKS for c in decomposed) > 1:
        return False
    base = unicodedata.normalize("NFC", "".join(c for c in decomposed if c not in _TONE_MARKS))
    return bool(_SYLLABLE_RE.match(base))


def invalid_syllable_ratio(text: str) -> float:
    """Tỷ lệ token chữ cái không phải âm tiết tiếng Việt hợp lệ (0 nếu không có token nào).
    Văn bản đánh máy cẩn thận thường dưới 2–3% (phần còn lại là tên riêng, từ nước ngoài); bản OCR chưa soát cao hơn nhiều."""
    tokens = _ALPHA_TOKEN_RE.findall(normalize(text))
    if not tokens:
        return 0.0
    return sum(not is_valid_syllable(t) for t in tokens) / len(tokens)


def tokenize_words(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def split_sentences(text: str) -> list[str]:
    parts = SENT_SPLIT_RE.split(text.strip())
    return [p for p in (s.strip() for s in parts) if p]


@functools.lru_cache(maxsize=100_000)     # mỗi epoch gặp lại cùng văn bản, không cần tách từ lại
def segment_vietnamese(text: str) -> str:
    """PhoBERT được huấn luyện trên văn bản đã tách từ (học_sinh, đại_học).
    Nếu máy có thư viện pyvi hoặc underthesea thì dùng, nếu không thì giữ nguyên văn bản."""
    global _warned_vi_segmenter
    try:
        from pyvi import ViTokenizer  # type: ignore

        return ViTokenizer.tokenize(text)
    except ImportError:
        pass
    try:
        from underthesea import word_tokenize  # type: ignore

        return word_tokenize(text, format="text")
    except ImportError:
        pass
    if not _warned_vi_segmenter:
        warnings.warn(
            "Chưa cài pyvi hoặc underthesea nên văn bản tiếng Việt không được tách từ trước khi đưa "
            "vào PhoBERT. Mô hình vẫn chạy nhưng nên cài: pip install pyvi",
            stacklevel=2,
        )
        _warned_vi_segmenter = True
    return text


def prepare_for_lm(text: str, lang: str) -> str:
    """Chuẩn bị văn bản trước khi đưa vào tokenizer của mô hình ngôn ngữ."""
    text = normalize(text)
    if lang == "vi":
        text = segment_vietnamese(text)
    return text
