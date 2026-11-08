"""Câu nào làm máy nghĩ văn bản là của tác giả X? — phương pháp che câu (occlusion).

Bỏ lần lượt từng câu, đoán lại; câu nào bỏ đi làm xác suất của X giảm nhiều thì câu đó "giống X" nhiều nhất.
Chính xác với mọi mô hình (kể cả bộ gộp) vì chỉ gọi predict, không xấp xỉ gradient. Chi phí: (số câu + 1) lần đoán.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .text_utils import normalize, split_sentences

MAX_SENTENCES = 40


def sentence_influence(pred, text: str, author: str, candidates: Optional[Sequence[str]] = None) -> Optional[dict]:
    """Trả về {"base": xác suất của author với văn bản đủ, "sentences": [(câu, ảnh hưởng)], "scale": giá trị lớn nhất}.
    ảnh hưởng = xác suất lúc đủ − xác suất khi bỏ câu đó (dương: câu kéo về phía author). None nếu quá ít / quá nhiều câu."""
    sents = [s for s in split_sentences(normalize(text)) if s.strip()]
    if len(sents) < 2 or len(sents) > MAX_SENTENCES:
        return None
    variants = [" ".join(sents)] + [" ".join(sents[:i] + sents[i + 1:]) for i in range(len(sents))]
    res = pred.predict(variants, top_k=len(pred.names), explain=False, candidates=candidates)
    probs = [next((t["prob"] for t in r["top_k"] if t["author"] == author), 0.0) for r in res]
    base = probs[0]
    infl = [base - p for p in probs[1:]]
    scale = max(1e-9, max(abs(v) for v in infl))
    return {"base": base, "sentences": list(zip(sents, infl)), "scale": scale}
