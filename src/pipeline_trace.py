"""Ghi lại từng bước hệ thống làm với MỘT văn bản để giao diện trực quan hóa (app.py, tab "Hệ thống hoạt động thế nào").

Các bước giống hệt AuthorPredictor.predict (test PipelineTraceTest kiểm tra xác suất trùng khớp):
  1. chuẩn hóa văn bản (normalize) — đếm từng loại thay đổi
  2. cắt văn bản dài thành đoạn cỡ dữ liệu huấn luyện (split_long_text)
  3. tách từ (pyvi) và mã hóa thành token cho PhoBERT
  4. đặc trưng phong cách: giá trị thô và z-score so với kho huấn luyện
  5. hai luồng mạng nơ-ron và phần đóng góp của mỗi luồng vào điểm từng tác giả. Lớp phân loại là tuyến tính nên
     điểm = W·(phần PhoBERT) + W·(phần phong cách) + hệ số tự do + độ ưu tiên tác giả, tách đúng tuyệt đối, không xấp xỉ.
  6. hiệu chỉnh nhiệt độ, softmax, trung bình các đoạn
  7. lọc theo thể loại / ứng viên, ngưỡng từ chối (decide)
  8. so sánh với hồ sơ phong cách trung bình của từng tác giả trong kho (author_style_profiles)
Không phụ thuộc Streamlit; mọi danh sách tác giả lấy từ checkpoint nên đổi số tác giả không phải sửa code.
"""
from __future__ import annotations

import unicodedata
from typing import Optional, Sequence

import numpy as np
import torch

from . import text_utils as tu
from .calibration import softmax
from .feature_extractor import Z_CLIP
from .predict import AuthorPredictor, split_long_text

MAX_SHOWN_TOKENS = 400


def normalization_steps(text: str) -> tuple[str, list[dict]]:
    """Chạy lại normalize() từng bước, đếm số chỗ mỗi quy tắc thay đổi. Kết quả cuối trùng normalize(text)."""
    steps = []

    def apply(name: str, why: str, before: str, after: str, count: int) -> str:
        if count:
            steps.append({"rule": name, "why": why, "count": int(count)})
        return after

    t = text
    n = sum(ord(c) in tu.HOMOGLYPHS for c in t)       # bảng str.maketrans có khóa là mã ký tự
    t = apply("Chữ nước ngoài trông giống chữ Việt, đổi về chữ thường", "lỗi khi quét ảnh hoặc gõ", t, t.translate(tu.HOMOGLYPHS), n)
    n = len(tu.INVISIBLE_RE.findall(t))
    t = apply("Xóa ký tự ẩn", "sót lại khi sao chép từ web hoặc PDF", t, tu.INVISIBLE_RE.sub("", t), n)
    nfc = unicodedata.normalize("NFC", t)
    t = apply("Thống nhất cách lưu chữ có dấu", "cùng một chữ có thể được máy lưu theo hai cách", t, nfc,
              int(nfc != t) and max(1, len(t) - len(nfc)))
    n = t.count("\r")
    t = apply("Thống nhất dấu xuống dòng", "mỗi hệ điều hành xuống dòng một kiểu", t,
              t.replace("\r\n", "\n").replace("\r", "\n"), n)
    n = len(tu.LINE_END_HYPHEN_RE.findall(t))
    t = apply("Nối lại chữ bị ngắt ở cuối dòng", "thường gặp ở bản chép từ sách in", t, tu.LINE_END_HYPHEN_RE.sub(" ", t), n)
    n = len(tu.INTRA_WORD_HYPHEN_RE.findall(t))
    t = apply("Bỏ gạch nối kiểu “văn-minh”", "cách viết cũ, do người in chọn chứ không phải tác giả", t,
              tu.INTRA_WORD_HYPHEN_RE.sub(" ", t), n)
    n = sum(m.group(0) != "—" for m in tu.DASH_RE.finditer(t))
    t = apply("Gộp các kiểu gạch ngang làm một", "mỗi bản chép dùng một kiểu", t, tu.DASH_RE.sub("—", t), n)
    n = t.count("«") + t.count("»")
    t = apply("Ngoặc « » đổi thành “ ”", "mỗi bản chép dùng một kiểu", t, t.replace("«", "“").replace("»", "”"), n)
    n = len(tu.SPACE_BEFORE_PUNCT_RE.findall(t)) + len(tu.SPACE_AFTER_OPEN_RE.findall(t))
    t = apply("Bỏ dấu cách thừa trước dấu câu", "lối in cũ theo kiểu Pháp", t,
              tu.SPACE_AFTER_OPEN_RE.sub("", tu.SPACE_BEFORE_PUNCT_RE.sub("", t)), n)
    return t, steps


@torch.no_grad()
def _forward_parts(pred: AuthorPredictor, norm_text: str) -> dict:
    """Một lần chạy mô hình trên một đoạn, giữ lại các vector trung gian và phần đóng góp của từng luồng."""
    model, cfg = pred.model, pred.cfg
    style_np = pred.style_features([norm_text])
    batch = pred.collator([{"text": norm_text, "style": torch.as_tensor(style_np[0]), "label": torch.tensor(0)}])
    batch = {k: v.to(pred.device) for k, v in batch.items()}
    e_lm = e_sty = gate = None
    if model.backbone is not None:
        e_lm = model.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state[:, 0]
    if model.style_enc is not None:
        e_sty = model.style_enc(batch["style"])
    if cfg.fusion == "concat":
        parts = {"lm": torch.cat([e_lm, torch.zeros_like(e_sty)], -1), "style": torch.cat([torch.zeros_like(e_lm), e_sty], -1)}
    elif cfg.fusion == "gated":
        g_in = torch.cat([e_lm, e_sty, batch["length"].unsqueeze(-1)], dim=-1)
        gate = torch.sigmoid(model.gate(g_in))
        parts = {"lm": gate * model.proj_lm(e_lm), "style": (1.0 - gate) * model.proj_sty(e_sty)}
    elif cfg.fusion == "lm_only":
        parts = {"lm": e_lm}
    else:
        parts = {"style": e_sty}
    W, b = model.classifier.weight, model.classifier.bias
    contrib = {k: (v @ W.T)[0].float().cpu().numpy() for k, v in parts.items()}
    out = {"contrib": contrib, "intercept": b.float().cpu().numpy(),
           "style_input_dim": int(style_np.shape[1]),
           "lm_dim": int(e_lm.shape[-1]) if e_lm is not None else 0,
           "style_out_dim": int(e_sty.shape[-1]) if e_sty is not None else 0,
           "fused_dim": int(next(iter(parts.values())).shape[-1])}
    if gate is not None:
        out["gate_mean"] = float(gate.mean())
    return out


def _tokens(pred: AuthorPredictor, norm_text: str) -> Optional[dict]:
    tok = pred.collator.tokenizer
    if tok is None:
        return None
    seg = tu.prepare_for_lm(norm_text, pred.cfg.lang)
    ids = tok(seg, truncation=False)["input_ids"]
    return {"segmented": seg, "n_tokens": len(ids), "max_length": int(pred.cfg.max_length),
            "truncated": len(ids) > pred.cfg.max_length,
            "tokens": tok.convert_ids_to_tokens(ids[:min(MAX_SHOWN_TOKENS, pred.cfg.max_length)])}


def trace(pred: AuthorPredictor, text: str) -> dict:
    """Mọi bước trung gian cho một văn bản (bước 1–6). Bước 7 (lọc, ngưỡng) làm ở decide() vì phụ thuộc lựa chọn
    của người dùng và không cần chạy lại mô hình."""
    norm, steps = normalization_steps(text)
    assert norm == tu.normalize(text)
    words = norm.split()
    raw_chunks = split_long_text(text)
    names = pred.names
    chunks = []
    for c in raw_chunks:
        nc = tu.normalize(c)
        raw = pred.extractor._raw(nc)
        z = np.clip((raw - pred.extractor.mean) / pred.extractor.std, -Z_CLIP, Z_CLIP)
        fp = _forward_parts(pred, nc)
        logit_model = sum(fp["contrib"].values()) + fp["intercept"]
        logits = logit_model + pred.class_bias
        item = {
            "text": nc, "n_words": len(nc.split()), "tokens": _tokens(pred, nc),
            "style_raw": raw, "style_z": z, **fp,
            "logits": logits, "probs_uncalibrated": softmax(logits[None], 1.0)[0],
            "probs": softmax(logits[None], pred.temperature)[0],
        }
        if pred.ensemble is not None:
            # bộ gộp: xác suất cuối kết hợp mô hình chính (bước 5) với các mô hình phụ và bộ n-gram
            parts, _ = pred.ensemble_parts([nc])
            item["probs_main"] = item["probs"]
            item["members"] = {k: softmax(v, 1.0)[0] for k, v in parts.items()}
            z = pred.ensemble.combine_parts(parts.pop("hybrid"), parts)
            item["probs"] = softmax(z, pred.ensemble.temperature)[0]
        chunks.append(item)
    return {
        "names": list(names), "fusion": pred.cfg.fusion, "char_ngram": pred.stacker is not None,
        "backbone": pred.cfg.backbone if pred.cfg.fusion != "style_only" else None,
        "temperature": pred.temperature, "class_bias": np.asarray(pred.class_bias, dtype=np.float64),
        "feature_names": pred.extractor.feature_names, "corpus_mean": pred.extractor.mean,
        "corpus_std": pred.extractor.std,
        "raw_text": text, "normalized": norm, "norm_steps": steps,
        "n_chars": len(norm), "n_words": len(words), "n_sentences": len(tu.split_sentences(norm)),
        "invalid_ratio": tu.invalid_syllable_ratio(text),
        "chunks": chunks, "probs": np.mean([c["probs"] for c in chunks], axis=0),
        "ensemble": pred.ensemble_info,
    }


def decide(probs: np.ndarray, names: Sequence[str], candidates: Optional[Sequence[str]] = None,
           threshold: Optional[float] = None) -> dict:
    """Bước 7, giống AuthorPredictor.predict: giữ xác suất của ứng viên, chuẩn hóa lại cho tổng = 1, chọn cao nhất,
    dưới ngưỡng thì 'unknown'. Trả về cả xác suất trước và sau lọc để so sánh."""
    names = list(names)
    probs = np.asarray(probs, dtype=np.float64)
    keep = np.ones(len(names), bool) if not candidates else np.isin(names, list(candidates))
    if not keep.any():
        raise ValueError("Không có tác giả ứng viên nào thuộc mô hình.")
    after = probs * keep
    after = after / after.sum().clip(1e-12)
    best = int(np.argmax(np.where(keep, after, -1.0)))
    conf = float(after[best])
    unknown = threshold is not None and threshold > 0 and conf < threshold
    return {"keep": keep, "probs_before": probs, "probs_after": after, "best": names[best], "confidence": conf,
            "author": "unknown" if unknown else names[best], "unknown": bool(unknown),
            "best_without_filter": names[int(np.argmax(probs))]}


def author_style_profiles(extractor, texts: Sequence[str], authors: Sequence[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Hồ sơ phong cách mỗi tác giả = trung bình giá trị thô các đặc trưng trên các đoạn của tác giả đó.
    Trả về (tên tác giả, ma trận trung bình thô [tác giả x đặc trưng], số đoạn mỗi tác giả)."""
    authors = np.asarray(list(authors))
    raw = np.stack([extractor._raw(t) for t in texts])
    names = sorted(set(authors.tolist()))
    means = np.stack([raw[authors == a].mean(axis=0) for a in names])
    counts = np.asarray([int((authors == a).sum()) for a in names])
    return names, means, counts


def style_distance(text_z: np.ndarray, profile_raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Khoảng cách phong cách kiểu Burrows' Delta: trung bình |z văn bản − z hồ sơ| trên mọi đặc trưng (nhỏ = giống)."""
    prof_z = np.clip((profile_raw - mean) / std, -Z_CLIP, Z_CLIP)
    return np.abs(prof_z - text_z[None, :]).mean(axis=1)
