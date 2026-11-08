"""Nạp checkpoint và dự đoán tác giả cho văn bản mới, kèm bằng chứng phong cách."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from config import CHECKPOINT_DIR, TrainConfig
from .calibration import softmax
from .dataset import Collator, make_loader
from .feature_extractor import StyleFeatureExtractor
from .models import HybridAuthorModel
from .utils import get_device, load_json

# Văn bản dài hơn đoạn huấn luyện (50-200 âm tiết) được cắt thành các đoạn cỡ đó rồi lấy trung bình xác suất,
# thay vì để PhoBERT chỉ đọc 256 token đầu còn đặc trưng phong cách lại tính trên toàn văn.
CHUNK_MIN_WORDS, CHUNK_MAX_WORDS, CHUNK_TARGET_WORDS = 50, 200, 150


def resolve_checkpoint(arg: str | Path) -> Path:
    """Chấp nhận đường dẫn đầy đủ hoặc chỉ tên thư mục trong outputs/checkpoints."""
    p = Path(arg).expanduser()
    if (p / "meta.json").exists():
        return p.resolve()
    q = CHECKPOINT_DIR / str(arg)
    if (q / "meta.json").exists():
        return q.resolve()
    raise FileNotFoundError(f"Không tìm thấy checkpoint '{arg}' (đã thử {p} và {q}).")


def split_long_text(text: str) -> list[str]:
    """Văn bản dài hơn CHUNK_MAX_WORDS âm tiết được cắt thành các đoạn cỡ dữ liệu huấn luyện."""
    from .corpus_builder import pack_passages, split_paragraphs
    from .text_utils import normalize

    if len(normalize(text).split()) <= CHUNK_MAX_WORDS:
        return [text]
    return pack_passages(split_paragraphs(text), CHUNK_MIN_WORDS, CHUNK_MAX_WORDS, CHUNK_TARGET_WORDS) or [text]


class AuthorPredictor:
    def __init__(self, model: HybridAuthorModel, cfg: TrainConfig, extractor: StyleFeatureExtractor,
                 label_map: dict[str, int], tokenizer, device, stacker=None, calibration: Optional[dict] = None) -> None:
        self.model = model.to(device).eval()
        self.cfg = cfg
        self.extractor = extractor
        self.stacker = stacker
        self.calibration = calibration or {}
        self.temperature = float(self.calibration.get("temperature", 1.0))
        # độ ưu tiên từng tác giả học trên val (0 với checkpoint không dùng --class-bias)
        self.class_bias = np.asarray(self.calibration.get("class_bias") or [0.0] * len(label_map), dtype=np.float64)
        self.names = [a for a, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
        self.collator = Collator(tokenizer, cfg.max_length, cfg.lang)
        self.device = device
        self.ensemble = None          # EnsembleHead (src/ensemble.py) nếu checkpoint có ensemble.joblib
        self.extra_models: list["AuthorPredictor"] = []

    @property
    def reject_threshold(self) -> Optional[float]:
        """Ngưỡng từ chối học trên tập val lúc huấn luyện (None với checkpoint cũ); có bộ gộp thì lấy ngưỡng của bộ gộp."""
        if self.ensemble is not None and self.ensemble.reject_threshold is not None:
            return self.ensemble.reject_threshold
        return self.calibration.get("reject_threshold")

    @property
    def ensemble_info(self) -> Optional[dict]:
        """Mô tả bộ gộp cho giao diện (None nếu chỉ một mô hình)."""
        if self.ensemble is None:
            return None
        return {"n_hybrid": 1 + len(self.extra_models), "weights": dict(self.ensemble.weights),
                "temperature": self.ensemble.temperature, "report": self.ensemble.report}

    @classmethod
    def from_checkpoint(cls, ckpt: str | Path, device: str = "auto", use_ensemble: bool = True) -> "AuthorPredictor":
        """use_ensemble: nếu thư mục có ensemble.joblib (main.py ensemble --save) thì nạp thêm các mô hình lai phụ và
        bộ n-gram, xác suất là kết hợp của tất cả. False: chỉ mô hình chính (như trước)."""
        ckpt = resolve_checkpoint(ckpt)
        meta = load_json(ckpt / "meta.json")
        cfg = TrainConfig.from_dict(meta["config"])
        extractor = StyleFeatureExtractor.load(ckpt / "features.json")
        stacker = None
        if (ckpt / "ngram.joblib").exists():
            from .ngram_features import CharNgramStacker

            stacker = CharNgramStacker.load(ckpt / "ngram.joblib")
        tokenizer, backbone_config = None, None
        if cfg.fusion != "style_only":
            from transformers import AutoConfig, AutoTokenizer

            backbone_config = AutoConfig.from_pretrained(ckpt / "backbone_config")
            tokenizer = AutoTokenizer.from_pretrained(ckpt / "tokenizer")
        model = HybridAuthorModel(
            num_labels=len(meta["label_map"]), style_dim=meta["style_dim"], fusion=cfg.fusion,
            backbone_config=backbone_config, freeze_backbone=cfg.freeze_backbone,
            style_hidden=cfg.style_hidden, style_out=cfg.style_out, fused_dim=cfg.fused_dim,
            dropout=cfg.dropout)
        state = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        obj = cls(model, cfg, extractor, meta["label_map"], tokenizer, get_device(device), stacker,
                  meta.get("calibration"))
        if use_ensemble and (ckpt / "ensemble.joblib").exists():
            from .ensemble import EnsembleHead

            obj.ensemble = EnsembleHead.load(ckpt / "ensemble.joblib")
            obj.extra_models = [cls.from_checkpoint(n, device, use_ensemble=False) for n in obj.ensemble.hybrids]
        return obj

    def style_features(self, texts: Sequence[str]) -> np.ndarray:
        """Vector đưa vào luồng phong cách: đặc trưng stylometry (+ xác suất n-gram ký tự nếu mô hình có dùng)."""
        style = self.extractor.transform(list(texts))
        if self.stacker is not None:
            style = np.hstack([style, self.stacker.transform(list(texts))]).astype(np.float32)
        return style

    @torch.no_grad()
    def predict_logits(self, texts: Sequence[str], batch_size: int = 16) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Logits (chưa hiệu chỉnh) và trọng số cổng trung bình (chỉ chế độ gated)."""
        from .text_utils import normalize

        # Lúc huấn luyện, load_corpus đã chuẩn hóa mọi đoạn; văn bản người dùng dán vào cũng phải qua cùng bước đó,
        # nếu không n-gram ký tự sẽ thấy « », "quốc-ngữ"... là những chuỗi chưa từng gặp khi học.
        texts = [normalize(t) for t in texts]
        loader = make_loader(texts, self.style_features(texts), [0] * len(texts), self.collator, batch_size,
                             shuffle=False)
        logits, gates = [], []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            z, gate = self.model(style=batch["style"], input_ids=batch.get("input_ids"),
                                 attention_mask=batch.get("attention_mask"), length=batch.get("length"),
                                 return_gate=True)
            logits.append(z.float().cpu().numpy())
            if gate is not None:
                gates.append(gate.mean(dim=-1).cpu().numpy())
        return np.concatenate(logits) + self.class_bias, (np.concatenate(gates) if gates else None)

    def predict_proba(self, texts: Sequence[str], batch_size: int = 16,
                      calibrated: bool = True) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Ma trận xác suất (số văn bản x số tác giả), đã chia logits cho nhiệt độ T nếu calibrated.
        Có bộ gộp: kết hợp mọi thành viên (EnsembleHead), calibrated=False bỏ qua nhiệt độ cuối."""
        if self.ensemble is not None:
            parts, gates = self.ensemble_parts(texts, batch_size)
            z = self.ensemble.combine_parts(parts.pop("hybrid"), parts)
            return softmax(z, self.ensemble.temperature if calibrated else 1.0), gates
        logits, gates = self.predict_logits(texts, batch_size)
        return softmax(logits, self.temperature if calibrated else 1.0), gates

    def ensemble_parts(self, texts: Sequence[str], batch_size: int = 16) -> tuple[dict, Optional[np.ndarray]]:
        """Log-xác suất từng thành viên bộ gộp: "hybrid" (trung bình các mô hình lai) và từng bộ n-gram."""
        from .ensemble import log_softmax
        from .text_utils import normalize

        logits, gates = self.predict_logits(texts, batch_size)
        lps = [log_softmax(logits, self.temperature)]
        lps += [log_softmax(m.predict_logits(texts, batch_size)[0], m.temperature) for m in self.extra_models]
        parts = {"hybrid": np.mean(lps, axis=0)}
        parts.update(self.ensemble.member_logps([normalize(t) for t in texts]))
        return parts, gates

    def predict(self, texts: Sequence[str], top_k: int = 3, explain: bool = True,
                reject_below: Optional[float] = None, batch_size: int = 16,
                candidates: Optional[Sequence[str]] = None) -> list[dict]:
        """Với mỗi văn bản trả về tác giả dự đoán, xác suất, top-k và các đặc trưng phong cách nổi bật.

        candidates: chỉ xét các tác giả này (ví dụ lọc theo thể loại); xác suất được chuẩn hóa lại trên nhóm đó.
        Nếu tác giả thật không nằm trong nhóm thì không bao giờ được chọn.

        Văn bản dài được cắt thành nhiều đoạn (split_long_text); xác suất là trung bình các đoạn.
        reject_below: nếu xác suất cao nhất thấp hơn ngưỡng này thì trả về "unknown"
        (cách xử lý đơn giản cho tình huống tập mở).
        """
        texts = list(texts)
        chunks, owner = [], []
        for i, t in enumerate(texts):
            parts = split_long_text(t)
            chunks += parts
            owner += [i] * len(parts)
        owner = np.asarray(owner)
        chunk_probs, chunk_gates = self.predict_proba(chunks, batch_size)
        if candidates:
            mask = np.isin(self.names, list(candidates)).astype(np.float64)
            if not mask.any():
                raise ValueError("Không có tác giả nào trong danh sách ứng viên thuộc mô hình.")
            chunk_probs = chunk_probs * mask
            chunk_probs = chunk_probs / chunk_probs.sum(axis=1, keepdims=True).clip(1e-12)

        results = []
        for i, text in enumerate(texts):
            idx = np.flatnonzero(owner == i)
            probs = chunk_probs[idx].mean(axis=0)
            order = [j for j in np.argsort(-probs) if not candidates or self.names[j] in candidates][:top_k]
            best = int(order[0])
            conf = float(probs[best])
            unknown = reject_below is not None and conf < reject_below
            item = {
                "author": "unknown" if unknown else self.names[best],
                "confidence": conf,
                "top_k": [{"author": self.names[j], "prob": float(probs[j])} for j in order],
                "n_chunks": int(len(idx)),
            }
            if len(idx) > 1:
                item["chunks"] = [{"n_words": len(chunks[j].split()), "author": self.names[int(chunk_probs[j].argmax())],
                                   "prob": float(chunk_probs[j].max())} for j in idx]
            if chunk_gates is not None:
                item["lm_weight"] = float(chunk_gates[idx].mean())  # g gần 1: tin luồng ngôn ngữ; gần 0: tin phong cách
            if explain:
                item["style_evidence"] = self.extractor.explain(text, top=6)
            results.append(item)
        return results


def evaluate_checkpoint(ckpt: str | Path, data_path: Optional[Path] = None, split: str = "test",
                        device: str = "auto") -> dict:
    """Đánh giá lại một checkpoint trên một tập (mặc định test).

    Dùng lại đúng cách chia dữ liệu lúc huấn luyện (cùng seed, cùng tỷ lệ) nên cho kết quả khớp.
    data_path cho phép trỏ tới dữ liệu ở vị trí khác khi chép checkpoint sang máy mới.
    """
    from .data_loader import load_corpus, make_splits
    from .evaluate import collect_logits, save_evaluation

    ckpt = resolve_checkpoint(ckpt)
    meta = load_json(ckpt / "meta.json")
    info = meta["data"]
    path = Path(data_path) if data_path else Path(info["path"])
    df = load_corpus(path, fmt=info["fmt"], text_col=info["text_col"], author_col=info["author_col"])
    cfg = TrainConfig.from_dict(meta["config"])
    df = make_splits(df, cfg.val_size, cfg.test_size, cfg.data_seed, info.get("test_topics"), cfg.topic_holdout or None)
    part = df[df["split"] == split]
    if part.empty:
        raise ValueError(f"Tập '{split}' rỗng.")
    pred = AuthorPredictor.from_checkpoint(ckpt, device)
    label_map = meta["label_map"]
    part = part[part["author"].isin(label_map)]
    loader = make_loader(part["text"].tolist(), pred.style_features(part["text"].tolist()),
                         part["author"].map(label_map).tolist(), pred.collator, cfg.batch_size, shuffle=False)
    y_true, logits = collect_logits(pred.model, loader, pred.device)
    return save_evaluation(y_true, (logits + pred.class_bias).argmax(axis=1), pred.names, ckpt / "reports",
                           f"eval_{split}")
