"""Cấu hình trung tâm của dự án.

Mọi đường dẫn đều được tính từ vị trí của chính file này nên có thể chép cả
thư mục sang máy khác (Windows, Linux, macOS) mà không phải sửa gì.
Muốn đặt dữ liệu và kết quả ở nơi khác, đặt biến môi trường AA_HOME.
"""
from __future__ import annotations

import os
from typing import Optional
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# Thư mục gốc của dự án (cha của thư mục config). AA_HOME ghi đè nếu có.
_DEFAULT_BASE = Path(__file__).resolve().parent.parent
BASE_DIR: Path = Path(os.environ.get("AA_HOME", _DEFAULT_BASE)).expanduser().resolve()

DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
OUTPUT_DIR: Path = BASE_DIR / "outputs"
CHECKPOINT_DIR: Path = OUTPUT_DIR / "checkpoints"
REPORT_DIR: Path = OUTPUT_DIR / "reports"
EXPERIMENT_DIR: Path = OUTPUT_DIR / "experiments"

# Backbone mặc định theo ngôn ngữ. Có thể thay bằng tên khác trên Hugging Face
# hoặc đường dẫn tới thư mục mô hình đã tải sẵn (chạy offline).
DEFAULT_BACKBONES = {
    "en": "roberta-base",
    "vi": "vinai/phobert-base-v2",
}

FUSION_MODES = ("concat", "gated", "lm_only", "style_only")


def ensure_dirs() -> None:
    """Tạo các thư mục làm việc nếu chưa có. Gọi ở main.py, không gọi lúc import."""
    for p in (RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR, REPORT_DIR, EXPERIMENT_DIR):
        p.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainConfig:
    """Siêu tham số huấn luyện. Được lưu cùng checkpoint để dự đoán lại đúng cấu hình."""

    lang: str = "en"
    backbone: str = ""            # rỗng nghĩa là lấy theo DEFAULT_BACKBONES[lang]
    fusion: str = "concat"        # concat | gated | lm_only | style_only
    max_length: int = 256
    batch_size: int = 16
    epochs: int = 5
    lr_backbone: float = 2e-5
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.3
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    freeze_backbone: bool = False
    num_function_words: int = 150
    style_hidden: int = 256
    style_out: int = 128
    fused_dim: int = 256          # chiều không gian chung cho gated fusion
    patience: int = 2             # số epoch chờ trước khi dừng sớm
    seed: int = 42
    val_size: float = 0.15
    test_size: float = 0.15
    topic_holdout: float = 0.0    # > 0: mỗi tác giả giữ lại các tác phẩm chiếm tỷ lệ này làm test (chéo chủ đề)
    char_ngram: bool = False      # ghép xác suất của bộ phân loại n-gram ký tự (out-of-fold) vào luồng phong cách
    accept_rate: float = 0.95     # ngưỡng từ chối chọn sao cho tỷ lệ này số đoạn val được chấp nhận
    class_weight: bool = False    # trọng số lớp tỷ lệ nghịch với số đoạn train (khi số đoạn giữa tác giả chênh lệch)
    oversample: bool = False      # rút đoạn train theo xác suất tỷ lệ nghịch số đoạn của tác giả (cân bằng mỗi epoch)
    class_bias: bool = False      # học độ ưu tiên cộng vào logit từng tác giả trên tập val (tối đa macro-F1)
    num_workers: int = 0
    split_seed: Optional[int] = None   # seed riêng cho cách chia train/val/test (None = dùng seed); khác seed mô hình
                                       # để huấn luyện nhiều mô hình trên CÙNG một cách chia rồi gộp (ensemble)

    @property
    def data_seed(self) -> int:
        return self.seed if self.split_seed is None else int(self.split_seed)

    def __post_init__(self) -> None:
        if self.lang not in DEFAULT_BACKBONES:
            raise ValueError(f"lang phải là một trong {list(DEFAULT_BACKBONES)}")
        if self.fusion not in FUSION_MODES:
            raise ValueError(f"fusion phải là một trong {FUSION_MODES}")
        if not self.backbone:
            self.backbone = DEFAULT_BACKBONES[self.lang]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
