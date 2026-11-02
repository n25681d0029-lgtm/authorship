"""Các hàm tiện ích dùng chung: seed, thiết bị, log, đọc ghi JSON, thanh tiến trình."""
from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LOGGER_NAME = "aa"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    """Cố định ngẫu nhiên để chạy lại cho cùng kết quả (trong giới hạn của phần cứng)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(prefer: str = "auto"):
    """Chọn thiết bị: cuda, mps (Mac) hoặc cpu. prefer có thể là 'auto', 'cpu', 'cuda', 'mps'."""
    import torch

    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def progress(iterable: Iterable, **kwargs):
    """Dùng tqdm nếu có, nếu không thì trả về iterable nguyên vẹn."""
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def count_parameters(model) -> tuple[int, int]:
    """Trả về (tổng số tham số, số tham số được huấn luyện)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
