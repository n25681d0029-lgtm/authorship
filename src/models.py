"""Kiến trúc mạng lai: mô hình ngôn ngữ (ngữ cảnh) + MLP phong cách, hợp nhất rồi phân loại.

Bốn chế độ (tham số fusion) để làm thí nghiệm bỏ từng phần (ablation):
  concat      : ghép E_LM và E_sty rồi phân loại.
  gated       : học cổng g để trộn hai luồng theo từng chiều.
  lm_only     : chỉ dùng mô hình ngôn ngữ (baseline PLM thuần).
  style_only  : chỉ dùng MLP phong cách, không tải mô hình ngôn ngữ (chạy nhanh trên CPU).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StyleEncoder(nn.Module):
    """Vector phong cách đã chuẩn hóa (K chiều) -> vector biểu diễn (out_dim chiều)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridAuthorModel(nn.Module):
    def __init__(self, num_labels: int, style_dim: int, fusion: str = "concat",
                 backbone_name: str | None = None, backbone_config=None,
                 freeze_backbone: bool = False, style_hidden: int = 256, style_out: int = 128,
                 fused_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        if fusion not in ("concat", "gated", "lm_only", "style_only"):
            raise ValueError(f"fusion không hợp lệ: {fusion}")
        self.fusion = fusion
        self.freeze_backbone = freeze_backbone and fusion != "style_only"

        # ---- luồng 1: mô hình ngôn ngữ
        self.backbone = None
        lm_dim = 0
        if fusion != "style_only":
            from transformers import AutoConfig, AutoModel

            if backbone_name is not None:                    # tải trọng số tiền huấn luyện
                self.backbone = AutoModel.from_pretrained(backbone_name)
            elif backbone_config is not None:                # chỉ dựng khung, trọng số sẽ nạp từ checkpoint
                self.backbone = AutoModel.from_config(backbone_config)
            else:
                raise ValueError("Cần backbone_name hoặc backbone_config.")
            lm_dim = int(self.backbone.config.hidden_size)
            if self.freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False

        # ---- luồng 2: phong cách
        self.style_enc = None
        if fusion != "lm_only":
            self.style_enc = StyleEncoder(style_dim, style_hidden, style_out, dropout)

        # ---- hợp nhất
        if fusion == "concat":
            out_dim = lm_dim + style_out
        elif fusion == "gated":
            self.proj_lm = nn.Linear(lm_dim, fused_dim)
            self.proj_sty = nn.Linear(style_out, fused_dim)
            self.gate = nn.Linear(lm_dim + style_out + 1, fused_dim)   # +1: độ dài tương đối của văn bản
            out_dim = fused_dim
        elif fusion == "lm_only":
            out_dim = lm_dim
        else:
            out_dim = style_out
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, num_labels)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone and self.backbone is not None:
            self.backbone.eval()          # backbone đóng băng thì cũng tắt dropout của nó
        return self

    def _encode_lm(self, input_ids, attention_mask) -> torch.Tensor:
        ctx = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with ctx:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]       # vector của token đầu chuỗi ([CLS] hoặc <s>)

    def forward(self, style: torch.Tensor, input_ids=None, attention_mask=None,
                length: torch.Tensor | None = None, return_gate: bool = False):
        e_lm = self._encode_lm(input_ids, attention_mask) if self.backbone is not None else None
        e_sty = self.style_enc(style) if self.style_enc is not None else None
        gate = None

        if self.fusion == "concat":
            fused = torch.cat([e_lm, e_sty], dim=-1)
        elif self.fusion == "gated":
            if length is None:
                length = torch.ones(style.size(0), device=style.device)
            g_in = torch.cat([e_lm, e_sty, length.unsqueeze(-1)], dim=-1)
            gate = torch.sigmoid(self.gate(g_in))
            fused = gate * self.proj_lm(e_lm) + (1.0 - gate) * self.proj_sty(e_sty)
        elif self.fusion == "lm_only":
            fused = e_lm
        else:
            fused = e_sty

        logits = self.classifier(self.dropout(fused))
        return (logits, gate) if return_gate else logits
