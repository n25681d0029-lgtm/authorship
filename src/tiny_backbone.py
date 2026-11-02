"""Tạo một mô hình BERT NHỎ, khởi tạo ngẫu nhiên, hoàn toàn offline.

Mục đích: kiểm tra đường ống (token hóa, huấn luyện, lưu, nạp, dự đoán) khi không có mạng
hoặc không muốn tải roberta-base. Mô hình này KHÔNG được tiền huấn luyện nên không có
giá trị nghiên cứu, chỉ dùng để chạy thử nhanh trên CPU.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


def build_tiny_backbone(texts: Iterable[str], out_dir: Path, vocab_size: int = 2000,
                        hidden_size: int = 64, layers: int = 2, heads: int = 2) -> Path:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=specials)
    tok.train_from_iterator(list(texts), trainer)
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))],
    )
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    fast.save_pretrained(out_dir)

    cfg = BertConfig(vocab_size=len(fast), hidden_size=hidden_size, num_hidden_layers=layers,
                     num_attention_heads=heads, intermediate_size=hidden_size * 2,
                     max_position_embeddings=514, pad_token_id=fast.pad_token_id)
    BertModel(cfg).save_pretrained(out_dir)
    return out_dir
