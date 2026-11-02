"""Sinh kho văn bản NHÂN TẠO để chạy thử toàn bộ hệ thống mà không cần tải dữ liệu thật.

Mỗi "tác giả" giả lập có thói quen riêng: từ chức năng ưa dùng, độ dài câu, tần suất dấu phẩy,
dấu chấm phẩy, gạch ngang, dấu chấm than. Chủ đề được gán ngẫu nhiên cho từng văn bản
(không gắn với tác giả), nên có thể thử luôn kịch bản đổi chủ đề giữa train và test.

Lưu ý: đây chỉ là dữ liệu kiểm tra đường ống. Độ chính xác trên dữ liệu này không nói lên
điều gì về chất lượng trên văn bản thật (C50, Enron...).
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

FUNCTION_WORDS = {
    "en": "the of and to a in that is was it for with as his her he she on by at but not this which from or "
          "have had they were are be an we you would there their what so if about who when".split(),
    "vi": "và của là có những một các trong cho được với này khi để đã không người rằng nhưng cũng thì mà "
          "như còn lại sẽ đó nên vì rất hay từ đến ra vào bởi theo sau trước đều chỉ cả".split(),
}

TOPICS = {
    "en": {
        "technology": "software network server algorithm data processor memory cloud device signal code "
                      "system engineer circuit battery protocol platform sensor".split(),
        "literature": "novel poem chapter hero village morning shadow letter story voice garden silence "
                      "stranger memory journey river window".split(),
        "economy": "market price trade bank budget growth export tax profit shares inflation worker "
                   "factory demand supply investor".split(),
        "sport": "match team coach goal season player league stadium defence score victory injury "
                 "tournament referee training captain".split(),
    },
    "vi": {
        "congnghe": "phần_mềm mạng máy_chủ thuật_toán dữ_liệu bộ_nhớ điện_toán thiết_bị tín_hiệu mã_nguồn "
                    "hệ_thống kỹ_sư mạch pin giao_thức nền_tảng cảm_biến".split(),
        "vanhoc": "tiểu_thuyết bài_thơ chương nhân_vật ngôi_làng buổi_sáng bóng_tối lá_thư câu_chuyện "
                  "giọng_nói khu_vườn im_lặng người_lạ ký_ức dòng_sông".split(),
        "kinhte": "thị_trường giá_cả thương_mại ngân_hàng ngân_sách tăng_trưởng xuất_khẩu thuế lợi_nhuận "
                  "cổ_phiếu lạm_phát công_nhân nhà_máy nhu_cầu nhà_đầu_tư".split(),
        "thethao": "trận_đấu đội_bóng huấn_luyện_viên bàn_thắng mùa_giải cầu_thủ giải_đấu sân_vận_động "
                   "hàng_thủ tỷ_số chiến_thắng chấn_thương trọng_tài tập_luyện đội_trưởng".split(),
    },
}


def _author_profile(rng: random.Random, fw: list[str]) -> dict:
    weights = [rng.random() ** 2 + 0.02 for _ in fw]           # vài từ được ưa dùng hơn hẳn
    return {
        "fw_weights": weights,
        "content_ratio": rng.uniform(0.30, 0.50),
        "sent_mean": rng.uniform(8, 24),
        "p_comma": rng.uniform(0.02, 0.18),
        "p_semicolon": rng.choice([0.0, 0.0, 0.01, 0.03]),
        "p_dash": rng.choice([0.0, 0.0, 0.01, 0.02]),
        "end_marks": rng.choice([(".", ".", ".", "!"), (".", ".", "?", "?"), (".", ".", ".", ".")]),
    }


def _make_sentence(rng: random.Random, prof: dict, fw: list[str], content: list[str]) -> str:
    n = max(4, int(rng.gauss(prof["sent_mean"], 3)))
    words: list[str] = []
    for i in range(n):
        if rng.random() < prof["content_ratio"]:
            w = rng.choice(content)
        else:
            w = rng.choices(fw, weights=prof["fw_weights"], k=1)[0]
        if i < n - 1:
            r = rng.random()
            if r < prof["p_comma"]:
                w += ","
            elif r < prof["p_comma"] + prof["p_semicolon"]:
                w += ";"
            elif r < prof["p_comma"] + prof["p_semicolon"] + prof["p_dash"]:
                w += " -"
        words.append(w)
    sentence = " ".join(words).replace("_", " ")
    sentence = sentence[0].upper() + sentence[1:]
    return sentence + rng.choice(prof["end_marks"])


def generate_demo_corpus(out_dir: Path, lang: str = "en", n_authors: int = 5, docs_per_author: int = 40,
                         min_sentences: int = 6, max_sentences: int = 14, seed: int = 7,
                         holdout_topic: bool = False) -> Path:
    """Ghi file CSV demo_<lang>.csv vào out_dir và trả về đường dẫn.

    holdout_topic=True: thêm cột split, chủ đề cuối cùng chỉ xuất hiện ở test (kịch bản đổi chủ đề).
    """
    if lang not in FUNCTION_WORDS:
        raise ValueError(f"lang phải là một trong {list(FUNCTION_WORDS)}")
    rng = random.Random(seed)
    fw = FUNCTION_WORDS[lang]
    topics = TOPICS[lang]
    topic_names = list(topics)
    rows = []
    for a in range(n_authors):
        prof = _author_profile(rng, fw)
        for d in range(docs_per_author):
            topic = rng.choice(topic_names)
            n_sent = rng.randint(min_sentences, max_sentences)
            text = " ".join(_make_sentence(rng, prof, fw, topics[topic]) for _ in range(n_sent))
            row = {"author": f"tacgia_{a + 1:02d}", "topic": topic, "text": text}
            if holdout_topic:
                row["split"] = "test" if topic == topic_names[-1] else "train"
            rows.append(row)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"demo_{lang}.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return path
