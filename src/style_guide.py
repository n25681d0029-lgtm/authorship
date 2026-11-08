"""Hướng dẫn viết theo văn phong một tác giả — tính hoàn toàn bằng thống kê trên kho dữ liệu (không dùng mô hình
sinh văn, không gọi dịch vụ ngoài).

  build_guides(df, extractor) -> {(thể loại, tác giả): AuthorGuide}
    · từ / cụm hai chữ đặc trưng: log-odds có tiên nghiệm Dirichlet (Monroe, Colaresi & Quinn 2008) so với các tác giả
      CÙNG thể loại; bỏ tên riêng (chữ hầu như luôn viết hoa)
    · cách mở câu đặc trưng (hai chữ đầu câu), từ nối hay dùng
    · số đo câu: độ dài câu, tỷ lệ câu hỏi / cảm thán, lời thoại, số câu mỗi đoạn; thơ: số chữ mỗi dòng, thể thơ
    · thói quen phong cách lệch nhiều nhất so với trung bình thể loại (đặc trưng của StyleFeatureExtractor)
  check_draft(...)  : so bài người dùng viết với hồ sơ tác giả → điểm giống + gợi ý sửa cụ thể
  outline(...)      : dàn ý gợi ý cho một đề bài, theo thể loại và thói quen của tác giả
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .text_utils import normalize, split_sentences

SYL = re.compile(r"[^\W\d_]+", re.U)
DIALOGUE = re.compile(r"^\s*(—|-|–|“|\")", re.M)
CONNECTIVES = ["nhưng", "mà", "thì", "vì", "nên", "cho nên", "song", "tuy", "dẫu", "vậy", "vậy thì", "bởi vậy",
               "bởi thế", "thế mà", "thế nhưng", "rồi", "còn", "lại", "chớ", "đặng", "hễ", "hèn chi", "chẳng những",
               "không những", "huống chi", "ấy là", "nghĩa là", "tóm lại", "xem thế", "than ôi", "ôi"]


def syllables(text: str) -> list[str]:
    return SYL.findall(text)


@dataclass
class AuthorGuide:
    author: str
    genre: str
    n_passages: int
    n_words: int
    signature_words: list = field(default_factory=list)     # [(từ, z, số lần/10.000 chữ của tác giả, của thể loại)]
    avoided_words: list = field(default_factory=list)       # từ thể loại hay dùng mà tác giả hầu như tránh
    openers: list = field(default_factory=list)             # [(hai chữ đầu câu, tỷ lệ % câu)]
    connectives: list = field(default_factory=list)         # [(từ nối, lần/1000 chữ, so với thể loại)]
    stats: dict = field(default_factory=dict)               # số đo câu/đoạn của tác giả
    genre_stats: dict = field(default_factory=dict)         # số đo trung bình thể loại
    style_gaps: list = field(default_factory=list)          # [(tên đặc trưng, giá trị tác giả, trung bình thể loại, z)]
    examples: list = field(default_factory=list)            # vài đoạn mẫu ngắn
    vocab: Counter = field(default_factory=Counter)         # tần số âm tiết (chữ thường) của tác giả


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in syllables(text)]


def _ngrams(toks: list[str]) -> list[str]:
    return toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]


def text_stats(text: str, genre: str = "") -> dict:
    """Số đo đơn giản của một văn bản (cũng dùng cho bài người dùng)."""
    norm = normalize(text)
    sents = [s for s in split_sentences(norm) if syllables(s)]
    n = max(1, len(sents))
    lens = [len(syllables(s)) for s in sents] or [0]
    paras = [p for p in re.split(r"\n\s*\n|\n", norm) if p.strip()]
    lines = [len(syllables(l)) for l in norm.split("\n") if syllables(l)]
    out = {
        "so_chu": len(syllables(norm)),
        "do_dai_cau": float(np.mean(lens)),
        "cau_hoi": 100 * sum(s.rstrip().endswith("?") for s in sents) / n,
        "cau_cam_than": 100 * sum(s.rstrip().endswith("!") for s in sents) / n,
        "phay_moi_cau": norm.count(",") / n,
    }
    if genre != "tho":
        out["loi_thoai"] = 100 * len(DIALOGUE.findall(norm)) / max(1, len(paras))
        out["cau_moi_doan"] = n / max(1, len(paras))
    if genre == "tho":
        c = Counter(lines)
        out["chu_moi_dong"] = float(np.mean(lines)) if lines else 0.0
        out["the_tho"] = poem_form(lines)
        out["dong_pho_bien"] = c.most_common(3)
    return out


def poem_form(line_lengths: Sequence[int]) -> str:
    """Đoán thể thơ từ số chữ mỗi dòng: lục bát (6-8 xen kẽ), song thất lục bát, ngũ ngôn, thất ngôn, tám chữ, tự do."""
    ls = [x for x in line_lengths if x > 0]
    if len(ls) < 2:
        return "chưa đủ dòng"
    pairs = sum(1 for a, b in zip(ls, ls[1:]) if (a, b) == (6, 8))
    if pairs >= 0.35 * len(ls):
        sevens = sum(x == 7 for x in ls)
        return "song thất lục bát" if sevens >= 0.3 * len(ls) else "lục bát"
    c = Counter(ls)
    top, cnt = c.most_common(1)[0]
    if cnt >= 0.6 * len(ls):
        return {4: "bốn chữ", 5: "năm chữ (ngũ ngôn)", 6: "sáu chữ", 7: "bảy chữ (thất ngôn)", 8: "tám chữ"}.get(
            top, f"{top} chữ")
    return "tự do (số chữ mỗi dòng thay đổi)"


STAT_NAMES = {
    "do_dai_cau": ("Độ dài câu", "chữ/câu"),
    "cau_hoi": ("Câu hỏi", "% số câu"),
    "cau_cam_than": ("Câu cảm thán", "% số câu"),
    "phay_moi_cau": ("Dấu phẩy", "mỗi câu"),
    "loi_thoai": ("Đoạn là lời thoại", "% số đoạn"),
    "cau_moi_doan": ("Số câu mỗi đoạn", "câu"),
    "chu_moi_dong": ("Số chữ mỗi dòng thơ", "chữ"),
}


def _log_odds(a: Counter, b: Counter, prior: Counter, alpha0: float) -> dict[str, float]:
    """z-score log-odds có tiên nghiệm Dirichlet: dương = tác giả (a) dùng nhiều hơn phần còn lại (b)."""
    na, nb = sum(a.values()), sum(b.values())
    np_ = sum(prior.values())
    out = {}
    for w, pw in prior.items():
        aw = alpha0 * pw / np_
        ya, yb = a.get(w, 0), b.get(w, 0)
        la = math.log((ya + aw) / (na + alpha0 - ya - aw))
        lb = math.log((yb + aw) / (nb + alpha0 - yb - aw))
        var = 1.0 / (ya + aw) + 1.0 / (yb + aw)
        out[w] = (la - lb) / math.sqrt(var)
    return out


def build_guides(texts: Sequence[str], authors: Sequence[str], genres: Sequence[str], extractor=None,
                 min_count: int = 5, top: int = 30) -> dict:
    """Hồ sơ hướng dẫn cho mọi cặp (thể loại, tác giả) trong kho."""
    texts, authors, genres = list(texts), list(authors), list(genres)
    raw = np.stack([extractor._raw(normalize(t)) for t in texts]) if extractor is not None else None
    guides = {}
    for g in sorted(set(genres)):
        idx_g = [i for i, x in enumerate(genres) if x == g]
        counts, caps, mid, opens, stats = {}, Counter(), Counter(), {}, {}
        for i in idx_g:
            a = authors[i]
            t = normalize(texts[i])
            words = syllables(t)
            toks = [w.lower() for w in words]
            counts.setdefault(a, Counter()).update(_ngrams(toks))
            oc = opens.setdefault(a, Counter())
            for s in split_sentences(t):
                raw_w = syllables(s)
                # chữ đầu câu luôn viết hoa nên chỉ xét các vị trí giữa câu để nhận ra tên riêng
                mid.update(w.lower() for w in raw_w[1:])
                caps.update(w.lower() for w in raw_w[1:] if w[:1].isupper())
                sw = [w.lower() for w in raw_w]
                if len(sw) >= 3:
                    oc[" ".join(sw[:2])] += 1
            stats.setdefault(a, []).append(text_stats(t, g))
        total = sum(counts.values(), Counter())
        uni_total = Counter({w: c for w, c in total.items() if " " not in w})
        # tên riêng: chữ viết hoa ở > 60% số lần xuất hiện (không tính đầu câu thì vẫn cao với tên người, địa danh)
        proper = {w for w, c in caps.items() if c >= 0.6 * mid.get(w, 1)}
        genre_stats = _mean_stats([s for a in stats for s in stats[a]])
        for a, ca in counts.items():
            rest = total - ca
            prior = Counter({w: c for w, c in total.items() if c >= min_count
                             and not any(p in proper for p in w.split())})
            lo = _log_odds(ca, rest, prior, alpha0=1000.0)
            na, ng = sum(v for w, v in ca.items() if " " not in w), sum(uni_total.values())
            per = lambda c, n: 10000.0 * c / max(1, n)
            sig = [(w, z, per(ca.get(w, 0), na), per(total.get(w, 0), ng)) for w, z in
                   sorted(lo.items(), key=lambda kv: -kv[1]) if ca.get(w, 0) >= min_count][:top]
            avoid = [(w, z, per(ca.get(w, 0), na), per(total.get(w, 0), ng)) for w, z in
                     sorted(lo.items(), key=lambda kv: kv[1]) if total.get(w, 0) >= 5 * min_count][:15]
            n_sent = sum(opens[a].values())
            op_rest = sum((opens[b] for b in opens if b != a), Counter())
            op_lo = _log_odds(opens[a], op_rest, Counter({k: v for k, v in (opens[a] + op_rest).items() if v >= 3}), 200.0)
            op = [(k, 100.0 * opens[a][k] / max(1, n_sent)) for k, z in sorted(op_lo.items(), key=lambda kv: -kv[1])
                  if opens[a][k] >= 3 and not any(p in proper for p in k.split())][:10]
            conn = []
            for c in CONNECTIVES:
                ra, rg = 1000.0 * ca.get(c, 0) / max(1, na), 1000.0 * total.get(c, 0) / max(1, ng)
                if ca.get(c, 0) >= 3:
                    conn.append((c, ra, rg))
            conn.sort(key=lambda x: -(x[1] / max(x[2], 1e-6)))
            idx_a = [i for i in idx_g if authors[i] == a]
            gaps = []
            if raw is not None:
                mu_g, sd_g = raw[idx_g].mean(0), raw[idx_g].std(0) + 1e-9
                mu_a = raw[idx_a].mean(0)
                z = (mu_a - mu_g) / sd_g
                for j in np.argsort(-np.abs(z))[:10]:
                    gaps.append((extractor.feature_names[j], float(mu_a[j]), float(mu_g[j]), float(z[j])))
            ex = sorted(idx_a, key=lambda i: abs(len(texts[i].split()) - 110))[:3]
            guides[(g, a)] = AuthorGuide(
                author=a, genre=g, n_passages=len(idx_a), n_words=na, signature_words=sig, avoided_words=avoid,
                openers=op, connectives=conn[:10], stats=_mean_stats(stats[a]), genre_stats=genre_stats,
                style_gaps=gaps, examples=[texts[i] for i in ex],
                vocab=Counter({w: c for w, c in ca.items() if " " not in w}))
    return guides


def _mean_stats(rows: list[dict]) -> dict:
    out = {}
    for k in STAT_NAMES:
        v = [r[k] for r in rows if k in r]
        if v:
            out[k] = float(np.mean(v))
    forms = Counter(r["the_tho"] for r in rows if "the_tho" in r and r["the_tho"] != "chưa đủ dòng")
    if forms:
        n = sum(forms.values())
        out["the_tho"] = [(f, 100.0 * c / n) for f, c in forms.most_common(3)]
    return out


# ---------------------------------------------------------------------- kiểm tra bài viết
def check_draft(draft: str, guide: AuthorGuide, extractor=None) -> dict:
    """So bài viết với hồ sơ tác giả: số đo câu, thói quen phong cách, từ đặc trưng đã dùng / nên thử, từ lạ."""
    st = text_stats(draft, guide.genre)
    toks = _tokens(normalize(draft))
    grams = Counter(_ngrams(toks))
    tips = []
    for k, (label, unit) in STAT_NAMES.items():
        if k not in st or k not in guide.stats:
            continue
        mine, his, gen = st[k], guide.stats[k], guide.genre_stats.get(k, guide.stats[k])
        spread = max(abs(his - gen), 0.15 * max(abs(his), 1e-6), 0.5 if unit.startswith("%") else 0.05)
        if abs(mine - his) > spread:
            direction = "tăng" if his > mine else "giảm"
            tips.append({"muc": label, "ban": mine, "tac_gia": his, "the_loai": gen, "don_vi": unit,
                         "goi_y": f"{direction.capitalize()} {label.lower()}: bài của bạn {mine:.1f}, "
                                  f"tác giả {his:.1f} {unit}.".replace(".", ",").rstrip(",") + ".",
                         "do_lech": abs(mine - his) / spread})
    tips.sort(key=lambda t: -t["do_lech"])
    used = [w for w, *_ in guide.signature_words if grams.get(w)]
    try_words = [w for w, *_ in guide.signature_words if not grams.get(w)][:12]
    known = guide.vocab
    strange = [w for w, c in Counter(toks).most_common() if known.get(w, 0) == 0 and len(w) > 1][:12]
    style = []
    if extractor is not None and guide.style_gaps:
        raw = extractor._raw(normalize(draft))
        names = extractor.feature_names
        for name, val_a, val_g, z in guide.style_gaps:
            j = names.index(name)
            mine = float(raw[j])
            closer = abs(mine - val_a) <= abs(mine - val_g)
            style.append({"dac_trung": name, "ban": mine, "tac_gia": val_a, "the_loai": val_g, "giong": closer})
    return {"stats": st, "tips": tips, "used_signature": used, "try_words": try_words, "strange_words": strange,
            "style": style, "n_words": len(toks)}


# ---------------------------------------------------------------------- dàn ý gợi ý
GENRE_PLANS = {
    "tu_su": [("Mở cảnh", "giới thiệu thời gian, nơi chốn và nhân vật chính, gắn với đề bài"),
              ("Diễn biến", "một việc xảy ra làm nhân vật phải lựa chọn hoặc bộc lộ tính cách"),
              ("Đối thoại", "để nhân vật nói với nhau, mỗi lượt lời một dòng"),
              ("Cao trào", "chỗ cảm xúc hoặc mâu thuẫn lên cao nhất"),
              ("Kết", "hệ quả, hoặc một nhận xét ngắn của người kể")],
    "nghi_luan": [("Đặt vấn đề", "nêu vấn đề của đề bài và vì sao đáng bàn"),
                  ("Giải thích", "làm rõ khái niệm, phạm vi"),
                  ("Lập luận và dẫn chứng", "hai đến ba ý, mỗi ý một đoạn, có ví dụ"),
                  ("Phản biện", "nêu ý kiến trái chiều rồi bác bỏ hoặc dung hòa"),
                  ("Kết luận", "tóm ý và rút ra bài học hoặc lời kêu gọi")],
    "tho": [("Khổ mở", "gợi cảnh hoặc cảm xúc ban đầu về đề bài"),
            ("Khổ giữa", "đào sâu cảm xúc, dùng hình ảnh cụ thể"),
            ("Khổ kết", "đúc kết, để lại dư âm")],
}


def outline(topic: str, guide: AuthorGuide, length_words: int) -> dict:
    """Dàn ý cho đề bài `topic`: các phần theo thể loại, số chữ mỗi phần, và lời khuyên lấy từ thói quen tác giả."""
    plan = GENRE_PLANS.get(guide.genre, GENRE_PLANS["tu_su"])
    weights = {"tu_su": [0.15, 0.25, 0.3, 0.2, 0.1], "nghi_luan": [0.12, 0.15, 0.45, 0.16, 0.12],
               "tho": [0.3, 0.4, 0.3]}.get(guide.genre, [1 / len(plan)] * len(plan))
    s = guide.stats
    parts = [{"phan": name, "noi_dung": what, "so_chu": max(10, int(round(w * length_words)))}
             for (name, what), w in zip(plan, weights)]
    advice = []
    if "do_dai_cau" in s:
        advice.append(f"Câu dài khoảng {s['do_dai_cau']:.0f} chữ (trung bình thể loại {guide.genre_stats.get('do_dai_cau', 0):.0f}).")
    if guide.genre == "tu_su" and "loi_thoai" in s:
        advice.append(f"Khoảng {s['loi_thoai']:.0f}% số đoạn là lời thoại, bắt đầu bằng dấu gạch ngang.")
    if guide.genre == "tho" and s.get("the_tho"):
        f, pct = s["the_tho"][0]
        advice.append(f"Tác giả hay viết thể {f} ({pct:.0f}% số bài trong kho).")
        if "chu_moi_dong" in s:
            n_lines = max(4, int(round(length_words / max(s['chu_moi_dong'], 1))))
            advice.append(f"Với {length_words} chữ, viết khoảng {n_lines} dòng.")
    if guide.openers:
        advice.append("Thử mở câu như tác giả: " + ", ".join(f"“{o.capitalize()}…”" for o, _ in guide.openers[:5]) + ".")
    if guide.connectives:
        advice.append("Từ nối hay dùng: " + ", ".join(f"“{c}”" for c, *_ in guide.connectives[:6]) + ".")
    words = [w for w, *_ in guide.signature_words[:12]]
    if words:
        advice.append("Từ ngữ đặc trưng nên lồng vào: " + ", ".join(f"“{w}”" for w in words) + ".")
    if guide.avoided_words:
        advice.append("Tác giả ít dùng: " + ", ".join(f"“{w}”" for w, *_ in guide.avoided_words[:6]) + ".")
    return {"de_bai": topic.strip(), "the_loai": guide.genre, "tong_chu": length_words, "cac_phan": parts,
            "loi_khuyen": advice}
