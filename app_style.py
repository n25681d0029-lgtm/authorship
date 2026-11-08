"""Tab "Học viết theo tác giả": hồ sơ văn phong, dàn ý cho một đề bài, kiểm tra bài người dùng tự viết, và máy viết thử.

Ba phần đầu tính bằng thống kê trên kho dữ liệu (src/style_guide.py) và bộ định danh của nhóm. Phần "Máy viết thử" dùng
GPT nhóm tự huấn luyện từ đầu trong thư mục style_gpt/ (không dùng mô hình hay API có sẵn); nếu chưa có mô hình cho
tác giả đó thì phần này báo rõ và chỉ hiện dàn ý.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import streamlit as st

import app_theme as ui
from src.explain import feature_label
from src.history import History
from src.style_guide import STAT_NAMES, build_guides, check_draft, outline, text_stats

ROOT = Path(__file__).resolve().parent
STYLE_GPT = ROOT.parent / "style_gpt"
SECTIONS = {"profile": "Hồ sơ văn phong", "outline": "Dàn ý cho đề bài", "check": "Kiểm tra bài bạn viết",
            "write": "Máy viết thử"}


def _num(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}".replace(".", ",")


@st.cache_resource(show_spinner="Đang phân tích văn phong các tác giả trong kho (chỉ lần đầu)…")
def load_guides(path: str, mtime: float, _extractor) -> dict:
    from src.data_loader import load_corpus

    df = load_corpus(Path(path))
    if "genre" not in df.columns:
        df = df.assign(genre="tu_su")
    return build_guides(df["text"].tolist(), df["author"].tolist(), df["genre"].tolist(), _extractor)


# ------------------------------------------------------------------ máy viết (style_gpt)
def _gpt_available() -> dict:
    if not (STYLE_GPT / "gen.py").exists():
        return {}
    if str(STYLE_GPT) not in sys.path:
        sys.path.insert(0, str(STYLE_GPT))
    try:
        from gen import available_models
        return available_models()
    except Exception:  # noqa: BLE001 — thiếu thư viện tokenizers... thì coi như chưa có máy viết
        return {}


def writer_for(author: str, genre: str) -> Optional[tuple[str, str, Optional[str]]]:
    """(loại mô hình, đường dẫn, tên tác giả trong lời nhắc) — ưu tiên mô hình riêng HBC cho HBC."""
    models = _gpt_available()
    if author == "ho_bieu_chanh" and "hbc" in models and genre in models["hbc"][1]:
        return "hbc", models["hbc"][0], None
    if "multi" in models:
        path, ga, names = models["multi"]
        if author in ga.get(genre, []):
            return "multi", path, names.get(author, author)
    return None


@st.cache_resource(show_spinner="Đang nạp máy viết…")
def load_writer(path: str):
    from gen import Writer
    return Writer(path)


@st.cache_resource(show_spinner=False)
def load_keywords():
    import pickle
    p = STYLE_GPT / "data" / "keywords.pkl"
    if not p.exists():
        return None
    if str(STYLE_GPT) not in sys.path:
        sys.path.insert(0, str(STYLE_GPT))
    import __main__
    import prepare_data
    __main__.Keywords = prepare_data.Keywords
    return pickle.load(open(p, "rb"))


# chữ trong lời yêu cầu của đề ("Anh/chị hãy viết một đoạn văn trình bày..."), không phải chủ đề cần viết
INSTRUCTION_WORDS = {"anh", "chị", "em", "bạn", "hãy", "viết", "một", "đoạn", "văn", "bài", "chữ", "từ", "khoảng",
                     "trình", "bày", "nêu", "phân", "tích", "bàn", "luận", "câu", "dòng", "ký", "tự", "về", "hay",
                     "sau", "đây", "dưới", "thể", "hiện", "cảm", "nhận", "của", "những", "các", "trước", "và"}


def topic_keywords(topic: str) -> list[str]:
    """Đề bài -> vài từ khóa: có dấu phẩy thì tách theo dấu phẩy; không thì giữ các chữ ít gặp (theo idf của kho GPT)."""
    if "," in topic and len(topic.split()) <= 25:      # danh sách từ khóa ngắn, người dùng tự tách
        return [k.strip() for k in topic.split(",") if k.strip()][:6]
    topic = re.sub(r"\([^)]*\)", " ", topic)            # bỏ phần trong ngoặc, vd "(khoảng 200 chữ)"
    kw = load_keywords()
    words = [w for w in re.findall(r"[^\W\d_]+", topic.lower()) if w not in INSTRUCTION_WORDS]
    if kw is None:
        return [w for w in words if len(w) > 1][:5]
    out, i = [], 0
    while i < len(words):
        w = words[i]
        if i + 1 < len(words) and f"{w} {words[i + 1]}" in kw.bigrams and w not in kw.stop:
            out.append(f"{w} {words[i + 1]}"); i += 2; continue
        if w not in kw.stop and len(w) > 1:
            out.append(w)
        i += 1
    return out[:6] or words[:3]


def _judge(pred, genre_authors: dict):
    from gen import Judge, hbc_ngrams
    j = Judge.__new__(Judge)          # dùng chung bộ định danh của app, không nạp thêm lần nữa
    j.pred, j.genre_authors, j.grams = pred, genre_authors, hbc_ngrams()
    return j


# ------------------------------------------------------------------ các phần
def _profile(g, nice: Callable, genre_name: str) -> None:
    s, gs = g.stats, g.genre_stats
    items = [(STAT_NAMES[k][0], _num(v), f"{STAT_NAMES[k][1]} · thể loại {_num(gs.get(k, v))}")
             for k, v in s.items() if k in STAT_NAMES]
    if s.get("the_tho"):
        f, pct = s["the_tho"][0]
        items.append(("Thể thơ hay dùng", f, f"{_num(pct, 0)}% số bài"))
    st.caption(f"Tính trên {g.n_passages} đoạn ({g.n_words:,} chữ) của {nice(g.author)} trong kho, so với các tác giả "
               f"{genre_name} khác.".replace(",", "."))
    ui.stat_grid(items)
    c = st.columns(2)
    with c[0]:
        st.markdown("**Từ ngữ đặc trưng** — tác giả dùng nhiều hơn hẳn người cùng thể loại")
        st.html(ui.chips([w for w, *_ in g.signature_words[:24]], "warm"))
        st.markdown("**Hay mở câu bằng**")
        st.html(ui.chips([f"{o.capitalize()}… ({_num(p)}%)" for o, p in g.openers[:8]]))
    with c[1]:
        st.markdown("**Từ nối ưa dùng** (lần/1000 chữ, so với thể loại)")
        st.html(ui.chips([f"{w}: {_num(a)} / {_num(b)}" for w, a, b in g.connectives[:8]]))
        st.markdown("**Tác giả hầu như tránh** — người cùng thể loại dùng nhiều")
        st.html(ui.chips([w for w, *_ in g.avoided_words[:12]], "gray"))
    if g.style_gaps:
        with st.expander("Thói quen phong cách lệch nhiều nhất so với thể loại (số đo máy dùng khi đoán)"):
            st.dataframe(pd.DataFrame([{"Thói quen": feature_label(n), "Tác giả": round(a, 3), "Trung bình thể loại": round(b, 3),
                                        "Mức lệch (độ lệch chuẩn)": round(z, 2)} for n, a, b, z in g.style_gaps]),
                         hide_index=True, width="stretch")
    with st.expander("Đọc vài đoạn mẫu"):
        for t in g.examples:
            st.markdown(f"> {t[:900]}".replace("\n", "\n> "))


def _length_input(key: str, default_words: int, chars_per_word: float) -> int:
    c = st.columns([1, 1.2])
    unit = c[0].radio("Đơn vị độ dài", ["chữ", "ký tự"], horizontal=True, key=f"{key}_unit")
    if unit == "chữ":
        n = c[1].number_input("Độ dài mong muốn", 30, 600, default_words, 10, key=f"{key}_n")
        return int(n)
    n = c[1].number_input("Độ dài mong muốn (ký tự, kể cả dấu cách)", 150, 3000, int(default_words * chars_per_word), 50,
                          key=f"{key}_c")
    st.caption(f"≈ {int(n / chars_per_word)} chữ (văn tác giả trung bình {_num(chars_per_word)} ký tự mỗi chữ).")
    return max(30, int(n / chars_per_word))


def _chars_per_word(g) -> float:
    t = " ".join(g.examples)
    w = len(t.split())
    return len(t) / w if w else 4.7


def _outline(g, nice: Callable) -> None:
    topic = st.text_input("Đề bài hoặc vài từ khóa", key="sg_topic",
                          placeholder="Ví dụ: niềm vui khi được ăn ổ bánh mì nóng buổi sáng")
    words = _length_input("sg_ol", 200 if g.genre != "tho" else 80, _chars_per_word(g))
    if not topic.strip():
        st.info("Nhập đề bài để xem dàn ý gợi ý.")
        return
    o = outline(topic, g, words)
    st.markdown(f"**Dàn ý viết “{o['de_bai']}” theo lối {nice(g.author)}** (~{words} chữ)")
    cols = st.columns(len(o["cac_phan"]))
    for col, p in zip(cols, o["cac_phan"]):
        with col, st.container(border=True):
            st.markdown(f"**{p['phan']}**")
            st.caption(f"~{p['so_chu']} chữ")
            st.write(p["noi_dung"])
    st.markdown("**Lời khuyên lấy từ văn của tác giả**")
    for a in o["loi_khuyen"]:
        st.markdown(f"- {a}")
    st.session_state["sg_from_outline"] = topic


def _check(g, pred, nice: Callable, genre_authors: dict, hist: History) -> None:
    draft = st.text_area("Dán bài bạn tự viết", key="sg_draft", height=200,
                         placeholder="Viết thử một đoạn theo lối tác giả rồi bấm Kiểm tra…")
    if not st.button("Kiểm tra", type="primary", key="sg_check", disabled=not draft.strip()):
        return
    cands = genre_authors.get(g.genre) or pred.names
    r = pred.predict([draft], top_k=len(cands), explain=False, candidates=cands)[0]
    p_target = next((t["prob"] for t in r["top_k"] if t["author"] == g.author), 0.0)
    res = check_draft(draft, g, pred.extractor)
    c = st.columns([1, 1.4])
    with c[0]:
        rank = [t["author"] for t in r["top_k"]].index(g.author) + 1 if g.author in [t["author"] for t in r["top_k"]] else None
        ui.result_card(f"Bài của bạn giống {nice(g.author)}", f"{_num(100 * p_target)}%",
                       f"xếp thứ {rank}/{len(cands)} trong các tác giả cùng thể loại" if rank else "")
        st.caption("Bộ định danh của nhóm đọc bài và cho biết khả năng đây là văn của tác giả này. Văn thật của tác giả "
                   "thường được 50–90%.")
    with c[1]:
        st.markdown("Máy thấy bài giống ai nhất:")
        ui.prob_bars([(nice(t["author"]), t["prob"]) for t in r["top_k"][:5]])
    st.markdown("**Gợi ý sửa**")
    if res["tips"]:
        for t in res["tips"][:5]:
            st.markdown(f"- {t['goi_y']}")
    else:
        st.markdown("- Số đo câu đã gần với tác giả.")
    cc = st.columns(3)
    with cc[0]:
        st.markdown("Từ đặc trưng bạn **đã dùng**")
        st.html(ui.chips(res["used_signature"] or ["(chưa có)"], "warm"))
    with cc[1]:
        st.markdown("Từ đặc trưng **nên thử**")
        st.html(ui.chips(res["try_words"]))
    with cc[2]:
        st.markdown(f"Chữ **chưa gặp** trong {g.n_words:,} chữ văn mẫu".replace(",", "."))
        st.html(ui.chips(res["strange_words"] or ["(không có)"], "gray"))
    if res["style"]:
        with st.expander("So từng thói quen phong cách"):
            st.dataframe(pd.DataFrame([{"Thói quen": feature_label(s["dac_trung"]), "Bài của bạn": round(s["ban"], 3),
                                        nice(g.author): round(s["tac_gia"], 3), "Thể loại": round(s["the_loai"], 3),
                                        "Đã giống?": "có" if s["giong"] else "chưa"} for s in res["style"]]),
                         hide_index=True, width="stretch")
    from src.influence import sentence_influence
    inf = sentence_influence(pred, draft, g.author, cands)
    if inf:
        st.markdown(f"**Câu nào giống {nice(g.author)} nhất** (cam: câu kéo bài về phía tác giả, xanh: kéo ra xa)")
        ui.highlight([(s, v / inf["scale"]) for s, v in inf["sentences"]])
    hist.add("kiem_tra", f"Giống {nice(g.author)} {_num(100 * p_target)}%", draft,
             {"author": g.author, "genre": g.genre, "giong": p_target,
              "top": [(t["author"], t["prob"]) for t in r["top_k"][:5]]})


def _write(g, pred, nice: Callable, genre_authors: dict, hist: History) -> None:
    w = writer_for(g.author, g.genre)
    if w is None:
        st.info(f"Chưa có máy viết cho {nice(g.author)} ở thể loại này. Máy viết là GPT nhóm tự huấn luyện trong thư mục "
                "style_gpt (xem style_gpt/README.md); khi huấn luyện xong, phần này tự bật. Trong lúc chờ, dùng mục "
                "Dàn ý cho đề bài.")
        return
    kind, path, prompt_name = w
    default_topic = st.session_state.get("sg_from_outline", "bánh mì, ngon, hạnh phúc")
    topic = st.text_input("Đề bài hoặc từ khóa (cách nhau bằng dấu phẩy)", value=default_topic, key="sg_w_topic")
    opening = st.text_input("Câu mở đầu (không bắt buộc)", key="sg_w_open")
    words = _length_input("sg_w", 150 if g.genre != "tho" else 60, _chars_per_word(g))
    with st.expander("Tùy chỉnh"):
        n = st.slider("Số bản viết thử", 2, 12, 8, key="sg_w_nd",
                      help="Máy viết nhiều bản rồi chọn bản bộ định danh thấy giống tác giả nhất và bám đề nhất.")
        temp = st.slider("Độ phóng khoáng", 0.5, 1.0, 0.7, 0.05, key="sg_w_t",
                         help="Thấp: chắc chữ, ít lỗi, hơi nhạt. Cao: nhiều chữ lạ, dễ sai câu và lạc đề. 0,6–0,8 là hợp nhất.")
    kws = topic_keywords(topic)
    if kws:
        st.caption("Từ khóa máy dùng: " + ", ".join(kws))
    go = st.button("Viết", type="primary", key="sg_w_go", disabled=not topic.strip())
    st.caption("Máy chỉ bắt chước giọng văn và cố đưa các từ khóa vào bài; nó không hiểu đề nên ý có thể rời rạc hoặc lạc đề. "
               "Văn do máy viết, không phải tác phẩm thật. "
               + ("Mô hình riêng cho Hồ Biểu Chánh." if kind == "hbc" else "Mô hình chung nhiều tác giả."))
    if go:
        writer = load_writer(path)
        ph = st.empty()
        ph.html('<div class="dots"><span></span><span></span><span></span></div>')
        ids = writer.prompt_ids(kws, g.genre, opening, author_name=prompt_name)
        cands = writer.sample(ids, n=n, words=min(words, 330), temperature=temp, keywords=kws)
        if opening.strip():
            for c in cands:
                c["gen_text"], c["text"] = c["text"], opening.strip() + " " + c["text"]
        ranked = _judge(pred, genre_authors).score(cands, kws, g.genre, g.author)
        ph.empty()
        st.session_state["sg_w_res"] = {"ranked": ranked, "author": g.author, "genre": g.genre, "topic": topic}
        b = ranked[0]
        hist.add("viet", f"{nice(g.author)}: {topic[:60]}", b["text"],
                 {"author": g.author, "genre": g.genre, "giong": b["giong_hbc"], "chep": b["chep"], "tu_khoa": b["tu_khoa"]})
    res = st.session_state.get("sg_w_res")
    if res and res["author"] == g.author and res["genre"] == g.genre:
        b = res["ranked"][0]
        with st.container(border=True):
            for para in b["text"].split("\n"):
                st.write(para)
        ui.stat_grid([("Giống " + nice(g.author), f"{_num(100 * b['giong_hbc'])}%", "theo bộ định danh"),
                      ("Chép nguyên văn", f"{_num(100 * b['chep'])}%", "cụm 8 chữ trùng văn thật"),
                      ("Có nhắc từ khóa", f"{_num(100 * b['tu_khoa'], 0)}%", ""),
                      ("Độ dài", f"{len(b['text'].split())} chữ", f"{len(b['text'])} ký tự")])
        st.download_button("Tải bài về (.txt)", b["text"].encode("utf-8"), file_name="bai_may_viet.txt", key="sg_w_dl")
        with st.expander(f"Xem {len(res['ranked']) - 1} bản còn lại"):
            for i, r in enumerate(res["ranked"][1:], 2):
                st.markdown(f"**Bản {i}** — giống {_num(100 * r['giong_hbc'])}%, chép {_num(100 * r['chep'])}%")
                st.write(r["text"])


def render(pred, meta: dict, data_path: Optional[Path], nice: Callable, genre_names: dict, hist: History,
           default_genre: Optional[str] = None) -> None:
    st.markdown("### Học viết theo một tác giả")
    st.caption("Chọn tác giả để xem họ viết thế nào, nhận dàn ý cho đề bài của bạn, nhờ máy chấm bài bạn tự viết, hoặc "
               "xem máy viết thử một đoạn.")
    if data_path is None:
        st.warning("Không tìm thấy kho dữ liệu đã dùng để dạy máy nên chưa phân tích được văn phong.")
        return
    guides = load_guides(str(data_path), data_path.stat().st_mtime, pred.extractor)
    genre_authors = meta["data"].get("genre_authors") or {}
    genres = sorted({g for g, _ in guides}, key=lambda g: list(genre_authors).index(g) if g in genre_authors else 99)
    if default_genre in genres and st.session_state.get("_sg_side") != default_genre:
        st.session_state["_sg_side"] = default_genre        # đổi thể loại ở thanh bên thì tab này đổi theo
        st.session_state["sg_genre"] = default_genre
    c = st.columns([1, 1.5, 2.5])
    genre = c[0].selectbox("Thể loại", genres, format_func=lambda g: genre_names.get(g, g), key="sg_genre")
    authors = sorted([a for gg, a in guides if gg == genre], key=lambda a: nice(a))
    author = c[1].selectbox("Tác giả", authors, format_func=nice, key="sg_author")
    with c[2]:
        sec = st.segmented_control("Phần", list(SECTIONS), format_func=SECTIONS.get, default="profile", key="sg_sec",
                                   label_visibility="hidden") or "profile"
    g = guides[(genre, author)]
    with st.container(border=True):
        if sec == "profile":
            _profile(g, nice, genre_names.get(genre, genre).lower())
        elif sec == "outline":
            _outline(g, nice)
        elif sec == "check":
            _check(g, pred, nice, genre_authors, hist)
        else:
            _write(g, pred, nice, genre_authors, hist)
