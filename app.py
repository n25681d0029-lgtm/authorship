"""Giao diện demo định danh tác giả (Streamlit).

Chạy (từ thư mục authorship_attribution):
    ..\\.venv\\Scripts\\streamlit run app.py
rồi mở http://localhost:8501 trên trình duyệt.

Thanh bên: chọn thể loại của văn bản — mô hình chỉ xét các tác giả viết thể loại đó (thí nghiệm
outputs/experiments/v5_the_loai: lọc theo thể loại tăng độ chính xác, chọn sai thể loại thì gần như luôn sai).
Các tab: đoán tác giả (kèm câu nào ảnh hưởng nhiều nhất); xem từng bước (app_pipeline.py); học viết theo tác giả
(app_style.py); so sánh hai văn bản (app_compare.py); tải file (.txt / .csv); lịch sử + yêu thích (app_history.py);
thông tin mô hình. Giao diện (CSS, hiệu ứng) ở app_theme.py. Không gọi dịch vụ AI bên ngoài.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CHECKPOINT_DIR, DATA_DIR  # noqa: E402
from src.explain import author_display_names, describe  # noqa: E402
from src.utils import load_json  # noqa: E402
import app_compare  # noqa: E402
import app_history  # noqa: E402
import app_pipeline  # noqa: E402
import app_style  # noqa: E402
import app_theme as ui  # noqa: E402
from src.history import History  # noqa: E402

st.set_page_config(page_title="Định danh tác giả", page_icon="✒️", layout="wide")
APP_DATA = ROOT / "outputs" / "app_data"


@st.cache_resource
def load_history() -> History:
    h = History(APP_DATA / "lich_su.db")
    h.cleanup()                      # mỗi lần mở app: xóa mục không yêu thích đã quá hạn
    return h


hist = load_history()


@st.cache_resource(show_spinner="Đang nạp mô hình…")
def load_predictor(ckpt: str):
    from src.predict import AuthorPredictor
    return AuthorPredictor.from_checkpoint(ckpt)


@st.cache_data(show_spinner="Đang xử lý văn bản…", max_entries=50)
def run_trace(ckpt: str, text: str) -> dict:
    from src.pipeline_trace import trace
    return trace(load_predictor(ckpt), text)


@st.cache_data
def genre_names() -> dict[str, str]:
    p = DATA_DIR / "genres.json"
    return {g: v["ten"] for g, v in load_json(p)["the_loai"].items()} if p.exists() else {}


@st.cache_data
def display_names() -> dict[str, str]:
    return author_display_names(DATA_DIR / "v5_works.csv")     # cột author, author_name


def list_checkpoints() -> list[str]:
    if not CHECKPOINT_DIR.exists():
        return []
    names = sorted(p.name for p in CHECKPOINT_DIR.iterdir() if (p / "meta.json").exists() and (p / "model.pt").exists())
    return sorted(names, key=lambda n: n != "best_model")      # best_model lên đầu


def nice(slug: str) -> str:
    return "Không xác định" if slug == "unknown" else display_names().get(slug, slug)


@st.cache_data(show_spinner="Máy đang đọc văn bản…", max_entries=100)
def run_predict(ckpt: str, text: str, top_k: int, threshold: float, cands: tuple, filtered: bool) -> dict:
    return load_predictor(ckpt).predict([text], top_k=top_k, reject_below=threshold or None,
                                        candidates=list(cands) if filtered else None)[0]


@st.cache_data(show_spinner="Đang tìm câu ảnh hưởng nhiều nhất…", max_entries=30)
def run_influence(ckpt: str, text: str, author: str, cands: tuple):
    from src.influence import sentence_influence
    return sentence_influence(load_predictor(ckpt), text, author, list(cands))


# ------------------------------------------------------------------ thanh bên
st.sidebar.title("Định danh tác giả")
ckpts = list_checkpoints()
if not ckpts:
    st.error(f"Chưa có mô hình nào trong {CHECKPOINT_DIR}. Hãy huấn luyện trước: python main.py train … --name best_model")
    st.stop()
ckpt = st.sidebar.selectbox("Mô hình", ckpts)
pred = load_predictor(ckpt)
meta = load_json(CHECKPOINT_DIR / ckpt / "meta.json")

# ---- thể loại (danh sách tác giả mỗi thể loại lưu trong checkpoint lúc huấn luyện)
UNKNOWN_GENRE = "Không rõ (xét tất cả tác giả)"
genre_authors = meta["data"].get("genre_authors") or {}
genre = None
if genre_authors:
    options = list(genre_authors) + [UNKNOWN_GENRE]
    genre = st.sidebar.radio("**Văn bản thuộc thể loại nào?**", options, index=None,
                             format_func=lambda g: g if g == UNKNOWN_GENRE else genre_names().get(g, g),
                             help="Máy chỉ so với các tác giả viết thể loại này. Chọn sai thể loại thì gần như chắc "
                                  "chắn đoán sai.")
need_genre = bool(genre_authors) and genre is None
with st.sidebar.expander("Chỉ xét một số tác giả", expanded=False):
    picked_authors = st.multiselect("Chọn tác giả", pred.names, format_func=lambda a: nice(a),
                                    placeholder="Để trống = xét tất cả")
in_genre = set(genre_authors.get(genre, pred.names)) if genre and genre != UNKNOWN_GENRE else set(pred.names)
candidates = [a for a in pred.names if a in in_genre and (not picked_authors or a in picked_authors)]
if genre == UNKNOWN_GENRE:
    genre_text = "không rõ thể loại"
elif genre:
    genre_text = f"thể loại {genre_names().get(genre, genre)}"
else:
    genre_text = "chưa chọn thể loại"
if picked_authors:
    genre_text += f", chỉ xét {len(picked_authors)} tác giả đã chọn"
if len(candidates) < len(pred.names):
    st.sidebar.caption(f"Đang so với {len(candidates)}/{len(pred.names)} tác giả: "
                       + ", ".join(nice(a) for a in candidates))
if not candidates:
    st.sidebar.error("Không còn tác giả nào để so. Hãy bỏ bớt lựa chọn.")

with st.sidebar.expander("Tùy chỉnh", expanded=False):
    default_thr = pred.reject_threshold if pred.reject_threshold is not None else 0.0
    threshold = st.slider("Ngưỡng tin cậy tối thiểu", 0.0, 1.0, float(round(default_thr, 2)), 0.01,
                          help="Nếu tác giả đứng đầu có độ tin cậy thấp hơn mức này, máy trả lời “Không xác định” "
                               "(có thể là tác giả máy chưa biết). Kéo lên thì máy thận trọng hơn.")
    top_k = st.slider("Số tác giả hiện trong biểu đồ", 2, min(10, len(pred.names)), 5)
    motion = st.toggle("Hiệu ứng chuyển động", value=True, help="Tắt nếu máy chậm hoặc bạn thấy rối mắt.")
st.sidebar.caption("Kết quả chỉ để tham khảo, không dùng làm bằng chứng giám định.")
ui.inject(motion)
if st.session_state.get("_flash"):
    st.toast(st.session_state.pop("_flash"))


def show_result(text: str, where: str = "main") -> None:
    if need_genre:
        st.warning("Hãy chọn thể loại của văn bản ở thanh bên trước.")
        return
    if not candidates:
        st.error("Không còn tác giả nào để so. Hãy bỏ bớt lựa chọn ở thanh bên.")
        return
    filtered = len(candidates) < len(pred.names)
    r = run_predict(ckpt, text, top_k, threshold, tuple(candidates), filtered)
    n_words = len(text.split())
    # ghi lịch sử một lần cho mỗi (văn bản, lựa chọn)
    log_key = (hash(text), genre_text, threshold, where)
    logged = st.session_state.setdefault("_logged", {})
    if log_key not in logged:
        logged[log_key] = hist.add("dinh_danh", f"{nice(r['author'])} ({r['confidence']:.0%})", text,
                                   {"author": r["author"], "confidence": r["confidence"], "genre_text": genre_text,
                                    "top": [(t["author"], t["prob"]) for t in r["top_k"]]})
    entry = hist.get(logged[log_key])
    c1, c2 = st.columns([1, 1.25], gap="large")
    with c1:
        unknown = r["author"] == "unknown"
        ui.result_card("Tác giả dự đoán", nice(r["author"]),
                       f"độ tin cậy {100 * r['confidence']:.1f}% · {genre_text}".replace(".", ","), unknown)
        if entry is not None:
            fav = entry["favorite"]
            if st.button("★ Đã lưu vào yêu thích" if fav else "☆ Lưu vào yêu thích", key=f"fav_now_{where}"):
                hist.set_favorite(entry["id"], not fav)
                st.rerun()
        if unknown:
            st.warning(f"Máy không chắc chắn: tác giả đứng đầu chỉ được {r['confidence']:.1%}, thấp hơn ngưỡng "
                       f"{threshold:.0%}. Có thể văn bản của một người máy chưa biết.")
        if n_words < 30:
            st.info("Văn bản hơi ngắn (dưới 30 từ) nên kết quả kém tin cậy.")
        if "lm_weight" in r:
            st.caption(f"Máy dựa {r['lm_weight']:.0%} vào nội dung (PhoBERT), {1 - r['lm_weight']:.0%} vào phong cách.")
        if pred.ensemble_info:
            st.caption(f"Kết quả gộp từ {pred.ensemble_info['n_hybrid']} lần huấn luyện mô hình lai và bộ đếm cụm ký tự.")
        if r["n_chunks"] > 1:
            st.caption(f"Văn bản dài ({n_words} từ) nên được chia làm {r['n_chunks']} đoạn, kết quả là trung bình các đoạn.")
            st.dataframe(pd.DataFrame([{"Đoạn": i + 1, "Số từ": c["n_words"], "Đoạn này giống": nice(c["author"]),
                                        "Độ tin cậy": f"{c['prob']:.1%}"} for i, c in enumerate(r["chunks"])]),
                         hide_index=True, width="stretch")
    with c2:
        st.markdown("**Các tác giả gần nhất**")
        ui.prob_bars([(nice(t["author"]), t["prob"]) for t in r["top_k"]])
    best = r["top_k"][0]["author"]
    t_inf, t_ev = st.tabs([f"Câu nào khiến máy nghĩ là {nice(best)}", "Điểm đáng chú ý trong cách viết"])
    with t_inf:
        inf = run_influence(ckpt, text, best, tuple(candidates)) if n_words <= 600 else None
        if inf:
            ui.highlight([(s, v / inf["scale"]) for s, v in inf["sentences"]])
            st.caption("Máy thử bỏ từng câu rồi đoán lại. Câu tô cam: bỏ đi thì máy bớt tin là "
                       f"{nice(best)}; tô xanh: bỏ đi thì máy lại tin hơn. Màu càng đậm ảnh hưởng càng lớn.")
        else:
            st.caption("Cần văn bản từ 2 đến 40 câu (dưới 600 từ) để xem phần này.")
    with t_ev:
        st.caption("So với mức trung bình của các văn bản máy đã học.")
        for ev in r["style_evidence"]:
            st.markdown(f"- {describe(ev)}")
        st.caption("Muốn xem máy xử lý văn bản này qua từng bước, mở tab Xem từng bước.")


ui.hero("Định danh tác giả", f"Đoán ai viết một đoạn văn tiếng Việt trong {len(pred.names)} tác giả, xem máy làm từng bước, "
                            "học cách viết theo một tác giả.")
tab1, tab_flow, tab_style, tab_cmp, tab2, tab_hist, tab3 = st.tabs(
    ["Đoán tác giả", "Xem từng bước", "Học viết theo tác giả", "So sánh hai văn bản", "Tải file", "Lịch sử", "Về mô hình"])

with tab1:
    text = st.text_area("Dán đoạn văn cần đoán tác giả (càng dài càng chính xác, nên từ 100 từ trở lên)", height=220,
                        key="main_text",
                        placeholder="Lão Hạc ơi! Bây giờ thì tôi hiểu tại sao lão không muốn bán con chó vàng của lão…")
    if st.button("Đoán tác giả", type="primary", disabled=not text.strip()):
        st.session_state["main_shown"] = text
    if text.strip() and st.session_state.get("main_shown") == text:
        show_result(text)

with tab_flow:
    app_pipeline.render(pred, ckpt, meta, candidates, genre_text, threshold, nice, need_genre, run_trace)

with tab_style:
    app_style.render(pred, meta, app_pipeline.data_file(meta), nice, genre_names(), hist, genre)

with tab_cmp:
    _f = app_pipeline.data_file(meta)
    _tp = app_pipeline._test_part(str(_f), _f.stat().st_mtime, meta["config"], meta["data"].get("test_topics")) if _f else None
    app_compare.render(pred, ckpt, candidates or pred.names, genre_text, nice, _tp, hist)

with tab_hist:
    app_history.render(hist, nice)

with tab2:
    st.markdown("Tải lên file **.txt** để đoán một văn bản, hoặc file **.csv** để đoán nhiều văn bản cùng lúc "
                "(mỗi dòng một văn bản trong cột `text`; thêm cột `author` nếu muốn máy tự chấm đúng sai).")
    up = st.file_uploader("Chọn file", type=["txt", "csv"])
    if up is not None and need_genre:
        st.warning("Hãy chọn thể loại của văn bản ở thanh bên trước.")
    elif up is not None:
        raw = up.getvalue()
        if up.name.lower().endswith(".txt"):
            from src.data_loader import read_text_file
            tmp = Path(st.session_state.get("_tmp_dir", ROOT / "outputs")) / "_upload.txt"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(raw)
            show_result(read_text_file(tmp), where="file")
        else:
            df = pd.read_csv(io.BytesIO(raw))
            if "text" not in df.columns:
                st.error(f"File CSV cần có cột 'text'. File này có các cột: {list(df.columns)}")
            else:
                texts = df["text"].astype(str).tolist()
                with st.spinner(f"Đang đoán {len(texts)} văn bản…"):
                    res = pred.predict(texts, top_k=2, explain=False, reject_below=threshold or None,
                                       candidates=candidates if len(candidates) < len(pred.names) else None)
                df["du_doan"] = [r["author"] for r in res]
                df["ten_tac_gia"] = [nice(r["author"]) for r in res]
                df["do_tin_cay"] = [round(r["confidence"], 4) for r in res]
                df["lua_chon_2"] = [nice(r["top_k"][1]["author"]) if len(r["top_k"]) > 1 else "" for r in res]
                if "author" in df.columns:
                    known = df["du_doan"] != "unknown"
                    st.metric("Tỷ lệ đoán đúng (không tính các văn bản máy không chắc)",
                              f"{(df.loc[known, 'du_doan'] == df.loc[known, 'author']).mean():.1%}",
                              f"{(~known).sum()} văn bản máy không chắc", delta_color="off")
                st.dataframe(df, width="stretch")
                st.download_button("Tải kết quả về (.csv)", df.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="ket_qua_dinh_danh.csv", mime="text/csv")

with tab3:
    note = app_pipeline.unknown_authors_note(meta, pred.names)
    if note:
        st.warning(note)
    st.markdown(f"Mô hình **{ckpt}**, học từ {meta['data'].get('n_docs', '?')} đoạn văn của "
                f"{len(pred.names)} tác giả (file `{Path(meta['data']['path']).name}`).")
    rep = CHECKPOINT_DIR / ckpt / "reports" / "test_metrics.json"
    if rep.exists():
        m = load_json(rep)
        st.markdown("**Kết quả trên các đoạn máy chưa từng thấy khi học:**")
        c = st.columns(3)
        c[0].metric("Tỷ lệ đoán đúng", f"{m['accuracy']:.1%}")
        c[1].metric("Macro-F1", f"{m['macro_f1']:.3f}", help="Trung bình độ chính xác của từng tác giả, mỗi tác giả "
                                                            "tính ngang nhau.")
        c[2].metric("Số đoạn kiểm tra", m["n"])
        per = pd.DataFrame([{"Tác giả": nice(a), "Precision": v["precision"], "Recall": v["recall"], "F1": v["f1-score"],
                             "Số đoạn kiểm tra": int(v["support"])} for a, v in m["per_class"].items() if a in pred.names])
        st.caption("Từng tác giả, xếp từ khó đoán nhất. Recall: trong các đoạn của tác giả, máy nhận ra bao nhiêu. "
                   "Precision: khi máy nói là tác giả này, đúng bao nhiêu.")
        st.dataframe(per.sort_values("F1"), hide_index=True, width="stretch")
        png = CHECKPOINT_DIR / ckpt / "reports" / "test_confusion.png"
        if png.exists():
            st.image(str(png), caption="Ma trận nhầm lẫn: hàng là tác giả thật, cột là tác giả máy đoán", width=650)
    if genre_authors:
        st.markdown("**Các tác giả theo thể loại:**")
        for g, authors in genre_authors.items():
            st.markdown(f"- {genre_names().get(g, g)} ({len(authors)}): " + ", ".join(nice(a) for a in authors))
    ens = pred.ensemble_info
    if ens and ens["report"]:
        rep_e = ens["report"]
        st.markdown(f"**Bộ gộp đang dùng:** {ens['n_hybrid']} lần huấn luyện mô hình lai + bộ đếm n-gram — trọng số "
                    + ", ".join(f"{app_pipeline.MEMBER_NAMES.get(k, k)} {v:.0%}" for k, v in ens["weights"].items()))
        c = st.columns(3)
        c[0].metric("Đoán đúng (bộ gộp, không lọc)", f"{rep_e['khong_loc']['accuracy']:.1%}")
        c[1].metric("Macro-F1 (bộ gộp, không lọc)", f"{rep_e['khong_loc']['macro_f1']:.3f}")
        c[2].metric("Đoán đúng khi lọc theo thể loại", f"{rep_e['loc_the_loai']['accuracy']:.1%}")
        st.caption("Cùng tập kiểm tra với các số liệu của mô hình lai bên trên. Chi tiết: reports/ensemble_study.json.")
    cal = meta.get("calibration") or {}
    if cal:
        st.caption(f"Chi tiết kỹ thuật: hiệu chỉnh xác suất T = {cal['temperature']:.2f} (ECE {cal['val_ece_before']:.3f} "
                   f"→ {cal['val_ece_after']:.3f}), ngưỡng tin cậy mặc định {cal['reject_threshold']:.2f}.")
