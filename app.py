"""Giao diện demo định danh tác giả (Streamlit).

Chạy (từ thư mục authorship_attribution):
    ..\\.venv\\Scripts\\streamlit run app.py
rồi mở http://localhost:8501 trên trình duyệt.

Bước 1 (thanh bên): chọn thể loại của văn bản — mô hình chỉ xét các tác giả viết thể loại đó (thí nghiệm
outputs/experiments/v5_the_loai: lọc theo thể loại tăng rõ độ chính xác, chọn sai thể loại thì gần như luôn sai).
Ba tab: nhập văn bản, tải file (.txt một văn bản hoặc .csv nhiều văn bản, có tải kết quả về), thông tin mô hình.
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

st.set_page_config(page_title="Định danh tác giả", page_icon="✒️", layout="wide")


@st.cache_resource(show_spinner="Đang nạp mô hình…")
def load_predictor(ckpt: str):
    from src.predict import AuthorPredictor
    return AuthorPredictor.from_checkpoint(ckpt)


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
    return "Không xác định (unknown)" if slug == "unknown" else display_names().get(slug, slug)


# ------------------------------------------------------------------ thanh bên
st.sidebar.title("✒️ Định danh tác giả")
ckpts = list_checkpoints()
if not ckpts:
    st.error(f"Chưa có checkpoint nào trong {CHECKPOINT_DIR}. Huấn luyện bằng: python main.py train … --name best_model")
    st.stop()
ckpt = st.sidebar.selectbox("Mô hình", ckpts)
pred = load_predictor(ckpt)
meta = load_json(CHECKPOINT_DIR / ckpt / "meta.json")
default_thr = pred.reject_threshold if pred.reject_threshold is not None else 0.0
threshold = st.sidebar.slider("Ngưỡng từ chối (dưới ngưỡng → “không xác định”)", 0.0, 1.0, float(round(default_thr, 2)),
                              0.01, help="Mặc định là ngưỡng học trên tập kiểm định (chấp nhận 95% đoạn). "
                                         "Tăng lên để từ chối văn bản lạ nhiều hơn, đổi lại từ chối nhầm cả văn bản quen.")
top_k = st.sidebar.slider("Số tác giả hiển thị", 2, min(10, len(pred.names)), 5)

# ---- bước 1: thể loại (danh sách tác giả mỗi thể loại lưu trong checkpoint lúc huấn luyện)
UNKNOWN_GENRE = "Không rõ thể loại (xét tất cả tác giả)"
genre_authors = meta["data"].get("genre_authors") or {}
genre = None
if genre_authors:
    options = list(genre_authors) + [UNKNOWN_GENRE]
    genre = st.sidebar.radio("**Bước 1 — Thể loại của văn bản**", options, index=None,
                             format_func=lambda g: g if g == UNKNOWN_GENRE else genre_names().get(g, g),
                             help="Mô hình chỉ xét tác giả viết thể loại này. Chọn sai thể loại thì tác giả thật "
                                  "thường bị loại khỏi danh sách và kết quả gần như chắc chắn sai.")
need_genre = bool(genre_authors) and genre is None
with st.sidebar.expander("🎯 Thu hẹp thêm: chọn tác giả nghi vấn", expanded=False):
    picked_authors = st.multiselect("Chỉ xét các tác giả này", pred.names, format_func=lambda a: nice(a))
in_genre = set(genre_authors.get(genre, pred.names)) if genre and genre != UNKNOWN_GENRE else set(pred.names)
candidates = [a for a in pred.names if a in in_genre and (not picked_authors or a in picked_authors)]
if len(candidates) < len(pred.names):
    st.sidebar.info(f"Đang xét {len(candidates)}/{len(pred.names)} tác giả: " + ", ".join(nice(a) for a in candidates))
if not candidates:
    st.sidebar.error("Không còn tác giả nào thỏa điều kiện lọc.")
cfg = meta["config"]
st.sidebar.caption(f"Chế độ: **{cfg['fusion']}{' + n-gram ký tự' if cfg.get('char_ngram') else ''}** · "
                   f"{len(pred.names)} tác giả · nhiệt độ hiệu chỉnh T = {pred.temperature:.2f}")
st.sidebar.caption("Kết quả chỉ mang tính tham khảo, không dùng làm bằng chứng giám định.")


def show_result(text: str) -> None:
    if need_genre:
        st.warning("Hãy chọn thể loại của văn bản ở thanh bên (Bước 1) trước khi dự đoán.")
        return
    if not candidates:
        st.error("Bộ lọc tác giả ứng viên đang loại hết tác giả.")
        return
    filtered = len(candidates) < len(pred.names)
    r = pred.predict([text], top_k=top_k, reject_below=threshold or None,
                     candidates=candidates if filtered else None)[0]
    if filtered:
        st.info(f"Kết quả chỉ xét {len(candidates)} tác giả theo thể loại/bộ lọc đã chọn; xác suất đã chuẩn hóa lại "
                "trong nhóm này nên thường cao hơn khi xét đủ tác giả.")
    n_words = len(text.split())
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.metric("Tác giả dự đoán", nice(r["author"]), f"độ tin cậy {r['confidence']:.1%}",
                  delta_color="off")
        if r["author"] == "unknown":
            st.warning(f"Độ tin cậy cao nhất ({r['confidence']:.1%}) thấp hơn ngưỡng {threshold:.0%}: "
                       f"văn bản có thể của người ngoài {len(pred.names)} tác giả mô hình biết.")
        if n_words < 30:
            st.info("Văn bản ngắn hơn 30 từ: độ chính xác giảm mạnh (xem báo cáo, mục văn bản ngắn).")
        if "lm_weight" in r:
            st.caption(f"Trọng số cổng: {r['lm_weight']:.0%} luồng ngôn ngữ (PhoBERT), "
                       f"{1 - r['lm_weight']:.0%} luồng phong cách.")
        if r["n_chunks"] > 1:
            st.caption(f"Văn bản dài ({n_words} từ) được cắt thành {r['n_chunks']} đoạn; xác suất là trung bình các đoạn.")
            st.dataframe(pd.DataFrame([{"Đoạn": i + 1, "Số âm tiết": c["n_words"], "Đoán": nice(c["author"]),
                                        "Tin cậy": f"{c['prob']:.1%}"} for i, c in enumerate(r["chunks"])]),
                         hide_index=True, width="stretch")
    with c2:
        top = pd.DataFrame([{"Tác giả": nice(t["author"]), "Xác suất": t["prob"]} for t in r["top_k"]])
        st.bar_chart(top, x="Tác giả", y="Xác suất", horizontal=True, sort="-Xác suất")
    st.subheader("Vì sao? Các đặc điểm phong cách nổi bật của văn bản")
    st.caption("So với trung bình kho huấn luyện (z-score). Đây là luồng phong cách; luồng PhoBERT không giải thích được "
               "bằng câu chữ.")
    for ev in r["style_evidence"]:
        st.markdown(f"- {describe(ev)}")


tab1, tab2, tab3 = st.tabs(["📝 Nhập văn bản", "📂 Tải file", "ℹ️ Về mô hình"])

with tab1:
    text = st.text_area("Dán đoạn văn tiếng Việt (nên 100–200 từ trở lên)", height=220,
                        placeholder="Lão Hạc ơi! Bây giờ thì tôi hiểu tại sao lão không muốn bán con chó vàng của lão…")
    if st.button("Dự đoán", type="primary", disabled=not text.strip()):
        show_result(text)

with tab2:
    st.markdown("**.txt**: một văn bản. **.csv**: nhiều văn bản, cần cột `text` (cột `author` nếu có sẽ được dùng để "
                "tính độ chính xác).")
    up = st.file_uploader("Chọn file", type=["txt", "csv"])
    if up is not None and need_genre:
        st.warning("Hãy chọn thể loại của văn bản ở thanh bên (Bước 1) trước khi dự đoán.")
    elif up is not None:
        raw = up.getvalue()
        if up.name.lower().endswith(".txt"):
            from src.data_loader import read_text_file
            tmp = Path(st.session_state.get("_tmp_dir", ROOT / "outputs")) / "_upload.txt"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(raw)
            show_result(read_text_file(tmp))
        else:
            df = pd.read_csv(io.BytesIO(raw))
            if "text" not in df.columns:
                st.error(f"File CSV cần cột 'text'. Các cột hiện có: {list(df.columns)}")
            else:
                texts = df["text"].astype(str).tolist()
                with st.spinner(f"Đang dự đoán {len(texts)} văn bản…"):
                    res = pred.predict(texts, top_k=2, explain=False, reject_below=threshold or None,
                                       candidates=candidates if len(candidates) < len(pred.names) else None)
                df["du_doan"] = [r["author"] for r in res]
                df["ten_tac_gia"] = [nice(r["author"]) for r in res]
                df["do_tin_cay"] = [round(r["confidence"], 4) for r in res]
                df["lua_chon_2"] = [nice(r["top_k"][1]["author"]) if len(r["top_k"]) > 1 else "" for r in res]
                if "author" in df.columns:
                    known = df["du_doan"] != "unknown"
                    st.metric("Độ chính xác (trên văn bản không bị từ chối)",
                              f"{(df.loc[known, 'du_doan'] == df.loc[known, 'author']).mean():.1%}",
                              f"{(~known).sum()} văn bản bị từ chối", delta_color="off")
                st.dataframe(df, width="stretch")
                st.download_button("Tải kết quả (.csv)", df.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="ket_qua_dinh_danh.csv", mime="text/csv")

with tab3:
    st.markdown(f"**Checkpoint:** `{ckpt}` · **dữ liệu:** `{Path(meta['data']['path']).name}` · "
                f"**số đoạn:** {meta['data'].get('n_docs', '?')}")
    rep = CHECKPOINT_DIR / ckpt / "reports" / "test_metrics.json"
    if rep.exists():
        m = load_json(rep)
        c = st.columns(3)
        c[0].metric("Accuracy (test)", f"{m['accuracy']:.1%}")
        c[1].metric("Macro-F1 (test)", f"{m['macro_f1']:.3f}")
        c[2].metric("Số đoạn test", m["n"])
        per = pd.DataFrame([{"Tác giả": nice(a), "Precision": v["precision"], "Recall": v["recall"], "F1": v["f1-score"],
                             "Số đoạn test": int(v["support"])} for a, v in m["per_class"].items() if a in pred.names])
        st.dataframe(per.sort_values("F1"), hide_index=True, width="stretch")
        png = CHECKPOINT_DIR / ckpt / "reports" / "test_confusion.png"
        if png.exists():
            st.image(str(png), caption="Ma trận nhầm lẫn trên tập test", width=650)
    if genre_authors:
        st.markdown("**Tác giả theo thể loại:**")
        for g, authors in genre_authors.items():
            st.markdown(f"- {genre_names().get(g, g)} ({len(authors)}): " + ", ".join(nice(a) for a in authors))
        st.caption("So sánh các cách dùng thể loại (3 seed): outputs/experiments/v5_the_loai/summary.md")
    cal = meta.get("calibration") or {}
    if cal:
        st.caption(f"Hiệu chỉnh xác suất: T = {cal['temperature']:.2f}, ECE trên tập kiểm định "
                   f"{cal['val_ece_before']:.3f} → {cal['val_ece_after']:.3f}; ngưỡng từ chối mặc định "
                   f"{cal['reject_threshold']:.3f}.")
