"""Tab "Hệ thống hoạt động thế nào" của app.py: cho người xem bấm chọn từng giai đoạn để thấy máy đọc văn bản,
trích đặc trưng và chọn tác giả ra sao. Số liệu tính bởi src/pipeline_trace.py, trùng với lần dự đoán thật.

Không ghi cứng tác giả, thể loại hay kích thước mô hình: mọi thứ lấy từ checkpoint (meta.json) và kho dữ liệu, nên
thêm tác giả / thể loại rồi huấn luyện lại thì tab vẫn chạy như cũ.
"""
from __future__ import annotations

import html
import random
from pathlib import Path
from typing import Callable, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from config import DATA_DIR, TrainConfig
from src.explain import describe, feature_label
from src.pipeline_trace import author_style_profiles, decide, style_distance, trace

STAGES = {
    "doc": ("1", "Làm sạch văn bản"),
    "chunk": ("2", "Chia đoạn"),
    "token": ("3", "Tách từ"),
    "style": ("4", "Đo cách viết"),
    "fusion": ("5", "Chấm điểm"),
    "prob": ("6", "Tính xác suất"),
    "decide": ("7", "Kết luận"),
    "profile": ("8", "So với từng tác giả"),
}
DEFAULT_STAGES = ["doc", "style", "fusion", "decide"]
STREAM_NAMES = {"lm": "Phần đọc hiểu nội dung (PhoBERT)", "style": "Phần đo cách viết",
                "fixed": "Điểm cộng sẵn của tác giả"}
STREAM_COLORS = {"lm": "#4C78A8", "style": "#F58518", "fixed": "#9D9DA0"}
# Sơ đồ graphviz: nền ô sáng, chữ tối — nếu để nền trong suốt thì chữ đen biến mất trên giao diện tối
NODE_STYLE = ('node [fontname=Arial, fontsize=11, shape=box, style="filled,rounded", fillcolor="#EDEDED", '
              'fontcolor="#1F1F1F", color="#8C8C8C"];')
PREVIEW_AUTHORS = 3     # bước ⑤: số tác giả ở biểu đồ xem trước; xem đủ trong mục mở rộng
MEMBER_NAMES = {"hybrid": "Mô hình lai (PhoBERT + cách viết)", "char_svm": "Bộ đếm cụm ký tự (SVM)",
                "word_svm": "Bộ đếm cụm từ (SVM)"}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%".replace(".", ",")


def _num(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}".replace(".", ",")


# ------------------------------------------------------------------ dữ liệu đi kèm checkpoint
def data_file(meta: dict) -> Optional[Path]:
    """Kho dữ liệu đã dùng để huấn luyện; nếu checkpoint chép từ máy khác thì tìm theo tên file trong data/."""
    p = Path(meta.get("data", {}).get("path", ""))
    for cand in (p, DATA_DIR / p.name):
        if cand.name and cand.is_file():
            return cand
    return None


@st.cache_data(show_spinner="Đang đọc kho dữ liệu…")
def _load_data(path: str, mtime: float) -> pd.DataFrame:
    from src.data_loader import load_corpus

    return load_corpus(Path(path))


@st.cache_data(show_spinner="Đang dựng lại tập kiểm tra…")
def _test_part(path: str, mtime: float, cfg_dict: dict, test_topics) -> pd.DataFrame:
    """Đoạn thuộc tập test — cùng cách chia lúc huấn luyện (cùng seed), mô hình chưa từng học các đoạn này."""
    from src.data_loader import make_splits

    df = _load_data(path, mtime)
    cfg = TrainConfig.from_dict(cfg_dict)
    if "genre" in df.columns and not test_topics and not cfg.topic_holdout:
        from src.genre_experiment import genre_splits

        df = genre_splits(df, cfg)
    else:
        df = make_splits(df, cfg.val_size, cfg.test_size, cfg.seed, test_topics, cfg.topic_holdout or None)
    return df[df["split"] == "test"].drop(columns=["split"])


def load_data(meta: dict) -> Optional[pd.DataFrame]:
    f = data_file(meta)
    return _load_data(str(f), f.stat().st_mtime) if f else None


def unknown_authors_note(meta: dict, model_names: list[str]) -> Optional[str]:
    """Cảnh báo khi kho dữ liệu đã có tác giả mà mô hình chưa học (thêm dữ liệu nhưng chưa huấn luyện lại)."""
    df = load_data(meta)
    if df is None:
        return None
    new = sorted(set(df["author"]) - set(model_names))
    if new:
        return (f"Dữ liệu có {len(new)} tác giả mà mô hình chưa được dạy ({', '.join(new[:8])}"
                f"{'…' if len(new) > 8 else ''}). Cần huấn luyện lại để máy nhận ra họ.")
    return None


@st.cache_data(show_spinner="Đang tính hồ sơ phong cách của các tác giả…")
def _profiles(ckpt: str, path: str, mtime: float, _extractor, model_names: tuple) -> tuple:
    df = _load_data(path, mtime)
    df = df[df["author"].isin(model_names)]
    return author_style_profiles(_extractor, df["text"].tolist(), df["author"].tolist())


# ------------------------------------------------------------------ hình vẽ
def _flow_dot(selected: list[str], uses_lm: bool) -> str:
    def node(key: str, extra: str = "") -> str:
        num, label = STAGES[key]
        on = key in selected
        style = 'fillcolor="#FFD666", penwidth=2' if on else 'fillcolor="#EDEDED"'
        return f'{key} [label="{num}. {label}{extra}", shape=box, {style}];'

    lines = ["digraph G {", "rankdir=LR; bgcolor=transparent; " + NODE_STYLE + " edge [color=gray55, fontcolor=gray55];",
             node("doc"), node("chunk"), node("style"), node("fusion"), node("prob"), node("decide"), node("profile")]
    if uses_lm:
        lines.append(node("token"))
        lines += ["doc -> chunk; chunk -> token; chunk -> style; token -> fusion; style -> fusion;"]
    else:
        lines += ["doc -> chunk; chunk -> style; style -> fusion;"]
    lines += ["fusion -> prob -> decide;", 'style -> profile [style=dashed, label=" để giải thích", fontsize=9];', "}"]
    return "\n".join(lines)


def _hbar(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, fmt: str = ".2f", height: int = None,
          domain=None, color_scale=None, tooltip=None):
    enc_color = alt.Color(color, scale=color_scale, legend=alt.Legend(orient="top", title=None, labelLimit=0)) if color else alt.value("#4C78A8")
    ch = alt.Chart(df).mark_bar().encode(
        x=alt.X(x, title=None, scale=alt.Scale(domain=domain) if domain else alt.Undefined, axis=alt.Axis(format=fmt)),
        y=alt.Y(y, sort=None, title=None, axis=alt.Axis(labelLimit=0)),
        color=enc_color, tooltip=tooltip or [y, alt.Tooltip(x, format=fmt)])
    return ch.properties(height=height or alt.Step(28))


def _token_chips(tokens: list[str]) -> str:
    chips = []
    for t in tokens:
        if t in ("<s>", "</s>", "<pad>", "<unk>", "[CLS]", "[SEP]"):
            bg, title = "rgba(150,150,150,.30)", "ký hiệu đánh dấu đầu/cuối"
        elif t.endswith("@@"):
            bg, title = "rgba(245,133,24,.30)", "một mảnh của từ dài"
        elif "_" in t.strip("_"):
            bg, title = "rgba(76,120,168,.30)", "từ ghép"
        else:
            bg, title = "rgba(120,120,120,.10)", "từ đơn hoặc dấu câu"
        chips.append(f'<span title="{title}" style="background:{bg};border-radius:6px;padding:1px 6px;margin:2px;'
                     f'display:inline-block;font-size:0.9em">{html.escape(t)}</span>')
    return '<div style="line-height:1.9">' + "".join(chips) + "</div>"


# ------------------------------------------------------------------ từng giai đoạn
def _header(key: str, what: str) -> None:
    num, label = STAGES[key]
    st.markdown(f"#### Bước {num}. {label}")
    st.caption(what)


def _stage_doc(tr: dict) -> None:
    _header("doc", "Mỗi bản chép trình bày một kiểu (gạch nối “văn-minh”, ngoặc « », dấu cách trước dấu câu…). Máy đưa "
                   "chúng về một dạng chung để chỉ nhìn vào cách viết của tác giả, không bị lẫn thói quen của người chép.")
    c = st.columns(4)
    c[0].metric("Số ký tự", f"{tr['n_chars']:,}".replace(",", "."))
    c[1].metric("Số từ", tr["n_words"])
    c[2].metric("Số câu", tr["n_sentences"])
    c[3].metric("Chữ lạ", _pct(tr["invalid_ratio"]),
                help="Tỷ lệ chữ không giống tiếng Việt (tên nước ngoài, lỗi gõ, lỗi quét ảnh). Dữ liệu máy học đều dưới 3%.")
    if tr["n_words"] < 30:
        st.warning("Văn bản hơi ngắn (dưới 30 từ) nên có ít dấu hiệu để nhận ra tác giả.")
    if tr["invalid_ratio"] > 0.03:
        st.warning("Văn bản có khá nhiều chữ lạ, có thể bị lỗi khi quét ảnh hoặc chứa nhiều tiếng nước ngoài.")
    if tr["norm_steps"]:
        st.markdown("Những chỗ đã sửa:")
        st.dataframe(pd.DataFrame([{"Sửa gì": s["rule"], "Số chỗ": s["count"], "Vì sao": s["why"]}
                                   for s in tr["norm_steps"]]), hide_index=True, width="stretch")
    else:
        st.write("Văn bản đã sạch, không cần sửa gì.")
    with st.expander("Xem văn bản sau khi sửa"):
        st.text(tr["normalized"])


def _stage_chunk(tr: dict, nice: Callable, candidates) -> None:
    _header("chunk", "Máy được dạy trên các đoạn dài 50–200 từ. Văn bản dài hơn thế sẽ được chia nhỏ, đoán từng đoạn "
                     "rồi lấy trung bình.")
    ch = tr["chunks"]
    if len(ch) == 1:
        st.write(f"Văn bản có {tr['n_words']} từ, đủ ngắn nên giữ nguyên.")
        return
    rows = []
    for i, c in enumerate(ch):
        d = decide(c["probs"], tr["names"], candidates)
        rows.append({"Đoạn": i + 1, "Số từ": c["n_words"], "Mở đầu": " ".join(c["text"].split()[:12]) + "…",
                     "Đoạn này giống": nice(d["best"]), "Độ tin cậy": _pct(d["confidence"])})
    st.write(f"Văn bản có {tr['n_words']} từ, được chia làm {len(ch)} đoạn:")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _stage_token(tr: dict, c: dict) -> None:
    _header("token", "PhoBERT là mô hình đã đọc rất nhiều tiếng Việt. Nó đọc theo từ chứ không theo tiếng, nên trước hết "
                     "máy ghép các tiếng thành từ (“học sinh” thành “học_sinh”), rồi chia câu thành các mảnh nhỏ mà PhoBERT biết.")
    t = c["tokens"]
    if t is None:
        st.write("Mô hình này không dùng PhoBERT nên bỏ qua bước này.")
        return
    n_comp = sum("_" in x.strip("_") for x in t["tokens"])
    k = st.columns(3)
    k[0].metric("Số mảnh", t["n_tokens"])
    k[1].metric("Số từ ghép", n_comp)
    k[2].metric("PhoBERT đọc tối đa", f"{t['max_length']} mảnh")
    if t["truncated"]:
        st.warning(f"Đoạn này dài {t['n_tokens']} mảnh, PhoBERT chỉ đọc {t['max_length']} mảnh đầu "
                   "(phần đo cách viết vẫn đọc hết).")
    st.markdown(_token_chips(t["tokens"]), unsafe_allow_html=True)
    st.caption("Xanh: từ ghép. Cam: mảnh của một từ dài (dấu @@ nghĩa là còn tiếp). Xám đậm: ký hiệu đánh dấu đầu và "
               "cuối đoạn.")


def _stage_style(tr: dict, c: dict) -> None:
    names = tr["feature_names"]
    n_fw = sum(n.startswith("fw[") for n in names)
    _header("style", f"Cùng lúc đó, máy đo {len(names)} thói quen viết ít phụ thuộc vào nội dung: câu dài hay ngắn, dùng "
                     f"dấu câu nào nhiều, dùng các từ quen thuộc như “thì”, “mà”, “của” ra sao ({n_fw} từ). Mỗi thói quen "
                     "được so với mức trung bình của các văn bản máy đã học: thanh càng dài thì càng khác thường.")
    z, raw = c["style_z"], c["style_raw"]
    top_n = st.slider("Số thói quen hiển thị", 5, min(30, len(names)), 10, key="style_top_n")
    order = np.argsort(-np.abs(z))[:top_n]
    df = pd.DataFrame({"Thói quen": [feature_label(names[i]) for i in order], "Mức khác thường": z[order],
                       "Văn bản này": raw[order], "Trung bình": tr["corpus_mean"][order],
                       "So với bình thường": ["nhiều hơn" if z[i] > 0 else "ít hơn" for i in order]})
    st.altair_chart(_hbar(df, "Mức khác thường", "Thói quen", "So với bình thường",
                          color_scale=alt.Scale(domain=["nhiều hơn", "ít hơn"], range=["#E45756", "#4C78A8"]),
                          tooltip=["Thói quen", alt.Tooltip("Văn bản này", format=".3f"),
                                   alt.Tooltip("Trung bình", format=".3f")]),
                    width="stretch")
    st.markdown("Nói đơn giản:")
    for i in order[:5]:
        st.markdown("- " + describe({"feature": names[i], "value": float(raw[i]), "z": float(z[i])}))
    if tr["char_ngram"]:
        st.caption("Mô hình này còn xét thêm các cụm ký tự hay gặp.")
    with st.expander(f"Xem đủ {len(names)} thói quen"):
        st.dataframe(pd.DataFrame({"Thói quen": [feature_label(n) for n in names], "Văn bản này": raw,
                                   "Trung bình": tr["corpus_mean"], "Mức khác thường": z}).round(3),
                     hide_index=True, width="stretch")


def _contrib_frame(tr: dict, c: dict, idx: list[int], nice: Callable) -> pd.DataFrame:
    rows = []
    for j in idx:
        for k, v in c["contrib"].items():
            rows.append({"Tác giả": nice(tr["names"][j]), "Phần": STREAM_NAMES[k], "Điểm": float(v[j])})
        rows.append({"Tác giả": nice(tr["names"][j]), "Phần": STREAM_NAMES["fixed"],
                     "Điểm": float(c["intercept"][j] + tr["class_bias"][j])})
    return pd.DataFrame(rows)


def _contrib_chart(df: pd.DataFrame, order: list[str]):
    """Thanh ngang xếp chồng; mỗi tác giả một dòng cao cố định (alt.Step) nên ít hay nhiều tác giả cũng không đè tên."""
    return alt.Chart(df).mark_bar().encode(
        y=alt.Y("Tác giả", sort=order, title=None, axis=alt.Axis(labelLimit=0)),
        x=alt.X("sum(Điểm)", title="điểm"),
        color=alt.Color("Phần", scale=alt.Scale(domain=list(STREAM_NAMES.values()), range=list(STREAM_COLORS.values())),
                        legend=alt.Legend(orient="top", title=None, direction="vertical", labelLimit=0)),
        tooltip=["Tác giả", "Phần", alt.Tooltip("Điểm", format=".2f")]).properties(height=alt.Step(40))


def _stage_fusion(tr: dict, c: dict, nice: Callable, candidates) -> None:
    _header("fusion", "Kết quả của phần đọc nội dung (PhoBERT) và phần đo cách viết được gộp lại để chấm điểm cho từng "
                      "tác giả. Biểu đồ cho thấy mỗi phần góp bao nhiêu điểm.")
    n_auth = len(tr["names"])
    if tr["fusion"] in ("concat", "gated", "lm_only"):
        dims = (f'lm [label="Đọc nội dung (PhoBERT)\\n→ {c["lm_dim"]} số", fillcolor="#C9D9EC", color="{STREAM_COLORS["lm"]}", penwidth=2];')
    else:
        dims = ""
    sty = (f'sty [label="Đo {c["style_input_dim"]} thói quen viết\\n→ {c["style_out_dim"]} số", '
           f'fillcolor="#FBDDBF", color="{STREAM_COLORS["style"]}", penwidth=2];') if c["style_out_dim"] else ""
    fuse_label = {"concat": "Gộp lại", "gated": "Trộn lại", "lm_only": "Chỉ PhoBERT", "style_only": "Chỉ cách viết"}
    dot = ["digraph G { rankdir=LR; bgcolor=transparent; " + NODE_STYLE + " edge [color=gray55];",
           'txt [label="Đoạn văn", shape=note, style=filled];', dims, sty,
           f'fu [label="{fuse_label[tr["fusion"]]}\\n{c["fused_dim"]} số"];',
           f'cls [label="Chấm điểm\\n{n_auth} tác giả", fillcolor="#FFD666"];']
    if dims:
        dot.append("txt -> lm -> fu;")
    if sty:
        dot.append("txt -> sty -> fu;")
    dot += ["fu -> cls; }"]
    st.graphviz_chart("\n".join(dot))
    if "gate_mean" in c:
        st.caption(f"Mô hình dựa {_pct(c['gate_mean'])} vào nội dung, {_pct(1 - c['gate_mean'])} vào cách viết.")

    d = decide(c["probs"], tr["names"], candidates)
    pool = [j for j in np.argsort(-c["logits"]) if d["keep"][j]]
    idx = pool[:PREVIEW_AUTHORS]
    st.markdown(f"{len(idx)} tác giả được điểm cao nhất:")
    st.altair_chart(_contrib_chart(_contrib_frame(tr, c, idx, nice), [nice(tr["names"][j]) for j in idx]),
                    width="stretch")
    if len(pool) > len(idx):
        with st.expander(f"Xem đủ {len(pool)} tác giả"):
            st.altair_chart(_contrib_chart(_contrib_frame(tr, c, pool, nice), [nice(tr["names"][j]) for j in pool]),
                            width="stretch")
    if len(idx) >= 2:
        a, b = idx[0], idx[1]
        diff = {k2: float(v[a] - v[b]) for k2, v in c["contrib"].items()}
        fixed = float(c["intercept"][a] + tr["class_bias"][a] - c["intercept"][b] - tr["class_bias"][b])
        short = {"lm": "nội dung", "style": "cách viết"}
        parts = [f"{short[k2]} {'+' if v >= 0 else '−'}{_num(abs(v))}" for k2, v in diff.items()]
        parts.append(f"điểm cộng sẵn {'+' if fixed >= 0 else '−'}{_num(abs(fixed))}")
        st.markdown(f"{nice(tr['names'][a])} hơn {nice(tr['names'][b])} "
                    f"{_num(float(c['logits'][a] - c['logits'][b]))} điểm ({'; '.join(parts)}).")
    st.caption("Điểm âm cũng không sao: điều quan trọng là ai cao hơn ai.")
    if tr.get("ensemble"):
        st.caption(f"Đây là điểm của mô hình lai chính. Bước 6 cho thấy máy gộp nó với "
                   f"{tr['ensemble']['n_hybrid'] - 1} lần huấn luyện khác và bộ đếm cụm ký tự như thế nào.")


def _stage_prob(tr: dict, nice: Callable, candidates) -> None:
    n = len(tr["chunks"])
    _header("prob", "Điểm được đổi thành phần trăm, cộng tất cả tác giả lại bằng 100%. Điểm càng cao thì phần trăm càng "
                    "lớn. Máy đã được chỉnh để con số này sát với khả năng đoán đúng thật."
                    + (f" Văn bản có {n} đoạn nên lấy trung bình {n} đoạn." if n > 1 else ""))
    p = tr["probs"]
    names = tr["names"]
    ens = tr.get("ensemble")
    if ens and "members" in tr["chunks"][0]:
        w = ens["weights"]
        st.markdown(f"Máy hỏi ý kiến nhiều bộ đoán rồi gộp lại theo trọng số (học trên dữ liệu kiểm tra riêng). "
                    f"Mô hình lai được huấn luyện {ens['n_hybrid']} lần với khởi đầu khác nhau, lấy trung bình.")
        mem = {k: np.mean([c["members"][k] for c in tr["chunks"]], axis=0) for k in tr["chunks"][0]["members"]}
        top = np.argsort(-p)[:5]
        rows = [{"Bộ đoán": MEMBER_NAMES.get(k, k), "Trọng số": _pct(w.get(k, 0.0)),
                 **{nice(names[j]): _pct(v[j]) for j in top}} for k, v in mem.items()]
        rows.append({"Bộ đoán": "Gộp lại", "Trọng số": "", **{nice(names[j]): _pct(p[j]) for j in top}})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    k = min(10, len(names))
    order = np.argsort(-p)
    df = pd.DataFrame({"Tác giả": [nice(names[j]) for j in order], "Xác suất": p[order]})
    st.markdown(f"{k} tác giả cao nhất trong cả {len(names)} người (chưa lọc theo thể loại):")
    st.altair_chart(_hbar(df.head(k), "Xác suất", "Tác giả", fmt=".0%", domain=[0, 1]), width="stretch")
    if n > 1:
        top = order[:5]
        st.markdown("Từng đoạn:")
        st.dataframe(pd.DataFrame([{"Đoạn": i + 1, **{nice(names[j]): _pct(c["probs"][j]) for j in top}}
                                   for i, c in enumerate(tr["chunks"])]), hide_index=True, width="stretch")
    with st.expander(f"Xem đủ {len(names)} tác giả"):
        c0 = tr["chunks"][0]
        full = pd.DataFrame({"Tác giả": [nice(names[j]) for j in order], "Xác suất": [_pct(p[j]) for j in order]})
        if n == 1 and not ens:
            full.insert(1, "Điểm", [round(float(c0["logits"][j]), 3) for j in order])
            full.insert(2, "Trước khi chỉnh", [_pct(c0["probs_uncalibrated"][j]) for j in order])
        st.dataframe(full, hide_index=True, width="stretch")


def _stage_decide(tr: dict, nice: Callable, candidates, genre_text: str, threshold: float, true_author) -> dict:
    d = decide(tr["probs"], tr["names"], candidates, threshold)
    n_keep, n_all = int(d["keep"].sum()), len(tr["names"])
    _header("decide", "Máy chỉ giữ các tác giả viết thể loại bạn chọn, tính lại phần trăm trong nhóm đó rồi chọn người "
                      "cao nhất. Nếu người cao nhất vẫn dưới ngưỡng tin cậy, máy trả lời “Không xác định”.")
    if n_keep < n_all:
        st.markdown(f"Bạn chọn {genre_text}, còn {n_keep}/{n_all} tác giả.")
    else:
        st.markdown(f"Bạn chọn {genre_text}, máy xét cả {n_all} tác giả.")
    names = tr["names"]
    a, b = st.columns(2)
    with a:
        st.markdown("Trước khi lọc")
        o = np.argsort(-d["probs_before"])[:5]
        st.dataframe(pd.DataFrame({"Tác giả": [nice(names[j]) + ("" if d["keep"][j] else " (bị loại)") for j in o],
                                   "Xác suất": [_pct(d["probs_before"][j]) for j in o]}), hide_index=True, width="stretch")
    with b:
        st.markdown("Sau khi lọc")
        o = [j for j in np.argsort(-d["probs_after"]) if d["keep"][j]][:5]
        st.dataframe(pd.DataFrame({"Tác giả": [nice(names[j]) for j in o],
                                   "Xác suất": [_pct(d["probs_after"][j]) for j in o]}), hide_index=True, width="stretch")
    if d["best_without_filter"] != d["best"]:
        st.caption(f"Nếu không lọc theo thể loại, máy sẽ chọn {nice(d['best_without_filter'])}.")
    if threshold:
        verb = "cao hơn" if not d["unknown"] else "thấp hơn"
        st.caption(f"Người đứng đầu được {_pct(d['confidence'])}, {verb} ngưỡng tin cậy {_pct(threshold)}.")
    if d["unknown"]:
        st.warning(f"**Kết luận: Không xác định.** Gần nhất là {nice(d['best'])} nhưng chỉ {_pct(d['confidence'])}.")
    else:
        st.success(f"**Kết luận: {nice(d['author'])}** ({_pct(d['confidence'])})")
    if true_author:
        if true_author not in names:
            st.caption(f"Tác giả thật là {nice(true_author)}, người máy chưa được học.")
        elif d["author"] == true_author:
            st.markdown(f"Máy đoán **đúng**: đoạn này của {nice(true_author)}.")
        else:
            st.markdown(f"Máy đoán **sai**: đoạn này của {nice(true_author)} "
                        f"(máy chỉ cho {_pct(d['probs_after'][names.index(true_author)])}).")
    return d


def _stage_profile(tr: dict, c: dict, nice: Callable, candidates, d: dict, ckpt: str, meta: dict, extractor) -> None:
    _header("profile", "Đặt các thói quen viết ở bước 4 cạnh thói quen trung bình của một tác giả để thấy văn bản giống "
                       "người đó ở điểm nào. Phần này chỉ để giải thích, vì khi đoán máy còn dùng cả PhoBERT.")
    f = data_file(meta)
    if f is None:
        st.write("Không tìm thấy dữ liệu đã dùng để dạy máy nên chưa so được.")
        return
    names_m = tuple(tr["names"])
    p_names, means, counts = _profiles(ckpt, str(f), f.stat().st_mtime, extractor, names_m)
    if not p_names:
        st.write("Dữ liệu không có đoạn nào của các tác giả này.")
        return
    pos = {a: i for i, a in enumerate(p_names)}
    pool = [a for a in np.array(tr["names"])[np.argsort(-d["probs_after"])] if d["keep"][tr["names"].index(a)] and a in pos]
    if not pool:
        st.write("Chưa có dữ liệu của các tác giả đang xét.")
        return
    who = st.selectbox("So với", pool, format_func=lambda a: f"{nice(a)} ({_pct(d['probs_after'][tr['names'].index(a)])})",
                       key="profile_author")
    z, raw, mean, std = c["style_z"], c["style_raw"], tr["corpus_mean"], tr["corpus_std"]
    prof = means[pos[who]]
    order = np.argsort(-np.abs(z))[:8]
    rows = []
    for i in order:
        closer = abs(raw[i] - prof[i]) < abs(raw[i] - mean[i])
        rows.append({"Thói quen nổi bật": feature_label(tr["feature_names"][i]), "Văn bản này": round(float(raw[i]), 3),
                     nice(who): round(float(prof[i]), 3), "Trung bình": round(float(mean[i]), 3),
                     f"Giống {nice(who)}?": "có" if closer else "không"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    n_yes = sum(r[f"Giống {nice(who)}?"] == "có" for r in rows)
    st.markdown(f"Văn bản giống {nice(who)} ở {n_yes}/{len(rows)} thói quen nổi bật nhất "
                f"(so với {counts[pos[who]]} đoạn của tác giả này).")

    cand = [a for a in pool]
    dist = style_distance(z, np.stack([means[pos[a]] for a in cand]), mean, std)
    o = np.argsort(dist)[:10]
    df = pd.DataFrame({"Tác giả": [nice(cand[i]) for i in o], "Khoảng cách phong cách": dist[o]})
    st.markdown("Nếu chỉ dựa vào cách viết, ai giống nhất (thanh càng ngắn càng giống):")
    st.altair_chart(_hbar(df, "Khoảng cách phong cách", "Tác giả", fmt=".2f"), width="stretch")
    st.caption("Thứ tự này có thể khác kết luận ở bước 7, vì máy còn dùng cả PhoBERT.")


# ------------------------------------------------------------------ toàn tab
def render(pred, ckpt: str, meta: dict, candidates: list[str], genre_text: str, threshold: float,
           nice: Callable, need_genre: bool, run_trace: Callable) -> None:
    st.markdown("### Máy đoán tác giả như thế nào?")
    st.markdown("Đưa vào một đoạn văn rồi chọn các bước muốn xem. Mọi con số đều là của chính lần đoán này.")
    note = unknown_authors_note(meta, pred.names)
    if note:
        st.warning(note)

    # ---- 1. văn bản
    def take_sample():
        f = data_file(meta)
        if f is None:
            return
        part = _test_part(str(f), f.stat().st_mtime, meta["config"], meta["data"].get("test_topics"))
        part = part[part["author"].isin(candidates or pred.names)]
        if part.empty:
            return
        r = part.iloc[random.randrange(len(part))]
        st.session_state["trace_text"] = r["text"]
        st.session_state["trace_sample"] = {"text": r["text"], "author": r["author"], "topic": r.get("topic", ""),
                                            "unseen": len(_load_data(str(f), f.stat().st_mtime)) == meta["data"].get("n_docs")}

    def take_main():
        st.session_state["trace_text"] = st.session_state.get("main_text", "")

    left, right = st.columns([3, 1])
    with right:
        st.button("Lấy một đoạn mẫu", on_click=take_sample, width="stretch", disabled=data_file(meta) is None,
                  help="Lấy ngẫu nhiên một đoạn máy chưa từng học, đúng thể loại đang chọn. Có đáp án để xem máy đoán "
                       "đúng hay sai.")
        st.button("Dùng văn bản ở tab Nhập văn bản", on_click=take_main, width="stretch",
                  disabled=not st.session_state.get("main_text", "").strip())
    with left:
        text = st.text_area("Văn bản", key="trace_text", height=160,
                            placeholder="Dán đoạn văn vào đây, hoặc bấm “Lấy một đoạn mẫu”…")
    sample = st.session_state.get("trace_sample")
    true_author = None
    if sample and sample["text"] == text:
        true_author = sample["author"]
        seen = "" if sample["unseen"] else \
            " Lưu ý: dữ liệu đã thay đổi sau khi dạy máy nên đoạn này có thể máy đã từng học."
        with st.expander("Xem đáp án (thử tự đoán trước)"):
            st.markdown(f"{nice(sample['author'])}, trích từ *{sample['topic']}*.{seen}")

    # ---- 2. chọn giai đoạn
    st.markdown("Chọn các bước muốn xem (bấm lần nữa để tắt):")
    labels = {k: f"{n} {lab}" for k, (n, lab) in STAGES.items()}
    uses_lm = meta["config"].get("fusion") != "style_only"
    options = [k for k in STAGES if uses_lm or k != "token"]
    if "trace_stages" not in st.session_state:
        st.session_state["trace_stages"] = [k for k in DEFAULT_STAGES if k in options]
    picked = st.pills("Các bước", options, selection_mode="multi", format_func=lambda k: labels[k],
                      label_visibility="collapsed", key="trace_stages")
    picked = [k for k in options if k in (picked or [])]
    def set_stages(keys):
        st.session_state["trace_stages"] = keys

    c1, c2 = st.columns([1, 1])
    c1.button("Xem tất cả", width="stretch", on_click=set_stages, args=(options,))
    c2.button("Bỏ chọn hết", width="stretch", on_click=set_stages, args=([],))
    st.graphviz_chart(_flow_dot(picked, uses_lm))

    if not text.strip():
        st.info("Hãy dán một đoạn văn hoặc bấm “Lấy một đoạn mẫu” để bắt đầu.")
        return
    if not picked:
        st.info("Hãy chọn ít nhất một bước để xem.")
        return
    if not candidates:
        st.error("Không còn tác giả nào để so. Hãy bỏ bớt lựa chọn ở thanh bên.")
        return
    if need_genre:
        st.warning("Bạn chưa chọn thể loại ở thanh bên nên máy đang xét tất cả tác giả.")
    tr = run_trace(ckpt, text)

    ci = 0
    if len(tr["chunks"]) > 1 and set(picked) & {"token", "style", "fusion", "profile"}:
        ci = st.selectbox(f"Văn bản được chia làm {len(tr['chunks'])} đoạn. Xem chi tiết đoạn số:",
                          range(len(tr["chunks"])), format_func=lambda i: f"{i + 1} ({tr['chunks'][i]['n_words']} từ)",
                          key="trace_chunk")
    c = tr["chunks"][ci]
    d = None
    for k in picked:
        with st.container(border=True):
            if k == "doc":
                _stage_doc(tr)
            elif k == "chunk":
                _stage_chunk(tr, nice, candidates)
            elif k == "token":
                _stage_token(tr, c)
            elif k == "style":
                _stage_style(tr, c)
            elif k == "fusion":
                _stage_fusion(tr, c, nice, candidates)
            elif k == "prob":
                _stage_prob(tr, nice, candidates)
            elif k == "decide":
                d = _stage_decide(tr, nice, candidates, genre_text, threshold, true_author)
            elif k == "profile":
                _stage_profile(tr, c, nice, candidates, d or decide(tr["probs"], tr["names"], candidates, threshold),
                               ckpt, meta, pred.extractor)
