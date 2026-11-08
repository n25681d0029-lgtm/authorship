"""Tab "So sánh hai văn bản": hai đoạn có cùng một người viết không?

Không cần biết trước tác giả: máy đoán phân bố tác giả của từng đoạn (trong các tác giả đang xét), đo độ giống hai phân
bố (src/compare.py) và đối chiếu với các cặp đoạn trong tập kiểm tra đã biết đáp án.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import streamlit as st

import app_theme as ui
from src.compare import lookup, pair_calibration, prob_vectors, similarity
from src.history import History


@st.cache_data(show_spinner="Đang đối chiếu với các cặp đoạn kiểm tra (chỉ lần đầu)…")
def _calibration(ckpt: str, cands: tuple, _pred, _test_texts: tuple, test_authors: tuple) -> list:
    probs = prob_vectors(_pred, list(_test_texts), list(cands))
    return pair_calibration(probs, test_authors)


def render(pred, ckpt: str, candidates: list[str], genre_text: str, nice: Callable, test_part, hist: History) -> None:
    st.markdown("### Hai văn bản có cùng một người viết?")
    st.caption(f"Máy đoán tác giả của từng văn bản ({genre_text}) rồi so hai kết quả. Dùng được cả khi người viết không "
               "nằm trong danh sách máy biết, nhưng khi đó kết quả kém chắc chắn hơn.")
    c = st.columns(2)
    a = c[0].text_area("Văn bản A", key="cmp_a", height=200)
    b = c[1].text_area("Văn bản B", key="cmp_b", height=200)
    if not st.button("So sánh", type="primary", key="cmp_go", disabled=not (a.strip() and b.strip())):
        return
    P = prob_vectors(pred, [a, b], candidates)
    s = similarity(P[0], P[1])
    calib = []
    if test_part is not None:
        tp = test_part[test_part["author"].isin(candidates)]
        if len(tp) >= 30:
            calib = _calibration(ckpt, tuple(candidates), pred, tuple(tp["text"]), tuple(tp["author"]))
    b_ = lookup(calib, s) if calib else None
    left, right = st.columns([1, 1.4])
    with left:
        if b_:
            rate = b_["ty_le_cung"]
            verdict = "Nhiều khả năng cùng người viết" if rate >= 0.7 else (
                "Nhiều khả năng khác người viết" if rate <= 0.3 else "Chưa nói chắc được")
            ui.result_card("Kết luận", verdict, f"độ giống {100 * s:.0f}% · {100 * rate:.0f}% cặp đoạn kiểm tra giống cỡ "
                                                 f"này là cùng tác giả".replace(".", ","), unknown=0.3 < rate < 0.7)
            st.caption(f"Đối chiếu {b_['n']} cặp đoạn kiểm tra có độ giống {100 * b_['lo']:.0f}–{100 * b_['hi']:.0f}%, "
                       "giả định trước khi so khả năng cùng tác giả là 50/50.")
        else:
            ui.result_card("Độ giống", f"{100 * s:.0f}%", "chưa có dữ liệu kiểm tra để đối chiếu")
    with right:
        names = pred.names
        for lab, p in (("A", P[0]), ("B", P[1])):
            o = np.argsort(-p)[:3]
            st.markdown(f"**Văn bản {lab}** giống nhất:")
            ui.prob_bars([(nice(names[j]), float(p[j])) for j in o])
    if calib:
        with st.expander("Độ giống bao nhiêu thì thường là cùng tác giả?"):
            import pandas as pd
            st.bar_chart(pd.DataFrame({"Độ giống": [f"{100 * x['lo']:.0f}–{100 * x['hi']:.0f}%" for x in calib],
                                       "Tỷ lệ cùng tác giả": [x["ty_le_cung"] for x in calib]}),
                         x="Độ giống", y="Tỷ lệ cùng tác giả")
    top = [(names[j], float(P[0][j])) for j in np.argsort(-P[0])[:3]]
    hist.add("so_sanh", f"So sánh: độ giống {100 * s:.0f}%", a + "\n=====\n" + b, {"giong": s, "top": top,
                                                                               "genre_text": genre_text})
