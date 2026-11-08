"""Tab "Lịch sử": xem lại các lần dùng theo tuần / tháng / năm, đánh dấu yêu thích, ghi chú, mở lại, xóa.

Dữ liệu ở outputs/app_data/lich_su.db (src/history.py). Mục không yêu thích tự xóa sau số ngày người dùng chọn.
"""
from __future__ import annotations

from itertools import groupby
from typing import Callable

import streamlit as st

from src.history import KEEP_CHOICES, KINDS, History, group_label

PAGE = 15


def _open_entry(e: dict) -> None:
    """Đưa văn bản của mục về đúng tab để dùng lại."""
    if e["kind"] in ("dinh_danh",):
        st.session_state["main_text"] = e["text"]
        st.session_state["_flash"] = "Đã đưa văn bản về tab Đoán tác giả."
    elif e["kind"] == "kiem_tra":
        st.session_state["sg_draft"] = e["text"]
        st.session_state["_flash"] = "Đã đưa bài viết về tab Học viết theo tác giả › Kiểm tra bài viết."
    elif e["kind"] == "so_sanh":
        a, _, b = e["text"].partition("\n=====\n")
        st.session_state["cmp_a"], st.session_state["cmp_b"] = a, b
        st.session_state["_flash"] = "Đã đưa hai văn bản về tab So sánh hai văn bản."
    else:
        st.session_state["main_text"] = e["text"]
        st.session_state["_flash"] = "Đã đưa bài máy viết về tab Đoán tác giả (để thử xem máy có nhận ra không)."


def render(hist: History, nice: Callable) -> None:
    s = hist.stats()
    top = st.columns([2, 1.2, 1.2, 1.4])
    with top[0]:
        st.markdown("### Lịch sử của bạn")
        st.caption("Lưu ngay trên máy này, không gửi đi đâu. Bấm ☆ để giữ một mục mãi mãi; mục khác tự xóa sau một thời gian.")
    top[1].metric("Đang lưu", s["total"])
    top[2].metric("Yêu thích", s["favorites"])
    with top[3]:
        keep = st.selectbox("Tự xóa mục không yêu thích sau", list(KEEP_CHOICES), format_func=KEEP_CHOICES.get,
                            index=list(KEEP_CHOICES).index(hist.keep_days) if hist.keep_days in KEEP_CHOICES else 1,
                            key="hist_keep")
        if keep != hist.keep_days:
            hist.keep_days = keep
            n = hist.cleanup()
            st.toast(f"Đã đổi thời hạn lưu. Xóa {n} mục quá hạn." if n else "Đã đổi thời hạn lưu.")
        if s["next_expire"] and hist.keep_days:
            st.caption(f"Mục cũ nhất sẽ tự xóa vào {s['next_expire']:%d/%m/%Y}.")

    f = st.columns([1.3, 1.3, 1, 2])
    period = f[0].segmented_control("Khoảng thời gian", ["week", "month", "year", "all"], default="month",
                                    format_func={"week": "7 ngày", "month": "30 ngày", "year": "1 năm",
                                                 "all": "Tất cả"}.get, key="hist_period") or "all"
    group_by = f[1].segmented_control("Gom theo", ["day", "week", "month", "year"], default="day",
                                      format_func={"day": "Ngày", "week": "Tuần", "month": "Tháng", "year": "Năm"}.get,
                                      key="hist_group") or "day"
    kind = f[2].selectbox("Loại", [None, *KINDS], format_func=lambda k: "Tất cả" if k is None else KINDS[k],
                          key="hist_kind")
    q = f[3].text_input("Tìm", placeholder="tên tác giả, một chữ trong văn bản, ghi chú…", key="hist_q")
    fav_only = st.toggle("Chỉ hiện mục yêu thích", key="hist_fav")

    rows = hist.list(period, kind, fav_only, q)
    if not rows:
        st.info("Chưa có gì ở đây. Mỗi lần bạn đoán tác giả, kiểm tra bài viết hay cho máy viết thử, kết quả sẽ được ghi lại.")
        return
    shown = st.session_state.get("hist_shown", PAGE)
    for label, items in groupby(rows[:shown], key=lambda e: group_label(e["created"], group_by)):
        items = list(items)
        st.markdown(f"**{label}** · {len(items)} mục")
        for e in items:
            _entry(hist, e, nice)
    if len(rows) > shown:
        if st.button(f"Xem thêm ({len(rows) - shown} mục nữa)", key="hist_more"):
            st.session_state["hist_shown"] = shown + PAGE
            st.rerun()
    with st.expander("Dọn dẹp"):
        c = st.columns(2)
        if c[0].button("Xóa mọi mục không yêu thích", key="hist_clear"):
            st.toast(f"Đã xóa {hist.clear(True)} mục.")
            st.rerun()
        if c[1].button("Xóa tất cả, kể cả yêu thích", key="hist_clear_all"):
            st.session_state["hist_confirm"] = True
        if st.session_state.get("hist_confirm"):
            st.warning("Xóa hết không lấy lại được.")
            if st.button("Đồng ý xóa hết", key="hist_clear_yes", type="primary"):
                hist.clear(False)
                st.session_state["hist_confirm"] = False
                st.rerun()


def _entry(hist: History, e: dict, nice: Callable) -> None:
    star = "★" if e["favorite"] else "☆"
    head = f"{star}  {e['created']:%H:%M %d/%m} · {KINDS.get(e['kind'], e['kind'])} · {e['title']}"
    with st.expander(head):
        d = e["detail"]
        c = st.columns([3, 1])
        with c[0]:
            st.text(e["text"][:1500] + ("…" if len(e["text"]) > 1500 else ""))
        with c[1]:
            if d.get("top"):
                for a, p in d["top"][:5]:
                    st.markdown(f"{nice(a)}: **{100 * p:.1f}%**".replace(".", ","))
            if d.get("genre_text"):
                st.caption(d["genre_text"])
            for k in ("giong", "chep", "tu_khoa"):
                if k in d:
                    st.caption({"giong": "Giống tác giả", "chep": "Chép nguyên văn", "tu_khoa": "Có từ khóa"}[k]
                               + f": {100 * d[k]:.1f}%".replace(".", ","))
        b = st.columns([1, 1, 1, 3])
        if b[0].button("Bỏ yêu thích" if e["favorite"] else "Yêu thích", key=f"fav_{e['id']}"):
            hist.set_favorite(e["id"], not e["favorite"])
            st.rerun()
        b[1].button("Dùng lại", key=f"open_{e['id']}", on_click=_open_entry, args=(e,))   # callback: chạy trước khi vẽ ô nhập
        if b[2].button("Xóa", key=f"del_{e['id']}"):
            hist.delete(e["id"])
            st.rerun()
        note = b[3].text_input("Ghi chú", value=e["note"], key=f"note_{e['id']}", label_visibility="collapsed",
                               placeholder="Ghi chú cho mục này…")
        if note != e["note"]:
            hist.set_note(e["id"], note)
