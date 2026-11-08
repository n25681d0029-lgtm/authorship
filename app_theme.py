"""Giao diện: CSS (hiệu ứng chuyển động nhẹ, thẻ kết quả, thanh xác suất động) và vài khối HTML dùng chung.

Chỉ là CSS/HTML tĩnh chèn vào trang Streamlit — không gọi dịch vụ ngoài. Màu dùng độ trong suốt để hợp cả nền sáng
lẫn nền tối của Streamlit. Người dùng có thể tắt hiệu ứng ở thanh bên (prefers-reduced-motion cũng được tôn trọng).
"""
from __future__ import annotations

import html

import streamlit as st

ACCENT = "#E8823A"
ACCENT2 = "#4C78A8"

CSS = """
<style>
:root { --acc: %(acc)s; --acc2: %(acc2)s; }
/* tận dụng chiều ngang màn hình rộng */
.block-container { padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1500px; }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* tiêu đề có dải màu chuyển động chậm */
.app-hero { border-radius: 16px; padding: 18px 22px; margin-bottom: 10px;
  background: linear-gradient(120deg, rgba(232,130,58,.18), rgba(76,120,168,.18), rgba(232,130,58,.10));
  background-size: 300%% 300%%; animation: heroShift 14s ease infinite; }
.app-hero h1 { font-size: 1.85rem; margin: 0; letter-spacing: .2px; }
.app-hero p { margin: 4px 0 0 0; opacity: .8; }
@keyframes heroShift { 0%% {background-position: 0%% 50%%} 50%% {background-position: 100%% 50%%} 100%% {background-position: 0%% 50%%} }

/* nút: nổi lên khi rê chuột, lún xuống khi bấm */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  transition: transform .12s ease, box-shadow .2s ease; border-radius: 10px; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(0,0,0,.18); }
.stButton > button:active { transform: translateY(0) scale(.98); }
.stButton > button[kind="primary"] { background: linear-gradient(90deg, var(--acc), #d4692a); border: 0; }

/* tab: gạch chân trượt */
button[data-baseweb="tab"] { transition: color .2s ease; }
div[data-baseweb="tab-highlight"] { background-color: var(--acc) !important; transition: all .35s ease; }

/* khối có viền hiện dần */
div[data-testid="stVerticalBlockBorderWrapper"] { animation: fadeUp .45s ease both; border-radius: 14px; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(8px);} to { opacity: 1; transform: none;} }

/* thẻ kết quả chính */
.result-card { border-radius: 16px; padding: 18px 20px; animation: popIn .5s cubic-bezier(.2,.9,.3,1.2) both;
  background: linear-gradient(135deg, rgba(232,130,58,.16), rgba(76,120,168,.10));
  border: 1px solid rgba(232,130,58,.35); }
.result-card .label { font-size: .85rem; opacity: .75; text-transform: uppercase; letter-spacing: .6px; }
.result-card .name { font-size: 2rem; font-weight: 700; margin: 2px 0 6px 0; }
.result-card .conf { font-size: .95rem; opacity: .85; }
.result-card.unknown { background: linear-gradient(135deg, rgba(150,150,150,.18), rgba(120,120,120,.08));
  border-color: rgba(150,150,150,.4); }
@keyframes popIn { from { opacity: 0; transform: scale(.94);} to { opacity: 1; transform: none;} }

/* thanh xác suất động */
.pbar { margin: 7px 0; }
.pbar .row { display: flex; justify-content: space-between; font-size: .92rem; margin-bottom: 3px; }
.pbar .track { height: 12px; border-radius: 8px; background: rgba(128,128,128,.18); overflow: hidden; }
.pbar .fill { height: 100%%; border-radius: 8px;
  background: linear-gradient(90deg, var(--acc2), var(--acc)); animation: grow 1s cubic-bezier(.2,.8,.2,1) both; }
.pbar.first .fill { background: linear-gradient(90deg, var(--acc), #f2a65a); }
@keyframes grow { from { width: 0; } }

/* câu được tô màu theo mức ảnh hưởng */
.hl { line-height: 1.95; font-size: 1.0rem; }
.hl span.s { border-radius: 5px; padding: 1px 2px; transition: background .2s ease; }
.hl span.s:hover { outline: 1px solid rgba(232,130,58,.7); }

/* chip từ ngữ */
.chips { line-height: 2.1; }
.chip { display: inline-block; padding: 2px 10px; margin: 2px 4px 2px 0; border-radius: 999px; font-size: .9rem;
  background: rgba(76,120,168,.16); border: 1px solid rgba(76,120,168,.35); animation: fadeUp .4s ease both; }
.chip.warm { background: rgba(232,130,58,.16); border-color: rgba(232,130,58,.4); }
.chip.gray { background: rgba(128,128,128,.14); border-color: rgba(128,128,128,.35); }

/* ô số liệu nhỏ */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.stat { border-radius: 12px; padding: 10px 12px; background: rgba(128,128,128,.10); animation: fadeUp .4s ease both; }
.stat .k { font-size: .8rem; opacity: .7; }
.stat .v { font-size: 1.25rem; font-weight: 650; }
.stat .d { font-size: .8rem; opacity: .7; }

/* chấm trạng thái chờ */
.dots span { display: inline-block; width: 7px; height: 7px; margin: 0 2px; border-radius: 50%%; background: var(--acc);
  animation: bounce 1s infinite ease-in-out; }
.dots span:nth-child(2) { animation-delay: .15s; } .dots span:nth-child(3) { animation-delay: .3s; }
@keyframes bounce { 0%%, 80%%, 100%% { transform: scale(.4); opacity: .5;} 40%% { transform: scale(1); opacity: 1;} }

@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; }
 }
</style>
"""
NO_MOTION = "<style>* { animation: none !important; transition: none !important; }</style>"


def inject(motion: bool = True) -> None:
    st.html(CSS % {"acc": ACCENT, "acc2": ACCENT2} + ("" if motion else NO_MOTION))


def hero(title: str, subtitle: str) -> None:
    st.html(f'<div class="app-hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>')


def result_card(label: str, name: str, conf_text: str, unknown: bool = False) -> None:
    st.html(f'<div class="result-card{" unknown" if unknown else ""}"><div class="label">{html.escape(label)}</div>'
            f'<div class="name">{html.escape(name)}</div><div class="conf">{html.escape(conf_text)}</div></div>')


def prob_bars(rows: list[tuple[str, float]]) -> None:
    """rows: (tên, xác suất 0..1), đã sắp xếp. Thanh đầu tiên màu nổi bật."""
    out = []
    for i, (name, p) in enumerate(rows):
        pct = f"{100 * p:.1f}%".replace(".", ",")
        out.append(f'<div class="pbar{" first" if i == 0 else ""}"><div class="row"><span>{html.escape(name)}</span>'
                   f'<b>{pct}</b></div><div class="track"><div class="fill" style="width:{max(0.5, 100 * p):.1f}%;'
                   f'animation-delay:{0.08 * i:.2f}s"></div></div></div>')
    st.html("".join(out))


def chips(words, kind: str = "") -> str:
    return '<div class="chips">' + "".join(f'<span class="chip {kind}" style="animation-delay:{0.03 * i:.2f}s">'
                                           f'{html.escape(str(w))}</span>' for i, w in enumerate(words)) + "</div>"


def stat_grid(items: list[tuple[str, str, str]]) -> None:
    """items: (nhãn, giá trị, ghi chú nhỏ)."""
    st.html('<div class="stat-grid">' + "".join(
        f'<div class="stat" style="animation-delay:{0.05 * i:.2f}s"><div class="k">{html.escape(k)}</div>'
        f'<div class="v">{html.escape(v)}</div><div class="d">{html.escape(d)}</div></div>'
        for i, (k, v, d) in enumerate(items)) + "</div>")


def highlight(sentences: list[tuple[str, float]]) -> None:
    """Tô màu từng câu: dương (cam) = làm máy nghiêng về tác giả được chọn, âm (xanh) = kéo ra xa. Giá trị đã chuẩn hóa -1..1."""
    parts = []
    for s, v in sentences:
        a = min(1.0, abs(v)) * 0.55
        col = f"rgba(232,130,58,{a:.2f})" if v > 0 else f"rgba(76,120,168,{a:.2f})"
        parts.append(f'<span class="s" style="background:{col}" title="{v:+.2f}">{html.escape(s)}</span> ')
    st.html('<div class="hl">' + "".join(parts) + "</div>")
