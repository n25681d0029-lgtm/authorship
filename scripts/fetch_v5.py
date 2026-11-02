#!/usr/bin/env python3
"""Kho dữ liệu v5 theo thể loại: khảo sát nguồn, duyệt nhãn thể loại, rồi tải văn bản.

Nguồn (đều thuộc phạm vi công cộng, chỉ tác giả mất trước 1976):
  ws        trang văn bản thường trên vi.wikisource.org (như kho v1–v4)
  ws_trang  các trang "Trang:" của bản chụp sách (Mục lục:…) ĐÃ ĐƯỢC SOÁT (chất lượng ≥ 3) — bản chép tay của
            người đóng góp, không phải lớp chữ OCR của PDF; trang chưa soát bị bỏ.
  local     văn bản bạn tự đặt vào data/v5_genre/raw/<thể loại>/<tác giả>/<tác phẩm>/*.txt (vd. sách Project
            Gutenberg, bản đánh máy tự làm). Script không tải gì cho nguồn này; nhớ ghi nguồn/giấy phép
            vào data/v5_genre/raw/manifest.csv.

Ba bước:
  1) python scripts/fetch_v5.py survey
       Liệt kê tác phẩm của mọi tác giả ứng viên (data/v5_sources.json + data/raw/extra_authors_30.json),
       đề xuất thể loại cho từng tác phẩm, ghi data/v5_works.csv. Chạy lại chỉ khảo sát tác giả chưa có
       (thêm --refresh để làm lại). Kết quả gọi API được lưu đệm ở data/v5_genre/cache.
  2) Duyệt data/v5_works.csv: cột genre là nhãn sẽ dùng (tho | tu_su | ky | nghi_luan | bo | để trống).
       Đã điền sẵn bằng đề xuất; sửa những dòng sai, điền những dòng "?" (để trống = bỏ qua). Quy tắc gán:
       data/genres.json. Với bản chụp (nguồn ws_trang) gộp nhiều tác phẩm, ghi khoảng trang cần lấy vào cột
       "trang" (vd. 11-120); để trống = lấy mọi trang đã soát.
  3) python scripts/fetch_v5.py fetch
       Tải các tác phẩm có genre hợp lệ vào data/v5_genre/raw/<thể loại>/<tác giả>/<tác phẩm>/*.txt,
       ghi manifest (data/v5_genre/raw/manifest.csv, bản tóm tắt ở data/v5_manifest_summary.csv).
  Sau đó: python main.py build-genre-corpus   -> data/v5_corpus.csv (+ .stats.json)

Tùy chọn chung: --only <slug ...> chỉ xử lý vài tác giả; --sleep giãn cách giữa các lần gọi API (mặc định 1 giây).
Kết quả API được lưu đệm trong data/v5_genre/cache: xóa thư mục này nếu muốn lấy lại dữ liệu mới từ Wikisource.
Cần: pip install beautifulsoup4
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, RAW_DATA_DIR  # noqa: E402

API = "https://vi.wikisource.org/w/api.php"
UA = {"User-Agent": "authorship-nlp-student/0.5 (university NLP coursework; contact via project owner)"}
V5 = DATA_DIR / "v5_genre"
RAW = V5 / "raw"
CACHE = V5 / "cache"
WORKS_CSV = DATA_DIR / "v5_works.csv"
SOURCES_JSON = DATA_DIR / "v5_sources.json"
GENRES_JSON = DATA_DIR / "genres.json"
V4_EXTRA = RAW_DATA_DIR / "extra_authors_30.json"

GENRES = ("tho", "tu_su", "ky", "nghi_luan")
MAX_WORKS = 120                # số tác phẩm tối đa khảo sát mỗi tác giả
MAX_PAGES_PER_WORK = 8         # mỗi tác phẩm lấy tối đa vài chương để kho đa dạng tác phẩm
TRANG_PAGES_PER_FILE = 10      # nguồn ws_trang: gộp 10 trang sách đã soát thành một file (một "bài")
WORD_BUDGET = {"tho": 30_000, "tu_su": 60_000, "ky": 60_000, "nghi_luan": 60_000}   # âm tiết / (tác giả, thể loại)
MIN_PAGE_WORDS = {"tho": 20, "tu_su": 40, "ky": 40, "nghi_luan": 40}
MIN_QUALITY = 3                # ProofreadPage: 0 không có chữ, 1 chưa soát, 2 có vấn đề, 3 đã soát, 4 đã duyệt
SLEEP = 1.0

WORK_FIELDS = ["author", "author_name", "died", "source", "work", "url", "header", "ok", "n_subpages",
               "words_main", "poem_ratio", "categories", "de_xuat", "ly_do", "genre", "ghi_chu", "trang"]
MANIFEST_FIELDS = ["genre", "author", "author_name", "died", "source", "work", "page", "file", "n_words",
                   "url", "license"]

DROP_SELECTORS = ["#headerContainer", ".ws-noexport", ".licenseContainer", ".licensetpl", "sup", ".reference",
                  ".references", ".mw-references-wrap", "table", ".pagenum", ".ws-pagenum", "style", "script",
                  ".mw-editsection", ".navbox", "#toc", ".toc", ".noprint", ".mw-heading", "h1", "h2", "h3",
                  "h4", "h5", "h6",
                  # số thứ tự dòng thơ do Wikisource chèn (vd. Truyện Kiều: "5 Lạ gì bỉ sắc tư phong") — không phải chữ
                  # của tác giả, giữ lại thì chữ số thành tín hiệu giả
                  ".OptionText", '[title="Số thứ tự"]', '[title="So thu tu"]']


# ====================================================================== hàm thuần (có unittest)
def slugify(s: str) -> str:
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")[:80] or "x"


def _poem_text(el) -> str:
    """Nội dung một khối thơ: mỗi dòng thơ một dòng, bỏ dòng trống (ranh giới khổ phụ thuộc bản chép)."""
    for br in el.find_all("br"):
        br.replace_with("\n")
    lines = [" ".join(x.split()) for x in el.get_text("").split("\n")]
    return "\n".join(x for x in lines if x)


def extract_content(html: str) -> dict:
    """Tách HTML một trang Wikisource thành header tác giả, cờ bản dịch, các bài thơ và các đoạn văn xuôi.

    Thơ trên Wikisource nằm trong khối <div class="poem"> (thẻ <poem>) hoặc đoạn <p> có nhiều <br>.
    Văn xuôi là các <p> còn lại. Hai phần tách riêng để tác phẩm văn xuôi có chèn thơ không lẫn thơ vào kho
    văn xuôi và ngược lại."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    ws_author = soup.select_one("#ws-author")
    header = ws_author.get_text(" ", strip=True) if ws_author else None
    translated = soup.select_one("#header_translator_text, #ws-translator") is not None
    root = soup.select_one(".mw-parser-output") or soup
    for sel in DROP_SELECTORS:
        for el in root.select(sel):
            el.decompose()
    poems = []
    for el in root.select(".poem"):
        t = _poem_text(el)
        if len(re.findall(r"\w+", t)) >= 4:
            poems.append(t)
        el.decompose()
    paras = []
    for p in root.find_all("p"):
        if len(p.find_all("br")) >= 2:          # nhiều dòng ngắt trong một đoạn: thơ không dùng thẻ <poem>
            t = _poem_text(p)
            if len(re.findall(r"\w+", t)) >= 4:
                poems.append(t)
            continue
        t = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
        if len(re.findall(r"\w+", t)) >= 3:
            paras.append(t)
    return {"header": header, "translated": translated, "poems": poems, "paras": paras}


def count_words(texts: list[str]) -> int:
    return sum(len(t.split()) for t in texts)


def poem_ratio(content: dict) -> float:
    a, b = count_words(content["poems"]), count_words(content["paras"])
    return a / (a + b) if a + b else 0.0


def _has_phrase(text: str, phrases: list[str]) -> Optional[str]:
    low = unicodedata.normalize("NFC", text).lower()
    for ph in phrases:
        if re.search(r"(?<!\w)" + re.escape(ph.lower()) + r"(?!\w)", low):
            return ph
    return None


DRAMA_WORDS = ["kịch", "hài kịch", "bi kịch", "tuồng", "chèo"]


def propose_genre(title: str, categories: list[str], ratio: Optional[float], hint: Optional[str],
                  genres_cfg: dict) -> tuple[str, str]:
    """Đề xuất thể loại cho một tác phẩm. Trả về (nhãn, lý do); nhãn '?' nghĩa là cần người duyệt điền.

    Thứ tự: hình thức thơ (đo được) > thể loại Wikisource > từ khóa tiêu đề > gợi ý theo tác giả.
    ratio = None khi chưa đo được tỷ lệ thơ (bản chụp sách): bỏ qua các điều kiện dựa trên tỷ lệ."""
    cfg = genres_cfg["the_loai"]
    if ratio is None:
        unknown, ratio = True, 0.3
    else:
        unknown = False
    if ratio >= 0.6:
        return "tho", f"{ratio:.0%} nội dung là dòng thơ"
    cats = " | ".join(categories)
    if _has_phrase(cats, DRAMA_WORDS) or _has_phrase(title, DRAMA_WORDS):
        return "bo", "kịch (ngoài 4 thể loại)"
    for g in GENRES:
        ph = _has_phrase(cats, cfg[g].get("tu_khoa_the_loai_wikisource", []))
        if ph and (g != "tho" or ratio >= 0.3):
            return g, f"thể loại Wikisource có '{ph}'"
    for g in GENRES:
        ph = _has_phrase(title, cfg[g].get("tu_khoa_tieu_de", []))
        if ph and (g != "tho" or ratio >= 0.3):
            return g, f"tiêu đề có '{ph}'"
    if hint in GENRES:
        if unknown:
            return hint, f"gợi ý theo tác giả ({hint}), chưa đo được tỷ lệ thơ"
        if hint == "tho" and ratio < 0.3:
            return "?", f"tác giả thường viết thơ nhưng trang này chỉ {ratio:.0%} là thơ"
        if hint != "tho" and ratio >= 0.3:
            return "?", f"gợi ý {hint} nhưng {ratio:.0%} nội dung là thơ"
        return hint, f"gợi ý theo tác giả ({hint}), cần kiểm tra"
    return "?", "không đủ căn cứ"


_TEMPLATE_RE = re.compile(r"\{\{(?:[^{}]|\{[^{}]*\})*\}\}")


def wikitext_to_content(wikitext: str) -> dict:
    """Nội dung một trang "Trang:" (ProofreadPage) từ wikitext: bỏ phần đầu/chân trang (<noinclude>),
    bản mẫu, chú thích; khối <poem> thành thơ, phần còn lại thành đoạn văn (cách nhau bởi dòng trống)."""
    t = re.sub(r"<noinclude>.*?</noinclude>", "", wikitext, flags=re.S)
    t = re.sub(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    for _ in range(5):                                   # bản mẫu lồng nhau: bóc từ trong ra ngoài
        t2 = _TEMPLATE_RE.sub("", t)
        if t2 == t:
            break
        t = t2
    t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", t)          # [[đích|chữ]] -> chữ
    t = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", t)
    t = re.sub(r"'{2,}", "", t)
    poems = []

    def _keep_poem(m):
        lines = [" ".join(re.sub(r"<[^>]+>", "", x).split()) for x in m.group(1).split("\n")]
        body = "\n".join(x for x in lines if x)
        if len(re.findall(r"\w+", body)) >= 4:
            poems.append(body)
        return "\n\n"

    t = re.sub(r"<poem[^>]*>(.*?)</poem>", _keep_poem, t, flags=re.S | re.I)
    t = re.sub(r"<br\s*/?>", " ", t, flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    paras = []
    for block in re.split(r"\n\s*\n", t):
        line = " ".join(block.split())
        if len(re.findall(r"\w+", line)) >= 3:
            paras.append(line)
    return {"header": None, "translated": False, "poems": poems, "paras": paras}


def page_number(title: str) -> int:
    """Số trang của một trang "Trang:Tên sách.pdf/12" (0 nếu không có)."""
    m = re.search(r"/(\d+)$", title)
    return int(m.group(1)) if m else 0


def parse_page_range(spec: str) -> Optional[set[int]]:
    """Cột 'trang' trong v5_works.csv: "11-120" hoặc "11-60, 75-80". Rỗng = mọi trang (None)."""
    spec = (spec or "").strip()
    if not spec:
        return None
    pages: set[int] = set()
    for part in re.split(r"[,;]", spec):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            pages.update(range(min(a, b), max(a, b) + 1))
        elif part.isdigit():
            pages.add(int(part))
        else:
            raise ValueError(f"Cột trang không hợp lệ: '{spec}' (ví dụ đúng: 11-120 hoặc 11-60, 75-80)")
    return pages


def excluded(title: str, exclude: list[str]) -> bool:
    for e in exclude:
        if e.startswith("="):
            if title == e[1:]:
                return True
        elif title == e or title.startswith(e + "/"):
            return True
    return False


def same_author(header: Optional[str], name: str) -> bool:
    if header is None:
        return True
    norm = lambda s: unicodedata.normalize("NFC", s).lower()  # noqa: E731
    return norm(name) in norm(header)


def content_for_genre(content: dict, genre: str) -> list[str]:
    """Phần nội dung giữ lại cho thể loại: thơ lấy các bài thơ, văn xuôi lấy các đoạn văn (bỏ thơ chèn giữa)."""
    return content["poems"] if genre == "tho" else content["paras"]


def serialize(units: list[str], genre: str) -> str:
    """Thơ: các bài cách nhau bởi dòng '***' (corpus_builder.split_poems đọc lại); văn xuôi: đoạn cách đoạn
    bởi dòng trống."""
    return "\n***\n".join(units) if genre == "tho" else "\n\n".join(units)


# ====================================================================== gọi API
class Api:
    def __init__(self, sleep: float = SLEEP) -> None:
        self.sleep = sleep
        self._ns: Optional[dict] = None
        CACHE.mkdir(parents=True, exist_ok=True)

    def call(self, cache: bool = True, **params) -> dict:
        params.setdefault("format", "json")
        params.setdefault("formatversion", "2")
        url = API + "?" + urllib.parse.urlencode(params)
        key = CACHE / (hashlib.md5(url.encode("utf-8")).hexdigest() + ".json")
        if cache and key.exists():
            return json.loads(key.read_text(encoding="utf-8"))
        for attempt in range(8):
            try:
                time.sleep(self.sleep)
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                    data = json.loads(r.read())
                if cache:
                    key.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                return data
            except urllib.error.HTTPError as e:
                if e.code in (429, 503):
                    time.sleep(10 * (attempt + 1))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError):
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"Không gọi được API sau nhiều lần thử: {url}")

    def ns(self) -> dict:
        """Mã không gian tên theo tên chuẩn của MediaWiki: Author (Tác gia), Page (Trang), Index (Mục lục)."""
        if self._ns is None:
            d = self.call(action="query", meta="siteinfo", siprop="namespaces")
            self._ns = {}
            for v in d["query"]["namespaces"].values():
                canon = v.get("canonical") or ""
                self._ns[canon] = (v["id"], v.get("name", canon))
        return self._ns

    def ns_id(self, canonical: str) -> Optional[int]:
        x = self.ns().get(canonical)
        return x[0] if x else None

    def ns_name(self, canonical: str, default: str) -> str:
        x = self.ns().get(canonical)
        return x[1] if x else default

    def existing(self, titles: list[str]) -> list[str]:
        keep = []
        for i in range(0, len(titles), 50):
            d = self.call(action="query", titles="|".join(titles[i:i + 50]), redirects=1)
            keep += [p["title"] for p in d["query"]["pages"] if not p.get("missing") and not p.get("invalid")]
        return keep

    def resolve_author(self, name: str) -> Optional[str]:
        """Tên trang tác gia thật trên Wikisource; thử đúng tên trước, sau đó tìm kiếm trong không gian Tác gia."""
        prefix = self.ns_name("Author", "Tác gia")
        title = f"{prefix}:{name}"
        if self.existing([title]):
            return title
        nid = self.ns_id("Author")
        if nid is None:
            return None
        d = self.call(action="query", list="search", srsearch=name, srnamespace=nid, srlimit=5)
        hits = [h["title"] for h in d["query"]["search"]]
        want = unicodedata.normalize("NFC", name).lower()
        for h in hits:
            if want in unicodedata.normalize("NFC", h).lower():
                return h
        return None

    def author_links(self, author_title: str) -> tuple[list[str], list[str]]:
        """(tác phẩm ở không gian chính, trang Mục lục bản chụp) mà trang tác gia liên kết tới."""
        index_ns = self.ns_id("Index")
        namespaces = "0" + (f"|{index_ns}" if index_ns is not None else "")
        d = self.call(action="query", titles=author_title, prop="links", pllimit="max", plnamespace=namespaces)
        links = d["query"]["pages"][0].get("links", [])
        main = [l["title"] for l in links if l["ns"] == 0 and not l["title"].startswith("Luật Sở hữu trí tuệ")]
        idx = [l["title"] for l in links if index_ns is not None and l["ns"] == index_ns]
        return self.existing(sorted(set(main))), self.existing(sorted(set(idx)))

    def parse(self, title: str) -> Optional[dict]:
        d = self.call(action="parse", page=title, prop="text|links|categories", redirects=1, disablelimitreport=1)
        if "parse" not in d:
            return None
        p = d["parse"]
        real = p["title"]
        subs = [l["title"] for l in p.get("links", [])
                if l.get("ns") == 0 and l.get("exists", True) and l["title"].startswith(real + "/")]
        cats = [c["category"].replace("_", " ") for c in p.get("categories", []) if not c.get("hidden")]
        return {"title": real, "html": p["text"], "subpages": subs, "categories": cats}

    def index_pages(self, index_title: str, cache: bool = True) -> list[tuple[str, int]]:
        """Các trang "Trang:" của một bản chụp kèm mức chất lượng soát (theo thứ tự trang).
        cache=False khi tải: người đóng góp có thể vừa soát thêm trang, bộ đệm cũ sẽ không thấy."""
        page_ns = self.ns_id("Page")
        if page_ns is None:
            return []
        base = index_title.split(":", 1)[1]
        titles, cont = [], {}
        while True:
            d = self.call(cache=cache, action="query", list="allpages", apnamespace=page_ns, apprefix=base + "/",
                          aplimit="max", **cont)
            titles += [p["title"] for p in d["query"]["allpages"]]
            if "continue" not in d:
                break
            cont = {"apcontinue": d["continue"]["apcontinue"]}
        out = []
        for i in range(0, len(titles), 50):
            d = self.call(cache=cache, action="query", titles="|".join(titles[i:i + 50]), prop="proofread")
            for p in d["query"]["pages"]:
                q = p.get("proofread", {}).get("quality")
                if q is not None:
                    out.append((p["title"], int(q)))
        return sorted(out, key=lambda x: page_number(x[0]))

    def page_wikitext(self, title: str, cache: bool = True) -> str:
        d = self.call(cache=cache, action="query", titles=title, prop="revisions", rvprop="content", rvslots="main")
        page = d["query"]["pages"][0]
        revs = page.get("revisions") or []
        return revs[0]["slots"]["main"].get("content", "") if revs else ""


# ====================================================================== danh sách tác giả
def load_authors() -> dict[str, dict]:
    """Tác giả ứng viên: data/v5_sources.json, cộng thêm các tác giả của kho v4 (extra_authors_30.json)."""
    src = json.loads(SOURCES_JSON.read_text(encoding="utf-8"))["tac_gia"]
    authors = {k: {"ten": v["ten"], "nam_mat": int(v["nam_mat"]), "goi_y": v.get("goi_y"),
                   "loai_tru": list(v.get("loai_tru", []))} for k, v in src.items()}
    if V4_EXTRA.exists():
        extra = json.loads(V4_EXTRA.read_text(encoding="utf-8"))
        for k, v in extra.items():
            if k not in authors:
                authors[k] = {"ten": v[0], "nam_mat": int(v[1]), "goi_y": None, "loai_tru": list(v[2])}
    return authors


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig để mở bằng Excel trên Windows không lỗi dấu tiếng Việt
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def wiki_url(title: str) -> str:
    return "https://vi.wikisource.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))


# ====================================================================== bước 1: khảo sát
def survey_author(api: Api, slug: str, info: dict, genres_cfg: dict) -> list[dict]:
    name, died = info["ten"], info["nam_mat"]
    base = {"author": slug, "author_name": name, "died": died}
    if died >= 1976:
        return [{**base, "source": "-", "work": "(tác giả)", "ok": 0, "genre": "bo",
                 "ly_do": f"mất năm {died}, còn thời hạn bảo hộ"}]
    author_title = api.resolve_author(name)
    if author_title is None:
        print(f"[{slug}] không tìm thấy trang tác gia '{name}'", flush=True)
        return [{**base, "source": "-", "work": "(tác giả)", "ok": 0, "genre": "",
                 "ly_do": "không tìm thấy trang Tác gia trên Wikisource", "ghi_chu": "sửa 'ten' trong data/v5_sources.json"}]
    works, indexes = api.author_links(author_title)
    works = [w for w in works if not excluded(w, info["loai_tru"])][:MAX_WORKS]
    print(f"[{slug}] {author_title}: {len(works)} tác phẩm, {len(indexes)} bản chụp", flush=True)
    rows = []
    for w in works:
        p = api.parse(w)
        if p is None:
            continue
        c = extract_content(p["html"])
        words = count_words(c["paras"]) + count_words(c["poems"])
        ratio = poem_ratio(c)
        if words < 40 and p["subpages"]:             # trang mục lục: xem chương đầu để biết là thơ hay văn xuôi
            first = api.parse(p["subpages"][0])
            if first is not None:
                ratio = poem_ratio(extract_content(first["html"]))
        ok = int(not c["translated"] and c["header"] is not None and same_author(c["header"], name))
        genre, why = propose_genre(p["title"], p["categories"], ratio, info["goi_y"], genres_cfg)
        if not ok:
            genre, why = "bo", ("bản dịch" if c["translated"] else f"header ghi '{c['header']}'")
        rows.append({**base, "source": "ws", "work": p["title"], "url": wiki_url(p["title"]), "header": c["header"] or "",
                     "ok": ok, "n_subpages": len(p["subpages"]), "words_main": words, "poem_ratio": round(ratio, 2),
                     "categories": "; ".join(p["categories"])[:200], "de_xuat": genre, "ly_do": why,
                     "genre": "" if genre == "?" else genre, "ghi_chu": ""})
    for idx in indexes:
        pages = api.index_pages(idx)
        good = sum(q >= MIN_QUALITY for _, q in pages)
        title = idx.split(":", 1)[1]
        genre, why = propose_genre(title, [], None, info["goi_y"], genres_cfg)
        if good == 0:
            genre, why = "bo", f"chưa có trang nào được soát (0/{len(pages)})"
        rows.append({**base, "source": "ws_trang", "work": idx, "url": wiki_url(idx), "header": "", "ok": int(good > 0),
                     "n_subpages": good, "words_main": "", "poem_ratio": "", "categories": "",
                     "de_xuat": genre, "ly_do": f"{good}/{len(pages)} trang đã soát; " + why,
                     "genre": "" if genre == "?" else genre,
                     "ghi_chu": "bản chụp: kiểm tra tác giả và thể loại bằng tay"})
    return rows


def cmd_survey(args, genres_cfg: dict) -> int:
    api = Api(args.sleep)
    authors = load_authors()
    rows = read_csv(WORKS_CSV)
    done = {r["author"] for r in rows}
    for slug, info in authors.items():
        if args.only and slug not in args.only:
            continue
        if slug in done and not args.refresh:
            continue
        old = {(r["author"], r["work"]): r for r in rows if r["author"] == slug}
        new = survey_author(api, slug, info, genres_cfg)
        for r in new:                                  # khảo sát lại không xóa nhãn đã duyệt bằng tay
            prev = old.get((r["author"], r["work"]))
            if prev and prev.get("genre", "") != prev.get("de_xuat", ""):
                r["genre"], r["ghi_chu"] = prev["genre"], prev.get("ghi_chu", "")
        rows = [r for r in rows if r["author"] != slug] + new
        write_csv(WORKS_CSV, rows, WORK_FIELDS)
    summary: dict[str, dict[str, int]] = {}
    for r in rows:
        g = r.get("genre") or "?"
        summary.setdefault(r["author"], {}).setdefault(g, 0)
        summary[r["author"]][g] += 1
    print("\nSố tác phẩm theo nhãn hiện tại (? = chưa gán):")
    for a, d in sorted(summary.items()):
        print(f"  {a:<22}" + "  ".join(f"{g}:{n}" for g, n in sorted(d.items())))
    print(f"\nĐã ghi {WORKS_CSV}. Duyệt cột genre rồi chạy: python scripts/fetch_v5.py fetch")
    return 0


# ====================================================================== bước 3: tải
def cmd_fetch(args, genres_cfg: dict) -> int:
    try:
        import bs4  # noqa: F401
    except ImportError:
        print("Cần cài: pip install beautifulsoup4")
        return 1
    api = Api(args.sleep)
    rows = [r for r in read_csv(WORKS_CSV) if r.get("genre") in GENRES and str(r.get("ok")) == "1"]
    if not rows:
        print(f"Không có tác phẩm nào có genre hợp lệ trong {WORKS_CSV}. Chạy survey và duyệt trước.")
        return 1
    authors = load_authors()
    manifest = read_csv(RAW / "manifest.csv")
    seen = {(m["genre"], m["page"]) for m in manifest}
    used: dict[tuple[str, str], int] = {}
    for m in manifest:
        used[(m["genre"], m["author"])] = used.get((m["genre"], m["author"]), 0) + int(m["n_words"])

    def save(r: dict, genre: str, page: str, units: list[str], fname: str) -> None:
        n = count_words(units)
        if (genre, page) in seen or n < MIN_PAGE_WORDS[genre]:
            return
        path = RAW / genre / r["author"] / slugify(r["work"].split(":", 1)[-1]) / f"{fname}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialize(units, genre), encoding="utf-8")
        manifest.append({"genre": genre, "author": r["author"], "author_name": r["author_name"], "died": r["died"],
                         "source": r["source"], "work": r["work"], "page": page,
                         "file": str(path.relative_to(RAW)).replace("\\", "/"), "n_words": n, "url": wiki_url(page),
                         "license": "Public domain (vi.wikisource)"})
        seen.add((genre, page))
        used[(genre, r["author"])] = used.get((genre, r["author"]), 0) + n

    by_pair: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if args.only and r["author"] not in args.only:
            continue
        by_pair.setdefault((r["genre"], r["author"]), []).append(r)
    for (genre, author), works in sorted(by_pair.items()):
        rng = random.Random(f"{args.seed}-{genre}-{author}")
        rng.shuffle(works)
        name = authors.get(author, {}).get("ten", works[0]["author_name"])
        print(f"[{genre}/{author}] {len(works)} tác phẩm, đã có {used.get((genre, author), 0)} âm tiết", flush=True)
        pending = []
        for r in works:
            if used.get((genre, author), 0) >= WORD_BUDGET[genre]:
                break
            if r["source"] == "ws_trang":
                # Bản chụp được tải lại trọn mỗi lần (không dùng bộ đệm): số trang đã soát có thể tăng giữa hai lần
                # chạy. Bỏ phần đã tải trước của tác phẩm này rồi ghi lại theo khối 10 trang cố định (1-10, 11-20...).
                wanted = parse_page_range(r.get("trang", ""))
                old = [m for m in manifest if m["work"] == r["work"] and m["genre"] == genre and m["source"] == "ws_trang"]
                for m in old:
                    (RAW / m["file"]).unlink(missing_ok=True)
                    seen.discard((genre, m["page"]))
                    used[(genre, author)] -= int(m["n_words"])
                manifest[:] = [m for m in manifest if not any(m is o for o in old)]
                pages = [t for t, q in api.index_pages(r["work"], cache=False)
                         if q >= MIN_QUALITY and (wanted is None or page_number(t) in wanted)]
                blocks: dict[int, list[str]] = {}
                for t in pages:
                    blocks.setdefault((page_number(t) - 1) // TRANG_PAGES_PER_FILE, []).append(t)
                for b, chunk in sorted(blocks.items()):
                    units = []
                    for t in chunk:
                        units += content_for_genre(wikitext_to_content(api.page_wikitext(t, cache=False)), genre)
                    save(r, genre, chunk[0], units, f"trang_{b * TRANG_PAGES_PER_FILE + 1:04d}")
                print(f"   bản chụp '{r['work']}': {len(pages)} trang đã soát"
                      + (f" trong khoảng {r.get('trang')}" if wanted else ""), flush=True)
                write_csv(RAW / "manifest.csv", manifest, MANIFEST_FIELDS)
                continue
            p = api.parse(r["work"])
            if p is None:
                continue
            save(r, genre, p["title"], content_for_genre(extract_content(p["html"]), genre), "main")
            subs = p["subpages"][:]
            rng.shuffle(subs)
            for sub in subs[:MAX_PAGES_PER_WORK]:
                q = api.parse(sub)
                if q is None:
                    continue
                c = extract_content(q["html"])
                if not c["translated"] and same_author(c["header"], name):
                    save(r, genre, q["title"], content_for_genre(c, genre), slugify(q["title"].split("/", 1)[1]))
            pending.append((r, subs[MAX_PAGES_PER_WORK:]))
            write_csv(RAW / "manifest.csv", manifest, MANIFEST_FIELDS)
        for r, rest in pending:                       # lượt 2: còn thiếu thì lấy nốt các chương đã bỏ qua
            for sub in rest:
                if used.get((genre, author), 0) >= WORD_BUDGET[genre]:
                    break
                q = api.parse(sub)
                if q is None:
                    continue
                c = extract_content(q["html"])
                if not c["translated"] and same_author(c["header"], name):
                    save(r, genre, q["title"], content_for_genre(c, genre), slugify(q["title"].split("/", 1)[1]))
        write_csv(RAW / "manifest.csv", manifest, MANIFEST_FIELDS)
        print(f"   -> {used.get((genre, author), 0)} âm tiết", flush=True)

    # bản tóm tắt nằm nông (data/) để đọc được từ xa; manifest đầy đủ nằm trong data/v5_genre/raw
    summary: dict[tuple[str, str], list] = {}
    for m in manifest:
        s = summary.setdefault((m["genre"], m["author"]), [0, set(), 0])
        s[0] += int(m["n_words"])
        s[1].add(m["work"])
        s[2] += 1
    out = [{"genre": g, "author": a, "words": s[0], "works": len(s[1]), "pages": s[2]}
           for (g, a), s in sorted(summary.items())]
    write_csv(DATA_DIR / "v5_manifest_summary.csv", out, ["genre", "author", "words", "works", "pages"])
    for g in GENRES:
        part = [x for x in out if x["genre"] == g]
        print(f"\n== {g}: {len(part)} tác giả")
        for x in part:
            print(f"   {x['author']:<22}{x['words']:>8} âm tiết {x['works']:>4} tác phẩm")
    print(f"\nTiếp theo: python main.py build-genre-corpus")
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["survey", "fetch"])
    ap.add_argument("--only", nargs="*", default=None, help="chỉ xử lý các tác giả này (slug)")
    ap.add_argument("--refresh", action="store_true", help="survey: khảo sát lại cả tác giả đã có trong v5_works.csv")
    ap.add_argument("--sleep", type=float, default=SLEEP)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    genres_cfg = json.loads(GENRES_JSON.read_text(encoding="utf-8"))
    return cmd_survey(args, genres_cfg) if args.step == "survey" else cmd_fetch(args, genres_cfg)


if __name__ == "__main__":
    sys.exit(main())
