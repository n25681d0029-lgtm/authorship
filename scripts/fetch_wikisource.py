#!/usr/bin/env python3
"""Tải văn xuôi tiếng Việt thuộc phạm vi công cộng từ vi.wikisource.org để làm kho dữ liệu thật.

Chỉ lấy tác giả đã mất trước năm 1976 (Luật SHTT Việt Nam: bảo hộ 50 năm sau khi tác giả mất),
và mọi trang tải về đều mang nhãn "phạm vi công cộng" của Wikisource.

Kết quả:
    data/raw/wikisource/<tác giả>/<tác phẩm>/<trang>.txt   (mỗi đoạn văn cách nhau một dòng trống)
    data/raw/wikisource/manifest.csv                        (nguồn, URL, số từ của từng trang)

Sau đó tạo CSV các đoạn văn bằng:
    python main.py build-corpus --src data/raw/wikisource --out data/raw/vi_wikisource.csv

Chạy:  python scripts/fetch_wikisource.py            (chạy lại sẽ bỏ qua trang đã tải)
       python scripts/fetch_wikisource.py --only nam_cao thach_lam
Cần: pip install beautifulsoup4
"""
from __future__ import annotations

import argparse
import csv
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_DIR  # noqa: E402

API = "https://vi.wikisource.org/w/api.php"
UA = {"User-Agent": "authorship-nlp-student/0.2 (university NLP coursework; contact via project owner)"}
OUT = RAW_DATA_DIR / "wikisource"

# slug -> (tên trên Wikisource, năm mất, các tác phẩm bỏ qua vì không phải văn xuôi của chính tác giả)
# Mỗi mục loại trừ bỏ cả trang đó lẫn các trang con của nó; mục bắt đầu bằng "=" chỉ bỏ đúng trang đó.
AUTHORS = {
    "nam_cao": ("Nam Cao", 1951, ["Đôi lứa xứng đôi (tập truyện ngắn)"]),
    "vu_trong_phung": ("Vũ Trọng Phụng", 1939, ["Không một tiếng vang"]),   # kịch
    "ngo_tat_to": ("Ngô Tất Tố", 1954, ["Kinh Dịch", "Mặc Tử", "Hoàng Lê nhất thống chí",
                                        "Thi văn bình chú", "Đường thi (Ngô Tất Tố)", "Hịch tướng sĩ"]),
    "thach_lam": ("Thạch Lam", 1942, []),
    "ho_bieu_chanh": ("Hồ Biểu Chánh", 1958, []),
    "lan_khai": ("Lan Khai", 1945, []),
    "phan_khoi": ("Phan Khôi", 1959, []),
    # Các bài Nam Phong mà trang tác gia của Phạm Quỳnh liên kết tới được giữ lại (bình luận, thời đàm, triết học...),
    # trừ trang mục lục của cả tạp chí, mục tiểu thuyết (truyện dịch) và bài diễn thuyết của quan chức Pháp (bản dịch).
    "pham_quynh": ("Phạm Quỳnh", 1945, [
        "=Nam Phong tạp chí",
        "Nam Phong tạp chí/Quyển I/Số 1/Tiểu-thuyết", "Nam Phong tạp chí/Quyển I/Số 2/Tiểu-thuyết",
        "Nam Phong tạp chí/Quyển I/Số 2/Luận-thuyết/Một bài diễn-thuyết của quan Toàn-quyền Sarraut",
        "Nam Phong tạp chí/Quyển I/Số 2/Tạp-trở/Bài diễn-thuyết của quan học-chính Russier về việc nước Mĩ nhập "
        "cuộc với Đồng-minh",
        "Nam Phong tạp chí/Quyển I/Số 2/Tạp-trở/Truyện người lính bằng tuyết"]),
    "nguyen_van_vinh": ("Nguyễn Văn Vĩnh", 1936, ["Bệnh tưởng", "Thơ ngụ ngôn La Fontaine", "Truyện Gil-Blas de Santillane",
                                                   "Truyện Tê-lê-mặc phiêu lưu ký", "Truyện các danh nhân Hi-lạp, La-mã đối nhau",
                                                   "Con ve và con kiến (Nguyễn Văn Vĩnh dịch 1)", "Qui-li-ve du ký",
                                                   "Mai-nương Lệ-cốt", "Con yêu râu xanh", "Jean de La Fontaine"]),
    "phan_ke_binh": ("Phan Kế Bính", 1921, ["Tam quốc diễn nghĩa", "Hịch tướng sĩ (Phan Kế Bính dịch)", "Nét mực tình"]),
    "tran_trong_kim": ("Trần Trọng Kim", 1953, ["Việt thi", "Văn tế Trần Trọng Kim", "Điếu văn Trần Trọng Kim (Nguyễn Hữu Trí)",
                                                 "Bình Ngô đại cáo (Trần Trọng Kim dịch)", "Hịch tướng sĩ (Trần Trọng Kim dịch)"]),
    "tan_da": ("Tản Đà", 1939, ["Thơ Tản Đà", "Khối tình con", "Ca dao, dân ca"]),
    "nguyen_chanh_sat": ("Nguyễn Chánh Sắt", 1947, []),
}

MAX_WORKS = 40            # số tác phẩm tối đa xét cho mỗi tác giả
MAX_PAGES_PER_WORK = 6    # mỗi tác phẩm chỉ lấy vài chương để kho đa dạng tác phẩm hơn
WORD_BUDGET = 90_000      # đủ số từ thì dừng tác giả đó
MIN_PAGE_WORDS = 40       # trang ít chữ hơn (mục lục, trang bìa) thì bỏ
SLEEP = 1.0               # giãn cách giữa các lần gọi API

DROP_SELECTORS = ["#headerContainer", ".ws-noexport", ".licenseContainer", ".licensetpl", "sup", ".reference",
                  ".references", ".mw-references-wrap", "table", ".pagenum", ".ws-pagenum", "style", "script",
                  ".poem", ".mw-editsection", ".navbox", "#toc", ".toc", ".noprint", ".mw-heading"]


def slugify(s: str) -> str:
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")[:80] or "x"


def api(**params) -> dict:
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(8):
        try:
            time.sleep(SLEEP)
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(10 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Không gọi được API sau nhiều lần thử: {url}")


def existing(titles: list[str]) -> list[str]:
    """Lọc các trang thật sự tồn tại (trang tác gia có cả liên kết đỏ)."""
    keep = []
    for i in range(0, len(titles), 50):
        d = api(action="query", titles="|".join(titles[i:i + 50]), redirects=1)
        keep += [p["title"] for p in d["query"]["pages"] if not p.get("missing") and not p.get("invalid")]
    return keep


def author_works(name: str) -> list[str]:
    d = api(action="query", titles=f"Tác gia:{name}", prop="links", pllimit="max", plnamespace="0")
    page = d["query"]["pages"][0]
    links = [l["title"] for l in page.get("links", []) if not l["title"].startswith("Luật Sở hữu trí tuệ")]
    return existing(sorted(set(links)))


def parse_page(title: str):
    """Trả về (tác giả ghi trên header, có người dịch không, danh sách đoạn văn, danh sách trang con)."""
    from bs4 import BeautifulSoup

    d = api(action="parse", page=title, prop="text|links", redirects=1, disablelimitreport=1)
    if "parse" not in d:
        return None, False, [], []
    soup = BeautifulSoup(d["parse"]["text"], "html.parser")
    ws_author = soup.select_one("#ws-author")
    author = ws_author.get_text(strip=True) if ws_author else None
    translated = soup.select_one("#header_translator_text, #ws-translator") is not None
    root = soup.select_one(".mw-parser-output") or soup
    for sel in DROP_SELECTORS:
        for el in root.select(sel):
            el.decompose()
    paras = []
    for p in root.find_all("p"):
        if len(p.find_all("br")) >= 2:          # nhiều dòng ngắt trong một đoạn: thường là thơ
            continue
        t = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
        if len(re.findall(r"\w+", t)) >= 3:
            paras.append(t)
    real = d["parse"]["title"]
    subs = [l["title"] for l in d["parse"].get("links", [])
            if l.get("ns") == 0 and l.get("exists", True) and l["title"].startswith(real + "/")]
    return author, translated, paras, subs


def excluded(title: str, exclude: list[str]) -> bool:
    for e in exclude:
        if e.startswith("="):
            if title == e[1:]:
                return True
        elif title == e or title.startswith(e + "/"):
            return True
    return False


def same_author(header: str | None, name: str) -> bool:
    if header is None:
        return True              # trang con thường không có header, đã kiểm tra ở trang cha
    norm = lambda s: unicodedata.normalize("NFC", s).lower()  # noqa: E731
    return norm(name) in norm(header)


def fetch_author(slug: str, name: str, died: int, exclude: list[str], rng: random.Random, manifest: list[dict]) -> None:
    adir = OUT / slug
    done_words = sum(int(r["n_words"]) for r in manifest if r["author"] == slug)
    if done_words >= WORD_BUDGET:
        print(f"[{slug}] đã đủ {done_words} từ, bỏ qua")
        return
    works = [w for w in author_works(name) if not excluded(w, exclude)]
    rng.shuffle(works)
    print(f"[{slug}] {name} (mất {died}): {len(works)} trang tác phẩm tồn tại", flush=True)
    seen = {r["page"] for r in manifest}

    def fetch_subpages(subs: list[str]) -> list[tuple[str, list[str]]]:
        out = []
        for sub in subs:
            if sub in seen:
                continue
            h2, tr2, p2, _ = parse_page(sub)
            if not tr2 and same_author(h2, name):
                out.append((sub, p2))
        return out

    def save_pages(work: str, pages: list[tuple[str, list[str]]]) -> None:
        nonlocal done_words
        for title, ps in pages:
            n = sum(len(p.split()) for p in ps)
            if title in seen or n < MIN_PAGE_WORDS:
                continue
            path = adir / slugify(work) / f"{slugify(title.split('/', 1)[1]) if '/' in title else 'main'}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n\n".join(ps), encoding="utf-8")
            manifest.append({"author": slug, "author_name": name, "died": died, "work": work, "page": title,
                             "file": str(path.relative_to(OUT)).replace("\\", "/"), "n_words": n,
                             "url": "https://vi.wikisource.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                             "license": "Public domain (vi.wikisource)"})
            seen.add(title)
            done_words += n
        save_manifest(manifest)

    pending = []          # các chương bị bỏ qua ở lượt 1 do giới hạn MAX_PAGES_PER_WORK
    for work in works[:MAX_WORKS]:
        if done_words >= WORD_BUDGET:
            break
        header, translated, paras, subs = parse_page(work)
        # trang tác phẩm chính bắt buộc có header ghi đúng tác giả (trang tác gia có thể liên kết tới văn bản
        # không phải của họ, ví dụ văn bản pháp luật); chỉ trang con (chương) mới được thiếu header
        if translated or header is None or not same_author(header, name):
            print(f"   bỏ '{work}': header ghi '{header}'{' (bản dịch)' if translated else ''}")
            continue
        rng.shuffle(subs)
        pages = [(work, paras)] + fetch_subpages(subs[:MAX_PAGES_PER_WORK])
        pending.append((work, subs[MAX_PAGES_PER_WORK:]))
        save_pages(work, pages)
        print(f"   '{work}': {len(pages)} trang, tổng {done_words} từ", flush=True)

    # Lượt 2: tác giả vẫn thiếu số từ thì lấy nốt các chương đã bỏ qua (ít đa dạng tác phẩm hơn nhưng thêm dữ liệu)
    for work, rest in pending:
        if done_words >= WORD_BUDGET:
            break
        pages = fetch_subpages(rest)
        if pages:
            save_pages(work, pages)
            print(f"   bổ sung '{work}': {len(pages)} trang, tổng {done_words} từ", flush=True)


def load_manifest() -> list[dict]:
    p = OUT / "manifest.csv"
    if not p.exists():
        return []
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_manifest(rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["author", "author_name", "died", "work", "page", "file", "n_words", "url", "license"]
    with open(OUT / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, help="chỉ tải các tác giả này (slug)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--authors-file", default=None,
                    help="JSON bổ sung tác giả: {slug: [tên trên Wikisource, năm mất, [loại trừ...]]}")
    args = ap.parse_args()
    authors = dict(AUTHORS)
    if args.authors_file:
        extra = json.loads(Path(args.authors_file).read_text(encoding="utf-8"))
        authors.update({k: (v[0], int(v[1]), list(v[2])) for k, v in extra.items()})
    try:
        import bs4  # noqa: F401
    except ImportError:
        print("Cần cài: pip install beautifulsoup4")
        return 1
    manifest = load_manifest()
    for slug, (name, died, exclude) in authors.items():
        if args.only and slug not in args.only:
            continue
        if died >= 1976:
            print(f"[{slug}] tác giả mất năm {died}, chưa hết thời hạn bảo hộ, bỏ qua")
            continue
        fetch_author(slug, name, died, exclude, random.Random(f"{args.seed}-{slug}"), manifest)
    save_manifest(manifest)
    by_author = {}
    for r in manifest:
        by_author.setdefault(r["author"], [0, set()])
        by_author[r["author"]][0] += int(r["n_words"])
        by_author[r["author"]][1].add(r["work"])
    for a, (n, ws) in sorted(by_author.items()):
        print(f"{a:<20}{n:>8} từ  {len(ws):>3} tác phẩm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
