"""Kiểm thử kho v5 theo thể loại: chuẩn hóa, lọc OCR, cắt thơ, dựng kho, các hàm phân tích của fetch_v5.

Chạy: python -m unittest tests.test_v5_genre
Không cần mạng: phần gọi API của scripts/fetch_v5.py không được kiểm thử ở đây.
"""
import importlib.util
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.corpus_builder import build_genre_corpus, load_genres, pack_poem_lines, split_poems  # noqa: E402
from src.text_utils import invalid_syllable_ratio, is_valid_syllable, normalize  # noqa: E402

GENRES = load_genres(ROOT / "data" / "genres.json")
VOCAB = ("người ta nói rằng trong làng có một ông đồ già ngày ngày ngồi viết chữ bên đường cho những ai "
         "muốn mua câu đối tết về treo nhà năm nay ông vẫn ngồi đấy nhưng chẳng còn ai thuê viết nữa").split()


def _fetch_v5():
    spec = importlib.util.spec_from_file_location("fetch_v5", ROOT / "scripts" / "fetch_v5.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _prose(rng, n_words):
    words = [rng.choice(VOCAB) for _ in range(n_words)]
    sents, i = [], 0
    while i < len(words):
        k = rng.randint(8, 16)
        sents.append(" ".join(words[i:i + k]).capitalize() + ".")
        i += k
    return " ".join(sents)


def _poem(rng, n_lines, per_line=7):
    return "\n".join(" ".join(rng.choice(VOCAB) for _ in range(per_line)) for _ in range(n_lines))


class NormalizeTest(unittest.TestCase):
    def test_homoglyph_invisible_and_line_hyphen(self):
        cyr = "giáo khоu"                      # chữ "о" Kirin
        self.assertEqual(normalize(cyr), "giáo khou")
        self.assertEqual(normalize("lợi​ khí­"), "lợi khí")
        self.assertEqual(normalize("sai-\nbiệt"), "sai biệt")
        self.assertEqual(normalize("quốc-ngữ « văn »"), "quốc ngữ “văn”")
        self.assertEqual(normalize("hai lối : Một là ? Phải ! ( chú )"), "hai lối: Một là? Phải! (chú)")
        self.assertEqual(normalize("Nói — rồi đi"), "Nói — rồi đi")         # gạch ngang giữ khoảng trắng
        self.assertEqual(normalize("năm 1934-1935"), "năm 1934-1935")   # số không bị đụng tới

    def test_normalize_idempotent(self):
        s = "Chữ quốc-ngữ — « văn-chương » sai-\nbiệt"
        self.assertEqual(normalize(normalize(s)), normalize(s))


class SyllableTest(unittest.TestCase):
    def test_valid(self):
        for w in ["người", "nghiêng", "khuya", "quyết", "giường", "xoong", "oẳn", "gì", "thuở", "nhứt", "chánh",
                  "Ngoằn", "ngoèo", "hoạ", "tuyết", "khuỷu"]:
            self.assertTrue(is_valid_syllable(w), w)

    def test_invalid(self):
        for w in ["nưưc", "liọc", "Voltaire", "khоu", "baa", "tháng̀"]:
            self.assertFalse(is_valid_syllable(w), w)

    def test_ratio_separates_ocr_from_clean(self):
        ocr = ("Chữ quốc-ngữ ngày nay đữ phồ-thông khắp trong nưưc, có thể dùng lắm cái lợi-khi đe truyễn-bả "
               "sự học trong dân-gian, ngoài những sách giáo-khоu đề dùng trong các trường so-liọc")
        clean = ("Chữ quốc ngữ ngày nay đã phổ thông khắp trong nước, có thể dùng làm cái lợi khí để truyền bá "
                 "sự học trong dân gian, ngoài những sách giáo khoa để dùng trong các trường sơ học")
        self.assertGreater(invalid_syllable_ratio(ocr), 0.05)
        self.assertEqual(invalid_syllable_ratio(clean), 0.0)


class PoemPackingTest(unittest.TestCase):
    def test_split_poems_drops_stanza_breaks(self):
        text = "dòng một\ndòng hai\n\ndòng ba\n***\nbài hai dòng một\n"
        self.assertEqual(split_poems(text), [["dòng một", "dòng hai", "dòng ba"], ["bài hai dòng một"]])

    def test_short_poems_are_merged_long_poem_is_cut(self):
        rng = random.Random(0)
        quatrain = _poem(rng, 4).split("\n")                  # 28 âm tiết, ngắn hơn min 40
        out = pack_poem_lines([quatrain, quatrain[::-1]], min_words=40, max_words=160, target_words=120)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0].split()), 56)
        self.assertEqual(out[0].count("\n"), 7)               # giữ ranh giới dòng thơ
        long_poem = _poem(rng, 60).split("\n")                # 420 âm tiết
        parts = pack_poem_lines([long_poem], 40, 160, 120)
        self.assertGreaterEqual(len(parts), 3)
        self.assertTrue(all(40 <= len(p.split()) <= 160 for p in parts))

    def test_labels_point_to_first_poem(self):
        rng = random.Random(1)
        poems = [_poem(rng, 4).split("\n") for _ in range(6)]
        out = pack_poem_lines(poems, 40, 160, 120, labels=list("abcdef"))
        self.assertEqual(out[0][1], "a")
        self.assertTrue(all(lab in "abcdef" for _, lab in out))


class GenreCorpusTest(unittest.TestCase):
    def _make_tree(self, root: Path, n_prose_authors=6, n_poets=3):
        rng = random.Random(7)
        for a in range(n_prose_authors):
            for w in range(3):
                d = root / "tu_su" / f"tg{a}" / f"tp{w}"
                d.mkdir(parents=True)
                for c in range(3):
                    (d / f"chuong{c}.txt").write_text(
                        "\n\n".join(_prose(rng, rng.randint(60, 140)) for _ in range(12)), encoding="utf-8")
        for a in range(n_poets):                              # 3 nhà thơ (< 5): thể loại tho phải bị loại
            for w in range(2):
                d = root / "tho" / f"nt{a}" / f"tap{w}"
                d.mkdir(parents=True)
                (d / "main.txt").write_text("\n***\n".join(_poem(rng, 8) for _ in range(60)), encoding="utf-8")
        d = root / "tu_su" / "it_du_lieu" / "tp0"             # một tác phẩm: bị loại vì < 2 tác phẩm
        d.mkdir(parents=True)
        (d / "a.txt").write_text("\n\n".join(_prose(rng, 100) for _ in range(80)), encoding="utf-8")
        d = root / "tu_su" / "tg0" / "ban_ocr"                # đoạn OCR bẩn: bị lọc
        d.mkdir(parents=True)
        (d / "x.txt").write_text(" ".join(["nưưc liọc khоu baa"] * 40), encoding="utf-8")

    def test_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_tree(root)
            df, stats = build_genre_corpus(root, GENRES, per_author=50)
            self.assertEqual(set(df["genre"]), {"tu_su"})
            self.assertIn("tho", stats["dropped_genres"])
            self.assertIn("tu_su/it_du_lieu", stats["dropped_pairs"])
            self.assertIn("tu_su/tg0", stats["noisy_passages_removed"])
            self.assertEqual(df["author"].nunique(), 6)
            self.assertTrue((df.groupby("author").size() <= 50).all())
            self.assertTrue((df["invalid_ratio"] <= 0.03).all())
            self.assertTrue(df["doc_id"].str.startswith("tu_su/").all())
            for col in ["text", "author", "genre", "topic", "doc_id", "n_words", "source"]:
                self.assertIn(col, df.columns)

    def test_poetry_kept_when_enough_poets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_tree(root, n_poets=5)
            df, stats = build_genre_corpus(root, GENRES, per_author=50)
            tho = df[df["genre"] == "tho"]
            self.assertEqual(tho["author"].nunique(), 5)
            self.assertTrue(tho["text"].str.contains("\n").all())        # thơ giữ xuống dòng
            self.assertTrue(tho["n_words"].between(40, 160).all())


class FetchV5ParsingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import bs4  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("cần beautifulsoup4")
        cls.f = _fetch_v5()

    def test_extract_header_is_read(self):
        html = '<div class="mw-parser-output"><span id="ws-author">Tản Đà</span><p>Một hai ba bốn năm.</p></div>'
        c = self.f.extract_content(html)
        self.assertEqual(c["header"], "Tản Đà")
        self.assertEqual(c["paras"], ["Một hai ba bốn năm."])

    def test_extract_poem_lines(self):
        html = """<div class="mw-parser-output"><p>Đoạn văn xuôi thứ nhất có đủ nhiều chữ.</p>
        <div class="poem"><p>Dòng thơ một năm chữ<br>Dòng thơ hai năm chữ<br><br>Dòng thơ ba năm chữ</p></div>
        <p>Dòng a có chữ<br>Dòng b có chữ<br>Dòng c thêm chữ</p></div>"""
        c = self.f.extract_content(html)
        self.assertEqual(c["poems"][0], "Dòng thơ một năm chữ\nDòng thơ hai năm chữ\nDòng thơ ba năm chữ")
        self.assertEqual(c["poems"][1], "Dòng a có chữ\nDòng b có chữ\nDòng c thêm chữ")
        self.assertEqual(c["paras"], ["Đoạn văn xuôi thứ nhất có đủ nhiều chữ."])
        self.assertGreater(self.f.poem_ratio(c), 0.5)

    def test_propose_genre(self):
        pg = self.f.propose_genre
        self.assertEqual(pg("Bài bất kỳ", [], 0.9, None, GENRES)[0], "tho")
        self.assertEqual(pg("Việc làng", ["Phóng sự"], 0.0, "tu_su", GENRES)[0], "ky")
        self.assertEqual(pg("Không một tiếng vang", ["Kịch"], 0.0, "tu_su", GENRES)[0], "bo")
        self.assertEqual(pg("Văn minh luận", [], 0.0, None, GENRES)[0], "nghi_luan")
        self.assertEqual(pg("Chuyến đi Bắc Kỳ năm Ất Hợi", [], 0.0, None, GENRES)[0], "ky")
        self.assertEqual(pg("Lão Hạc", [], 0.0, "tu_su", GENRES)[0], "tu_su")
        self.assertEqual(pg("Lão Hạc", [], 0.0, None, GENRES)[0], "?")
        self.assertEqual(pg("Một trang văn", [], 0.0, "tho", GENRES)[0], "?")    # nhà thơ nhưng trang văn xuôi
        self.assertEqual(pg("Các bài", [], 0.0, None, GENRES)[0], "?")           # "ca" không khớp "các"

    def test_wikitext_to_content(self):
        wt = ("<noinclude>{{rh||TRANG 3|}}</noinclude>Chữ '''quốc ngữ''' ngày nay đã [[phổ thông|phổ-thông]] "
              "khắp trong nước{{ref|1}}.<ref>chú thích</ref>\n\nĐoạn thứ hai cũng đủ chữ.\n"
              "<poem>Dòng thơ một\nDòng thơ hai\n\nDòng thơ ba</poem><noinclude>{{smallrefs}}</noinclude>")
        c = self.f.wikitext_to_content(wt)
        self.assertEqual(c["paras"][0], "Chữ quốc ngữ ngày nay đã phổ-thông khắp trong nước.")
        self.assertEqual(c["paras"][1], "Đoạn thứ hai cũng đủ chữ.")
        self.assertEqual(c["poems"], ["Dòng thơ một\nDòng thơ hai\nDòng thơ ba"])

    def test_serialize_roundtrip_with_split_poems(self):
        units = ["Dòng một\nDòng hai", "Bài hai dòng một\nBài hai dòng hai"]
        self.assertEqual(split_poems(self.f.serialize(units, "tho")),
                         [["Dòng một", "Dòng hai"], ["Bài hai dòng một", "Bài hai dòng hai"]])

    def test_page_range(self):
        self.assertIsNone(self.f.parse_page_range(""))
        self.assertEqual(self.f.parse_page_range("3-5, 9"), {3, 4, 5, 9})
        self.assertEqual(self.f.page_number("Trang:Com Thay Com Co va Luc Si 1937.pdf/12"), 12)
        with self.assertRaises(ValueError):
            self.f.parse_page_range("ba đến năm")

    def test_excluded(self):
        self.assertTrue(self.f.excluded("Kinh Dịch/Quyển 1", ["Kinh Dịch"]))
        self.assertFalse(self.f.excluded("Nam Phong tạp chí/Số 1", ["=Nam Phong tạp chí"]))
        self.assertTrue(self.f.excluded("Nam Phong tạp chí", ["=Nam Phong tạp chí"]))


if __name__ == "__main__":
    unittest.main()


class FetchV5FlowTest(unittest.TestCase):
    """Chạy survey -> fetch -> build_genre_corpus với API giả (không cần mạng)."""

    @classmethod
    def setUpClass(cls):
        try:
            import bs4  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("cần beautifulsoup4")

    def test_flow(self):
        import json
        import types

        f = _fetch_v5()
        rng = random.Random(3)
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            f.DATA_DIR, f.V5 = data, data / "v5_genre"
            f.RAW, f.CACHE = f.V5 / "raw", f.V5 / "cache"
            f.WORKS_CSV, f.SOURCES_JSON = data / "v5_works.csv", data / "v5_sources.json"
            f.V4_EXTRA = data / "none.json"
            f.SOURCES_JSON.write_text(json.dumps({"tac_gia": {
                "nha_tho": {"ten": "Nhà Thơ", "nam_mat": 1940, "goi_y": "tho", "loai_tru": []},
                "nha_van": {"ten": "Nhà Văn", "nam_mat": 1950, "goi_y": "tu_su", "loai_tru": ["Bản dịch X"]},
                "con_ban_quyen": {"ten": "Còn Bản Quyền", "nam_mat": 1990, "loai_tru": []}}}), encoding="utf-8")

            def page(author, body):
                return f'<div class="mw-parser-output"><span id="ws-author">{author}</span>{body}</div>'

            prose = lambda: "".join(f"<p>{_prose(rng, 120)}</p>" for _ in range(6))   # noqa: E731
            poems = lambda: "".join(f'<div class="poem"><p>{_poem(rng, 8).replace(chr(10), "<br>")}</p></div>'  # noqa: E731
                                    for _ in range(5))
            pages = {
                "Thơ A": (page("Nhà Thơ", poems()), [], []),
                "Thơ B": (page("Nhà Thơ", poems()), [], []),
                "Truyện C": (page("Nhà Văn", "<p>Mục lục</p>"), ["Truyện C/Chương 1", "Truyện C/Chương 2"], []),
                "Truyện C/Chương 1": ('<div class="mw-parser-output">' + prose() + "</div>", [], []),
                "Truyện C/Chương 2": ('<div class="mw-parser-output">' + prose() + "</div>", [], []),
                "Phóng sự D": (page("Nhà Văn", prose()), [], ["Phóng sự"]),
                "Bản dịch X": (page("Nhà Văn", prose()), [], []),
                "Bài của người khác": (page("Người Khác", prose()), [], []),
            }
            links = {"Tác gia:Nhà Thơ": (["Thơ A", "Thơ B"], ["Mục lục:Sách thơ.pdf"]),
                     "Tác gia:Nhà Văn": (["Truyện C", "Phóng sự D", "Bản dịch X", "Bài của người khác"], [])}
            api = f.Api.__new__(f.Api)
            api.resolve_author = lambda name: f"Tác gia:{name}" if f"Tác gia:{name}" in links else None
            api.author_links = lambda t: links[t]
            api.parse = lambda t: {"title": t, "html": pages[t][0], "subpages": pages[t][1], "categories": pages[t][2]}
            proofread = {1: 3, 2: 1, 3: 4}
            api.index_pages = lambda t, cache=True: [(f"Trang:Sách thơ.pdf/{n}", q) for n, q in sorted(proofread.items())]
            api.page_wikitext = lambda t, cache=True: ("<noinclude>{{rh}}</noinclude><poem>" + _poem(rng, 8) + "</poem>")
            f.Api = lambda sleep=1.0: api                                                     # noqa: E731
            args = types.SimpleNamespace(only=None, refresh=False, sleep=0, seed=42)

            f.cmd_survey(args, GENRES)
            rows = {r["work"]: r for r in f.read_csv(f.WORKS_CSV)}
            self.assertEqual(rows["Thơ A"]["genre"], "tho")
            self.assertEqual(rows["Phóng sự D"]["genre"], "ky")
            self.assertEqual(rows["Truyện C"]["genre"], "tu_su")          # mục lục: xét chương đầu, gợi ý tác giả
            self.assertEqual(rows["Bài của người khác"]["genre"], "bo")
            self.assertNotIn("Bản dịch X", rows)                          # nằm trong loai_tru
            self.assertEqual(rows["(tác giả)"]["author"], "con_ban_quyen")
            trang = [r for r in rows.values() if r["source"] == "ws_trang"][0]
            self.assertTrue(trang["ly_do"].startswith("2/3 trang đã soát"))
            self.assertEqual(trang["genre"], "tho")

            f.cmd_fetch(args, GENRES)
            files = sorted(str(p.relative_to(f.RAW)).replace("\\", "/") for p in f.RAW.rglob("*.txt"))
            self.assertIn("tho/nha_tho/tho_a/main.txt", files)
            self.assertIn("ky/nha_van/phong_su_d/main.txt", files)
            self.assertTrue(any(x.startswith("tu_su/nha_van/truyen_c/chuong_") for x in files))
            self.assertTrue(any("sach_tho_pdf/trang_" in x for x in files))
            self.assertFalse(any("ban_dich" in x or "nguoi_khac" in x for x in files))
            self.assertTrue((data / "v5_manifest_summary.csv").exists())
            # file thơ đọc lại được thành các bài
            text = (f.RAW / "tho/nha_tho/tho_a/main.txt").read_text(encoding="utf-8")
            self.assertEqual(len(split_poems(text)), 5)
            # chạy lại fetch không tải trùng
            n_before = len(f.read_csv(f.RAW / "manifest.csv"))
            f.cmd_fetch(args, GENRES)
            self.assertEqual(len(f.read_csv(f.RAW / "manifest.csv")), n_before)
            # Bản chụp: người dùng soát thêm trang 2 và 25, chỉ lấy khoảng 1-20 -> tải lại đủ trang 1-3, bỏ trang 25
            proofread.update({2: 3, 25: 3})
            rows_csv = f.read_csv(f.WORKS_CSV)
            for r in rows_csv:
                if r["source"] == "ws_trang":
                    r["trang"] = "1-20"
            f.write_csv(f.WORKS_CSV, rows_csv, f.WORK_FIELDS)
            f.cmd_fetch(args, GENRES)
            man = [m for m in f.read_csv(f.RAW / "manifest.csv") if m["source"] == "ws_trang"]
            self.assertEqual([m["file"] for m in man], ["tho/nha_tho/sach_tho_pdf/trang_0001.txt"])
            poems_in_block = split_poems((f.RAW / man[0]["file"]).read_text(encoding="utf-8"))
            self.assertEqual(len(poems_in_block), 3)                     # trang 1, 2, 3 (trang 25 ngoài khoảng)
            df, stats = build_genre_corpus(f.RAW, GENRES, min_per_author={"tho": 1, "tu_su": 1, "ky": 1},
                                           min_works=1, min_authors=1)
            self.assertEqual(set(df["genre"]), {"tho", "tu_su", "ky"})
            self.assertTrue(df["source"].str.startswith("https://vi.wikisource.org/").any())


class SyncManifestTest(unittest.TestCase):
    """scripts/sync_v5_manifest.py: bỏ tác phẩm đã tải nhưng sau đó bị đổi nhãn (vd. phát hiện là thơ chữ Hán)."""

    def test_stale_entries(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from sync_v5_manifest import stale_entries
        works = [{"genre": "ky", "author": "a", "work": "W1", "ok": "1"},
                 {"genre": "bo", "author": "b", "work": "Thu nhật ký hứng", "ok": "1"},
                 {"genre": "tho", "author": "c", "work": "W3", "ok": "0"}]
        manifest = [{"genre": "ky", "author": "a", "work": "W1", "source": "ws"},
                    {"genre": "ky", "author": "b", "work": "Thu nhật ký hứng", "source": "ws"},   # đã đổi sang bo
                    {"genre": "tho", "author": "c", "work": "W3", "source": "ws"},               # header không đạt
                    {"genre": "tu_su", "author": "a", "work": "W1", "source": "ws"},             # sai thể loại
                    {"genre": "tu_su", "author": "d", "work": "Sách tự gõ", "source": "local"}]   # nguồn local: giữ
        got = [(m["genre"], m["author"]) for m in stale_entries(manifest, works)]
        self.assertEqual(got, [("ky", "b"), ("tho", "c"), ("tu_su", "a")])


class LineNumberTest(unittest.TestCase):
    """Số thứ tự dòng thơ của Wikisource (bản mẫu 'Số thứ tự', vd. Truyện Kiều) phải bị bỏ khi trích nội dung."""

    def test_line_numbers_removed(self):
        spec = importlib.util.spec_from_file_location("fetch_v5_ln", ROOT / "scripts" / "fetch_v5.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        html = ('<div class="mw-parser-output"><div class="poem"><p>Trăm năm trong cõi người ta,<br />'
                'Chữ tài chữ mệnh khéo là ghét nhau.<br />'
                '<span style="visibility: display;"><span class="OptionText" title="So thu tu"><span id="dong5"></span>'
                '<span style="position:absolute;" title="Số thứ tự">5</span></span></span>\tLạ gì bỉ sắc tư phong,<br />'
                'Trời xanh quen thói má hồng đánh ghen.</p></div></div>')
        poems = mod.extract_content(html)["poems"]
        self.assertEqual(len(poems), 1)
        self.assertNotRegex(poems[0], r"\d")
        self.assertIn("Lạ gì bỉ sắc tư phong,", poems[0].split("\n"))


class GenreExperimentTest(unittest.TestCase):
    """Thí nghiệm theo thể loại chạy trọn (style_only, CPU) trên kho giả có một tác giả viết hai thể loại."""

    def test_runs_and_reports_all_approaches(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("cần torch")
        import pandas as pd
        from config import TrainConfig
        from src.demo_data import generate_demo_corpus
        from src.genre_experiment import APPROACHES, run_genre_experiment
        with tempfile.TemporaryDirectory() as d:
            src = pd.read_csv(generate_demo_corpus(Path(d), lang="vi", n_authors=4, docs_per_author=40))
            authors = sorted(src["author"].unique())
            # a0, a1 chỉ viết thể loại x; a3 chỉ viết y; a2 viết cả hai (nửa số đoạn mỗi thể loại)
            src["genre"] = src["author"].map({authors[0]: "x", authors[1]: "x", authors[2]: "x", authors[3]: "y"})
            a2 = src.index[src["author"] == authors[2]]
            src.loc[a2[len(a2) // 2:], "genre"] = "y"
            src["doc_id"] = [f"doc{i // 2}" for i in range(len(src))]
            src["topic"] = src["genre"] + src["author"]
            src["n_words"] = src["text"].str.split().str.len()
            data = Path(d) / "g.csv"
            src.to_csv(data, index=False)
            cfg = TrainConfig(lang="vi", epochs=1, batch_size=16, num_function_words=20, style_hidden=32, style_out=16)
            out = Path(d) / "exp"
            res = run_genre_experiment(data, "t", cfg, seeds=[1, 2], fusions=["style_only"], short_lengths=[10],
                                       device_pref="cpu", out_root=out)
            self.assertEqual(set(res["approach"]), set(APPROACHES))
            self.assertEqual(set(res["seed"]), {1, 2})
            self.assertIn("style_only", set(res["model"]))
            cross = pd.read_csv(out / "seed_1" / "cross_genre.csv")
            # mô hình riêng thể loại x đoán đoạn thể loại y của a2 và ngược lại
            self.assertEqual(set(cross["author"]), {authors[2]})
            self.assertEqual(set(zip(cross["train_genre"], cross["test_genre"])), {("x", "y"), ("y", "x")})
            split = pd.read_csv(out / "seed_1" / "split.csv")
            self.assertEqual(split.groupby("doc_id")["split"].nunique().max(), 1)
            # chọn sai thể loại: tác giả chỉ viết x thì lọc theo y không thể đúng
            wrong = res[(res.approach == "chung_chon_sai") & (res.genre == "x") & (res.variant == "full")]
            right = res[(res.approach == "chung_loc_the_loai") & (res.genre == "x") & (res.variant == "full")]
            self.assertTrue((wrong["accuracy"].to_numpy() <= 0.5).all())
            self.assertTrue((wrong["n"].to_numpy() == right["n"].to_numpy()).all())
            self.assertTrue((out / "summary.md").read_text(encoding="utf-8").count("chung_loc_the_loai") >= 1)
            # chạy lại thì dùng kết quả cũ, không huấn luyện lại
            res2 = run_genre_experiment(data, "t", cfg, seeds=[1, 2], fusions=["style_only"], short_lengths=[10],
                                        device_pref="cpu", out_root=out)
            self.assertEqual(len(res2), len(res))


class DisplayNamesTest(unittest.TestCase):
    def test_reads_csv_with_bom(self):
        from src.explain import author_display_names
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "works.csv"
            p.write_text("author,author_name,work\nnam_cao,Nam Cao,Lão Hạc\n", encoding="utf-8-sig")
            self.assertEqual(author_display_names(p), {"nam_cao": "Nam Cao"})
