"""Kiểm thử các chức năng thêm sau v5: lịch sử/yêu thích, hướng dẫn văn phong, bộ gộp, so sánh hai văn bản."""
from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compare import lookup, pair_calibration, similarity  # noqa: E402
from src.ensemble import _grid, bootstrap_diff, fit_weights  # noqa: E402
from src.history import History, group_label  # noqa: E402
from src.style_guide import build_guides, check_draft, outline, poem_form, text_stats  # noqa: E402


class HistoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.h = History(Path(self.tmp.name) / "h.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_list_dedup_favorite_cleanup(self):
        now = datetime(2026, 11, 7, 12, 0)
        a = self.h.add("dinh_danh", "A", "văn bản một", {"top": [["x", 0.9]]}, now=now - timedelta(days=40))
        b = self.h.add("dinh_danh", "B", "văn bản hai", now=now - timedelta(days=3))
        self.assertEqual(self.h.add("dinh_danh", "B2", "văn bản hai", now=now), b)     # bấm lại không tạo bản trùng
        self.assertEqual(len(self.h.list("all", now=now)), 2)
        self.assertEqual([e["id"] for e in self.h.list("week", now=now)], [b])
        self.h.set_favorite(a, True)
        self.h.keep_days = 30
        self.assertEqual(self.h.cleanup(now=now), 0)            # mục cũ là yêu thích nên được giữ
        self.h.set_favorite(a, False)
        self.assertEqual(self.h.cleanup(now=now), 1)
        self.assertEqual([e["title"] for e in self.h.list(now=now)], ["B2"])
        self.h.keep_days = 0
        self.assertEqual(self.h.cleanup(now=now + timedelta(days=999)), 0)  # 0 = không tự xóa

    def test_search_note_delete(self):
        i = self.h.add("viet", "Nam Cao: bánh mì", "Hắn ăn bánh mì.")
        self.h.set_note(i, "hay")
        self.assertEqual(len(self.h.list(search="hay")), 1)
        self.assertEqual(len(self.h.list(kind="dinh_danh")), 0)
        self.h.delete(i)
        self.assertIsNone(self.h.get(i))

    def test_group_label(self):
        d = datetime(2026, 11, 7)
        self.assertEqual(group_label(d, "month"), "Tháng 11/2026")
        self.assertEqual(group_label(d, "year"), "Năm 2026")
        self.assertTrue(group_label(d, "week").startswith("Tuần 45/2026"))


class StyleGuideTest(unittest.TestCase):
    def test_poem_form(self):
        self.assertEqual(poem_form([6, 8, 6, 8, 6, 8]), "lục bát")
        self.assertEqual(poem_form([7, 7, 7, 7]), "bảy chữ (thất ngôn)")
        self.assertTrue(poem_form([3, 9, 5, 12]).startswith("tự do"))

    def test_guides_and_draft(self):
        a = ["Cô Hai đi chợ, chớ không ở nhà đặng nấu cơm. Bởi vậy ông Hương giận lắm.",
             "— Cháu đi đâu vậy? Cô Hai hỏi. Thằng nhỏ thưa: chớ cháu đi học, đặng mai thi."] * 4
        b = ["Hắn ăn. Hắn uống. Thị Nở cười. Cái nhà ấy buồn thật!", "Hắn nghĩ mãi. Hắn khóc. Cái đời hắn thế đấy!"] * 4
        g = build_guides(a + b, ["x"] * 8 + ["y"] * 8, ["tu_su"] * 16, min_count=2)
        gx, gy = g[("tu_su", "x")], g[("tu_su", "y")]
        self.assertIn("chớ", [w for w, *_ in gx.signature_words])
        self.assertIn("hắn", [w for w, *_ in gy.signature_words])
        self.assertNotIn("hai", [w for w, *_ in gx.signature_words])        # "Hai" luôn viết hoa: tên riêng, bị bỏ
        self.assertLess(gy.stats["do_dai_cau"], gx.stats["do_dai_cau"])
        r = check_draft("Hắn ăn. Hắn cười!", gx)
        self.assertTrue(r["tips"])
        self.assertIn("chớ", r["try_words"])
        o = outline("bánh mì", gx, 200)
        self.assertEqual(sum(p["so_chu"] for p in o["cac_phan"]), 200)

    def test_text_stats_poem(self):
        s = text_stats("Trăm năm trong cõi người ta\nChữ tài chữ mệnh khéo là ghét nhau\n" * 3, "tho")
        self.assertEqual(s["the_tho"], "lục bát")
        self.assertNotIn("loi_thoai", s)


class EnsembleMathTest(unittest.TestCase):
    def test_grid_sums_to_one(self):
        for w in _grid(["a", "b", "c"], 0.25):
            self.assertAlmostEqual(sum(w.values()), 1.0)

    def test_fit_weights_prefers_good_member(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 300)
        good = np.log(np.full((300, 3), 0.1)); good[np.arange(300), y] = np.log(0.8)
        bad = np.log(np.full((300, 3), 1 / 3))
        w, _ = fit_weights({"good": good, "bad": bad}, y, 0.1)
        self.assertGreaterEqual(w["good"], 0.9)

    def test_bootstrap_diff(self):
        y = np.array([0, 1, 0, 1] * 25)
        a = np.zeros(100, int)
        d, lo, hi = bootstrap_diff(y, a, y.copy(), np.arange(100) // 4)
        self.assertAlmostEqual(d, 0.5)
        self.assertLessEqual(lo, 0.5 + 1e-9)
        self.assertGreaterEqual(hi, 0.5 - 1e-9)


class CompareTest(unittest.TestCase):
    def test_similarity_and_calibration(self):
        self.assertAlmostEqual(similarity(np.array([1.0, 0, 0]), np.array([1.0, 0, 0])), 1.0)
        self.assertAlmostEqual(similarity(np.array([1.0, 0]), np.array([0, 1.0])), 0.0)
        rng = np.random.default_rng(1)
        authors = np.repeat(np.arange(5), 40)
        probs = np.full((200, 5), 0.05)
        probs[np.arange(200), authors] = 0.8                  # mỗi đoạn được đoán đúng tác giả
        probs = probs + rng.uniform(0, 0.01, probs.shape)
        cal = pair_calibration(probs / probs.sum(1, keepdims=True), authors.astype(str))
        hi = lookup(cal, 0.95)
        lo = lookup(cal, 0.5)     # khác tác giả: 2·√(0,8·0,05) + 3·0,05 ≈ 0,55
        self.assertGreater(hi["ty_le_cung"], 0.9)
        self.assertLess(lo["ty_le_cung"], 0.1)


if __name__ == "__main__":
    unittest.main()
