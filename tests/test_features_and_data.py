"""Kiểm thử phần không cần torch: tiền xử lý, đặc trưng phong cách, đọc dữ liệu, baseline."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import build_label_map, detect_format, load_corpus, make_splits  # noqa: E402
from src.demo_data import generate_demo_corpus  # noqa: E402
from src.feature_extractor import BASE_FEATURES, StyleFeatureExtractor  # noqa: E402
from src.text_utils import normalize, split_sentences, tokenize_words  # noqa: E402


class TextUtilsTest(unittest.TestCase):
    def test_tokenize_keeps_vietnamese_and_apostrophes(self):
        self.assertEqual(tokenize_words("Tôi đi học, don't go."), ["Tôi", "đi", "học", "don't", "go"])

    def test_normalize_nfc(self):
        decomposed = "ê"          # e + dấu mũ rời
        self.assertEqual(normalize(decomposed), "ê")

    def test_normalize_transcription_conventions(self):
        self.assertEqual(normalize("văn-minh quốc-ngữ"), "văn minh quốc ngữ")
        self.assertEqual(normalize("- Ông giáo ạ!\nnăm 1934-1935 – rồi"), "— Ông giáo ạ!\nnăm 1934-1935 — rồi")
        self.assertEqual(normalize("«Tang a ta»"), "“Tang a ta”")

    def test_split_sentences(self):
        self.assertEqual(len(split_sentences("Một câu. Hai câu! Ba câu?")), 3)


class FeatureTest(unittest.TestCase):
    TEXTS = ["The cat sat on the mat. It was happy, very happy!",
             "A dog barked; the night was long and the road was dark.",
             "We went to the market and bought some bread, cheese and wine."]

    def test_known_values(self):
        ext = StyleFeatureExtractor(5)
        raw = ext._raw("tôi đi học rồi tôi đi chơi")
        names = BASE_FEATURES
        self.assertAlmostEqual(raw[names.index("ttr")], 5 / 7, places=6)
        self.assertAlmostEqual(raw[names.index("hapax_ratio")], 3 / 7, places=6)
        self.assertAlmostEqual(raw[names.index("yule_k")], 1e4 * (11 - 7) / 49, places=4)

    def test_shape_and_determinism(self):
        ext = StyleFeatureExtractor(10).fit(self.TEXTS)
        a, b = ext.transform(self.TEXTS), ext.transform(self.TEXTS)
        self.assertEqual(a.shape, (3, ext.dim))
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.isfinite(a).all())

    def test_empty_text_does_not_crash(self):
        ext = StyleFeatureExtractor(10).fit(self.TEXTS)
        self.assertTrue(np.isfinite(ext.transform([""])).all())

    def test_save_load_roundtrip(self):
        ext = StyleFeatureExtractor(10).fit(self.TEXTS)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.json"
            ext.save(p)
            again = StyleFeatureExtractor.load(p)
        self.assertTrue(np.allclose(ext.transform(self.TEXTS), again.transform(self.TEXTS)))

    def test_explain_returns_top_features(self):
        ext = StyleFeatureExtractor(10).fit(self.TEXTS)
        ev = ext.explain(self.TEXTS[0], top=3)
        self.assertEqual(len(ev), 3)
        self.assertGreaterEqual(abs(ev[0]["z"]), abs(ev[-1]["z"]))


class DataTest(unittest.TestCase):
    def test_csv_roundtrip_and_split(self):
        with tempfile.TemporaryDirectory() as d:
            path = generate_demo_corpus(Path(d), lang="en", n_authors=3, docs_per_author=20)
            self.assertEqual(detect_format(path), "csv")
            df = make_splits(load_corpus(path), 0.15, 0.15, 42)
            self.assertEqual(set(df["split"]), {"train", "val", "test"})
            self.assertEqual(len(build_label_map(df)), 3)
            # mọi tác giả đều có mặt trong train
            self.assertEqual(df[df["split"] == "train"]["author"].nunique(), 3)

    def test_topic_holdout(self):
        with tempfile.TemporaryDirectory() as d:
            path = generate_demo_corpus(Path(d), lang="vi", n_authors=3, docs_per_author=30)
            df = make_splits(load_corpus(path), 0.15, 0.15, 1, test_topics=["thethao"])
            test_topics = set(df[df["split"] == "test"]["topic"])
            train_topics = set(df[df["split"] != "test"]["topic"])
            self.assertEqual(test_topics, {"thethao"})
            self.assertNotIn("thethao", train_topics)

    def test_folder_and_c50_layouts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for split in ("C50train", "C50test"):
                for a in ("alice", "bob"):
                    (root / split / a).mkdir(parents=True)
                    for i in range(3):
                        (root / split / a / f"{i}.txt").write_text(f"{a} writes document number {i} here.", encoding="utf-8")
            self.assertEqual(detect_format(root), "c50")
            df = load_corpus(root)
            self.assertEqual(set(df["split"]), {"train", "test"})
            flat = root / "flat"
            for a in ("alice", "bob"):
                (flat / a).mkdir(parents=True)
                for i in range(6):
                    (flat / a / f"{i}.txt").write_text(f"{a} writes document number {i} here.", encoding="utf-8")
            self.assertEqual(detect_format(flat), "folder")
            self.assertEqual(len(load_corpus(flat)), 12)

    def test_baselines_learn_demo_data(self):
        from src.baselines import run_baselines
        with tempfile.TemporaryDirectory() as d:
            path = generate_demo_corpus(Path(d), lang="en", n_authors=4, docs_per_author=40)
            df = make_splits(load_corpus(path), 0.15, 0.15, 42)
            res = run_baselines(df, build_label_map(df))
        for name, r in res.items():
            self.assertGreater(r["test"]["accuracy"], 0.5, name)   # ngẫu nhiên là 0.25


if __name__ == "__main__":
    unittest.main()
