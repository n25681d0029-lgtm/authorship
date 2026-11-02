"""Kiểm thử hiệu chỉnh xác suất, cắt văn bản dài khi dự đoán, n-gram ký tự out-of-fold, quy tắc loại trừ của bộ tải."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.calibration import calibrate, ece, nll, softmax  # noqa: E402
from src.ngram_features import CharNgramStacker, stacking_groups  # noqa: E402


class CalibrationTest(unittest.TestCase):
    def test_temperature_fixes_overconfidence(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 4, 400)
        logits = rng.normal(0, 1, (400, 4))
        logits[np.arange(400), y] += 1.0          # tín hiệu yếu ...
        logits *= 6.0                              # ... nhưng logits bị phóng to: mô hình quá tự tin
        c = calibrate(logits, y, accept_rate=0.9)
        self.assertGreater(c["temperature"], 1.5)
        self.assertLess(c["val_nll_after"], c["val_nll_before"])
        self.assertLess(c["val_ece_after"], c["val_ece_before"])
        accepted = (softmax(logits, c["temperature"]).max(axis=1) >= c["reject_threshold"]).mean()
        self.assertAlmostEqual(accepted, 0.9, delta=0.02)

    def test_temperature_does_not_change_argmax(self):
        z = np.array([[2.0, 1.0, 0.1], [0.3, 0.2, 3.0]])
        self.assertTrue(np.array_equal(softmax(z, 5.0).argmax(1), softmax(z).argmax(1)))
        self.assertAlmostEqual(nll(z, np.array([0, 2])), nll(z, np.array([0, 2]), 1.0))
        self.assertEqual(ece(np.array([[1.0, 0.0]]), np.array([0])), 0.0)


class ClassBiasAndOversampleTest(unittest.TestCase):
    def test_bias_recovers_suppressed_class(self):
        from src.calibration import fit_class_bias
        rng = np.random.default_rng(1)
        y = np.repeat([0, 1, 2], 60)
        z = rng.normal(0, 0.3, (180, 3))
        z[np.arange(180), y] += 1.0
        z[:, 2] -= 1.5                      # lớp 2 bị mô hình "lờ đi"
        before = (z.argmax(1)[y == 2] == 2).mean()
        b = fit_class_bias(z, y)
        after = ((z + b).argmax(1)[y == 2] == 2).mean()
        self.assertLess(before, 0.2)
        self.assertGreater(after, 0.8)

    def test_balanced_loader_equalizes_authors(self):
        import torch  # noqa: F401
        from src.dataset import Collator, make_loader
        labels = [0] * 90 + [1] * 10
        loader = make_loader(["x"] * 100, np.zeros((100, 2), np.float32), labels, Collator(None, 8, "en"),
                             batch_size=100, shuffle=True, balanced=True, seed=0)
        drawn = np.concatenate([b["labels"].numpy() for _ in range(20) for b in loader])
        self.assertAlmostEqual((drawn == 1).mean(), 0.5, delta=0.05)


class SplitLongTextTest(unittest.TestCase):
    def test_long_text_is_chunked(self):
        from src.predict import split_long_text
        para = " ".join(f"Câu thứ {i} nói về một ngày mưa dài ở làng." for i in range(15))   # 150 âm tiết
        self.assertEqual(len(split_long_text(para)), 1)
        long = "\n\n".join([para] * 4)                                                      # 600 âm tiết
        chunks = split_long_text(long)
        self.assertEqual(len(chunks), 4)
        self.assertTrue(all(50 <= len(c.split()) <= 200 for c in chunks))


class NgramStackerTest(unittest.TestCase):
    def test_out_of_fold_probabilities(self):
        texts, labels, topics = [], [], []
        for a, (w1, w2) in enumerate([("mưa", "gió"), ("nắng", "trăng"), ("sông", "núi")]):
            for t in range(4):
                for k in range(6):
                    texts.append(f"tác phẩm {t} đoạn {k}: trời {w1} {w2}, {w1} {w1} rơi mãi trên mái nhà {a}.")
                    labels.append(a)
                    topics.append(f"{a}_{t}")
        df = pd.DataFrame({"text": texts, "author": labels, "topic": topics, "doc_id": topics})
        groups = stacking_groups(df)
        self.assertTrue(all(g.startswith("t:") for g in groups))          # đủ 4 tác phẩm/tác giả -> nhóm theo tác phẩm
        st = CharNgramStacker(n_labels=3, n_folds=4)
        oof = st.fit_transform(texts, labels, groups)
        self.assertEqual(oof.shape, (len(texts), 3))
        self.assertTrue(np.allclose(oof.sum(axis=1), 1.0, atol=1e-5))
        self.assertGreater((oof.argmax(1) == np.array(labels)).mean(), 0.8)
        self.assertTrue(np.allclose(st.transform(texts[:2]).sum(axis=1), 1.0, atol=1e-5))

    def test_few_topics_fall_back_to_documents(self):
        df = pd.DataFrame({"author": ["a"] * 4 + ["b"] * 4, "topic": ["x"] * 4 + ["y", "y", "z", "w"],
                           "doc_id": ["x1", "x2", "x3", "x4", "y1", "y2", "z1", "w1"]})
        g = stacking_groups(df)
        self.assertEqual(list(g[:4]), ["d:x1", "d:x2", "d:x3", "d:x4"])  # a chỉ có 1 tác phẩm
        self.assertEqual(list(g[4:]), ["t:y", "t:y", "t:z", "t:w"])


class TrainWithNgramTest(unittest.TestCase):
    def test_style_only_with_ngram_save_load_predict(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("cần torch")
        from config import TrainConfig
        from src.demo_data import generate_demo_corpus
        from src.predict import AuthorPredictor, evaluate_checkpoint
        from src.train import run_training
        with tempfile.TemporaryDirectory() as d:
            data = generate_demo_corpus(Path(d), lang="en", n_authors=3, docs_per_author=30)
            cfg = TrainConfig(fusion="style_only", char_ngram=True, epochs=3, batch_size=16,
                              num_function_words=20, style_hidden=32, style_out=16, patience=5)
            out = Path(d) / "ck"
            res = run_training(cfg, data, out_dir=out, device_pref="cpu")
            self.assertTrue((out / "ngram.joblib").exists())
            pred = AuthorPredictor.from_checkpoint(out, "cpu")
            self.assertIsNotNone(pred.stacker)
            self.assertGreater(pred.temperature, 0)
            self.assertIsNotNone(pred.reject_threshold)
            r = pred.predict(["The old house stood on the hill. " * 60], top_k=2)[0]   # 420 từ -> nhiều đoạn
            self.assertGreater(r["n_chunks"], 1)
            only = pred.predict(["The old house stood on the hill."], top_k=3, candidates=[pred.names[1]])[0]
            self.assertEqual(only["author"], pred.names[1])          # chỉ còn một ứng viên
            self.assertAlmostEqual(only["confidence"], 1.0, places=5)
            self.assertEqual([t["author"] for t in only["top_k"]], [pred.names[1]])
            self.assertAlmostEqual(sum(pred.predict_proba(["a b c d."])[0][0]), 1.0, places=5)
            m = evaluate_checkpoint(out, data, "test", "cpu")
            self.assertAlmostEqual(m["accuracy"], res["accuracy"], places=6)


class FetchExcludeTest(unittest.TestCase):
    def test_exact_and_prefix_rules(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from fetch_wikisource import excluded
        rules = ["=Nam Phong tạp chí", "Nam Phong tạp chí/Quyển I/Số 1/Tiểu-thuyết"]
        self.assertTrue(excluded("Nam Phong tạp chí", rules))
        self.assertFalse(excluded("Nam Phong tạp chí/Quyển I/Số 1/Luận-thuyết", rules))
        self.assertTrue(excluded("Nam Phong tạp chí/Quyển I/Số 1/Tiểu-thuyết/Truyện cái dấu đỏ", rules))


if __name__ == "__main__":
    unittest.main()
