"""Kiểm thử đầu-cuối với backbone nhỏ ngẫu nhiên (offline). Tự bỏ qua nếu thiếu torch/transformers."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False


@unittest.skipUnless(HAVE, "cần torch và transformers")
class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from config import TrainConfig
        from src.data_loader import load_corpus
        from src.demo_data import generate_demo_corpus
        from src.tiny_backbone import build_tiny_backbone
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.data = generate_demo_corpus(root, lang="en", n_authors=3, docs_per_author=24)
        cls.backbone = build_tiny_backbone(load_corpus(cls.data)["text"].tolist(), root / "tiny")
        cls.TrainConfig = TrainConfig
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _cfg(self, fusion):
        return self.TrainConfig(fusion=fusion, backbone=str(self.backbone), epochs=2, batch_size=8,
                                max_length=64, num_function_words=20, style_hidden=32, style_out=16,
                                fused_dim=32, patience=5)

    def test_all_fusions_train_save_load_predict(self):
        from src.predict import AuthorPredictor, evaluate_checkpoint
        from src.train import run_training
        for fusion in ("style_only", "lm_only", "concat", "gated"):
            with self.subTest(fusion=fusion):
                out = self.root / f"ck_{fusion}"
                res = run_training(self._cfg(fusion), self.data, out_dir=out, device_pref="cpu")
                self.assertTrue((out / "model.pt").exists())
                self.assertTrue((out / "meta.json").exists())
                pred = AuthorPredictor.from_checkpoint(out, "cpu")
                r = pred.predict(["The old house stood on the hill, and the wind was cold."], top_k=2)[0]
                self.assertIn(r["author"], pred.names)
                self.assertAlmostEqual(sum(t["prob"] for t in pred.predict(["x y z. a b c."], top_k=3)[0]["top_k"]), 1.0, places=4)
                m = evaluate_checkpoint(out, self.data, "test", "cpu")
                self.assertAlmostEqual(m["accuracy"], res["accuracy"], places=6)   # khớp lúc huấn luyện
                if fusion == "gated":
                    self.assertIn("lm_weight", r)

    def test_trace_matches_predict_for_all_fusions(self):
        """Tab trực quan hóa (src/pipeline_trace) phải cho đúng các con số của lần dự đoán thật, với mọi chế độ."""
        import numpy as np
        from src.pipeline_trace import decide, trace
        from src.predict import AuthorPredictor
        from src.train import run_training
        long_text = "\n\n".join(["The old house stood on the hill, and the wind was cold. " * 9] * 3)
        for fusion in ("style_only", "lm_only", "concat", "gated"):
            with self.subTest(fusion=fusion):
                out = self.root / f"ck_trace_{fusion}"
                run_training(self._cfg(fusion), self.data, out_dir=out, device_pref="cpu")
                pred = AuthorPredictor.from_checkpoint(out, "cpu")
                for text in ("A short line, with “quotes” – and a dash.", long_text):
                    tr = trace(pred, text)
                    r = pred.predict([text], top_k=len(pred.names))[0]
                    expect = {t["author"]: t["prob"] for t in r["top_k"]}
                    for j, a in enumerate(tr["names"]):
                        self.assertAlmostEqual(tr["probs"][j], expect[a], places=5)
                    self.assertEqual(len(tr["chunks"]), r["n_chunks"])
                    for c in tr["chunks"]:   # các phần đóng góp cộng lại đúng bằng logit của mô hình
                        logits, _ = pred.predict_logits([c["text"]])
                        np.testing.assert_allclose(sum(c["contrib"].values()) + c["intercept"] + pred.class_bias,
                                                   logits[0], atol=1e-4)
                        self.assertEqual(c["tokens"] is None, fusion == "style_only")
                d = decide(tr["probs"], tr["names"], candidates=tr["names"][:2], threshold=None)
                self.assertIn(d["author"], tr["names"][:2])
                self.assertAlmostEqual(float(d["probs_after"].sum()), 1.0, places=6)
                self.assertEqual(decide(tr["probs"], tr["names"], threshold=1.01)["author"], "unknown")

    def test_ensemble_predict_and_trace_agree(self):
        """Bộ gộp (ensemble.joblib): predict dùng mọi thành viên, trace cho đúng xác suất đó, use_ensemble=False như cũ."""
        import numpy as np
        from src.data_loader import load_corpus
        from src.ensemble import EnsembleHead, NgramMember
        from src.pipeline_trace import trace
        from src.predict import AuthorPredictor
        from src.train import run_training
        main, extra = self.root / "ck_ens_main", self.root / "ck_ens_extra"
        run_training(self._cfg("concat"), self.data, out_dir=main, device_pref="cpu")
        run_training(self._cfg("style_only"), self.data, out_dir=extra, device_pref="cpu")
        single = AuthorPredictor.from_checkpoint(main, "cpu")
        df = load_corpus(self.data)
        y = df["author"].map({a: i for i, a in enumerate(single.names)}).to_numpy()
        m = NgramMember("char_svm").fit(df["text"].tolist(), y, len(single.names))
        EnsembleHead([m], {"hybrid": 0.6, "char_svm": 0.4}, [str(extra)], temperature=1.3,
                     reject_threshold=0.2).save(main / "ensemble.joblib")
        pred = AuthorPredictor.from_checkpoint(main, "cpu")
        self.assertEqual(len(pred.extra_models), 1)
        self.assertAlmostEqual(pred.reject_threshold, 0.2)
        text = "The old house stood on the hill, and the wind was cold. It rained all night."
        r = pred.predict([text], top_k=len(pred.names))[0]
        expect = {t["author"]: t["prob"] for t in r["top_k"]}
        tr = trace(pred, text)
        for j, a in enumerate(tr["names"]):
            self.assertAlmostEqual(tr["probs"][j], expect[a], places=5)
        self.assertEqual(set(tr["chunks"][0]["members"]), {"hybrid", "char_svm"})
        r1 = single.predict([text], top_k=len(pred.names))[0]
        self.assertFalse(np.allclose([t["prob"] for t in r1["top_k"]], [expect[t["author"]] for t in r1["top_k"]]))
        self.assertIsNone(AuthorPredictor.from_checkpoint(main, "cpu", use_ensemble=False).ensemble)

    def test_reject_below_threshold(self):
        from src.predict import AuthorPredictor
        from src.train import run_training
        out = self.root / "ck_reject"
        run_training(self._cfg("concat"), self.data, out_dir=out, device_pref="cpu")
        r = AuthorPredictor.from_checkpoint(out, "cpu").predict(["some text here to test."], reject_below=1.01)[0]
        self.assertEqual(r["author"], "unknown")


if __name__ == "__main__":
    unittest.main()
