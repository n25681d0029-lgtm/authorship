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

    def test_reject_below_threshold(self):
        from src.predict import AuthorPredictor
        from src.train import run_training
        out = self.root / "ck_reject"
        run_training(self._cfg("concat"), self.data, out_dir=out, device_pref="cpu")
        r = AuthorPredictor.from_checkpoint(out, "cpu").predict(["some text here to test."], reject_below=1.01)[0]
        self.assertEqual(r["author"], "unknown")


if __name__ == "__main__":
    unittest.main()
