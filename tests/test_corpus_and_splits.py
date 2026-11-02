"""Kiểm thử dựng kho đoạn văn, chia theo bài (không rò rỉ), giữ lại chủ đề và lệnh experiment."""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus_builder import _allocate, build_corpus, pack_passages  # noqa: E402
from src.data_loader import check_group_leak, load_corpus, make_splits  # noqa: E402
from src.demo_data import generate_demo_corpus  # noqa: E402


def _sentence(i: int) -> str:
    return f"Câu thứ {i} nói về một ngày mưa dài ở làng."      # 10 từ


def _fake_corpus(n_authors=3, works=4, docs=4, passages_per_doc=6) -> pd.DataFrame:
    rows = []
    for a in range(n_authors):
        for w in range(works):
            for d in range(docs):
                for p in range(passages_per_doc):
                    rows.append({"text": f"tác giả {a} viết tác phẩm {w} bài {d} đoạn {p} " * 8,
                                 "author": f"a{a}", "topic": f"a{a}_w{w}", "doc_id": f"a{a}/w{w}/d{d}"})
    df = pd.DataFrame(rows)
    df["split"] = None
    return df


class PackTest(unittest.TestCase):
    def test_lengths_within_bounds(self):
        paras = [" ".join(_sentence(i) for i in range(k)) for k in (1, 3, 7, 30, 2, 2, 12, 1)]
        out = pack_passages(paras, min_words=50, max_words=200, target_words=150)
        self.assertTrue(out)
        for p in out:
            self.assertGreaterEqual(len(p.split()), 50)
            self.assertLessEqual(len(p.split()), 200)

    def test_long_paragraph_cut_at_sentences(self):
        para = " ".join(_sentence(i) for i in range(50))           # 500 từ, một đoạn
        out = pack_passages([para], 50, 200, 150)
        self.assertTrue(all(p.endswith(".") for p in out))
        self.assertTrue(all(len(p.split()) <= 200 for p in out))

    def test_allocate_spreads_across_topics(self):
        self.assertEqual(_allocate({"a": 100, "b": 5, "c": 100}, 60), {"a": 28, "b": 5, "c": 27})


class BuildCorpusTest(unittest.TestCase):
    def test_build_from_author_work_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for a in ("tac_gia_a", "tac_gia_b"):
                for w in ("tac_pham_1", "tac_pham_2"):
                    (root / a / w).mkdir(parents=True)
                    for c in range(3):
                        body = "\n\n".join(" ".join(_sentence(i + 100 * c) + f" {a} {w}" for i in range(j, j + 6))
                                           for j in range(0, 60, 6))
                        (root / a / w / f"chuong_{c}.txt").write_text(body, encoding="utf-8")
            # một bài trùng nguyên văn, phải bị bỏ khi khử trùng lặp
            dup = (root / "tac_gia_a" / "tac_pham_1" / "chuong_0.txt").read_text(encoding="utf-8")
            (root / "tac_gia_a" / "tac_pham_2" / "ban_sao.txt").write_text(dup, encoding="utf-8")
            df, stats = build_corpus(root, per_author=10, min_per_author=5)
        self.assertEqual(stats["n_authors"], 2)
        self.assertGreater(stats["duplicate_paragraphs_removed"], 0)
        self.assertEqual(df.groupby("author").size().tolist(), [10, 10])
        self.assertEqual(set(df["topic"]), {"tac_pham_1", "tac_pham_2"})
        self.assertTrue(df["n_words"].between(50, 200).all())


class LongPageTest(unittest.TestCase):
    def test_long_page_split_into_sections(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for a in ("x", "y"):
                (root / a / "tieu_thuyet").mkdir(parents=True)
                body = "\n\n".join(" ".join(_sentence(i + 20 * j) + f" {a}" for i in range(15)) for j in range(50))
                (root / a / "tieu_thuyet" / "ca_cuon.txt").write_text(body, encoding="utf-8")
            df, _ = build_corpus(root, per_author=1000, min_per_author=5, max_doc_passages=10)
        # 50 đoạn văn x 165 từ -> 50 đoạn mẫu mỗi tác giả -> 5 phần
        self.assertEqual(df.groupby("author")["doc_id"].nunique().tolist(), [5, 5])
        self.assertTrue(df["doc_id"].str.contains("#").all())


class GroupSplitTest(unittest.TestCase):
    def test_random_split_keeps_docs_together(self):
        df = make_splits(_fake_corpus(), 0.15, 0.15, 42)
        self.assertEqual(check_group_leak(df), [])
        self.assertEqual(set(df["split"]), {"train", "val", "test"})
        self.assertEqual(df[df["split"] == "test"]["author"].nunique(), 3)

    def test_topic_holdout_unseen_topics(self):
        df = make_splits(_fake_corpus(), 0.15, 0.15, 7, topic_holdout=0.3)
        self.assertEqual(check_group_leak(df), [])
        test_topics = set(df.loc[df["split"] == "test", "topic"])
        other = set(df.loc[df["split"] != "test", "topic"])
        self.assertFalse(test_topics & other)
        # mỗi tác giả có mặt ở test và vẫn còn tác phẩm để train
        self.assertEqual(df[df["split"] == "test"]["author"].nunique(), 3)
        self.assertEqual(df[df["split"] == "train"]["author"].nunique(), 3)


class ExperimentTest(unittest.TestCase):
    def test_style_only_experiment_runs(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("cần torch")
        from config import TrainConfig
        from src.experiment import run_experiment
        with tempfile.TemporaryDirectory() as d:
            data = generate_demo_corpus(Path(d), lang="vi", n_authors=3, docs_per_author=40)
            cfg = TrainConfig(lang="vi", epochs=2, batch_size=16, num_function_words=20,
                              style_hidden=32, style_out=16)
            res = run_experiment(data, "t", cfg, fusions=["style_only"], short_lengths=[10],
                                 device_pref="cpu", out_root=Path(d) / "exp")
            self.assertTrue((Path(d) / "exp" / "summary.md").exists())
            self.assertTrue((Path(d) / "exp" / "random" / "confusion" / "char_ngram_svm_confusion.csv").exists())
            pp = pd.read_csv(Path(d) / "exp" / "cross_topic" / "predictions.csv", index_col=0)
            self.assertIn("style_only", pp.columns)
            self.assertEqual(len(pp), int((res[(res.scenario == "cross_topic") & (res.variant == "full")].n.iloc[0])))
            self.assertEqual(len(load_corpus(data)), 120)
        self.assertEqual(set(res["scenario"]), {"random", "cross_topic"})
        self.assertEqual(set(res["model"]), {"burrows_delta", "char_ngram_svm", "stylometry_logreg", "style_only"})
        self.assertEqual(set(res["variant"]), {"full", "first_10w"})


if __name__ == "__main__":
    unittest.main()
