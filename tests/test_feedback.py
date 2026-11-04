import os
import stat
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.feedback import FeedbackError, FeedbackFull, FeedbackStore


def px(seed):
    return np.random.default_rng(seed).integers(0, 256, 784).astype(np.uint8)


class TestFeedbackStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "sub" / "samples.jsonl"
        self.store = FeedbackStore(self.path)

    def test_add_stats_and_arrays_roundtrip(self):
        r = self.store.add(px(1), 3)
        self.assertEqual((r["duplicate"], r["count"]), (False, 1))
        self.store.add(px(2), 7)
        self.assertEqual(self.store.stats()["per_digit"]["3"], 1)
        X, y, ids = self.store.arrays(range(1, 10))
        self.assertEqual(X.shape, (2, 784))
        np.testing.assert_array_equal(X[0], px(1))
        self.assertEqual(list(y), [2, 6])                      # digit -> class index (digit - 1)
        self.assertEqual(len(ids), 2)

    def test_duplicates_are_ignored(self):
        self.store.add(px(1), 3)
        r = self.store.add(px(1), 3)
        self.assertTrue(r["duplicate"]); self.assertEqual(self.store.stats()["total"], 1)
        self.assertFalse(self.store.add(px(1), 4)["duplicate"])   # same pixels, different label

    def test_validation(self):
        for bad in (0, 10, -1, 99):
            with self.assertRaises(FeedbackError):
                self.store.add(px(1), bad)
        with self.assertRaises(FeedbackError):
            self.store.add(px(1).astype(np.int64), 3)
        with self.assertRaises(FeedbackError):
            self.store.add(px(1)[:100], 3)

    def test_limit(self):
        small = FeedbackStore(self.path, max_samples=2)
        small.add(px(1), 1); small.add(px(2), 1)
        with self.assertRaises(FeedbackFull):
            small.add(px(3), 1)

    def test_delete_and_clear(self):
        sid = self.store.add(px(1), 3)["id"]
        self.store.add(px(2), 4)
        self.assertTrue(self.store.delete(sid)); self.assertFalse(self.store.delete(sid))
        self.assertEqual(self.store.clear(), 1)
        self.assertEqual(self.store.stats()["total"], 0)

    def test_malformed_lines_are_skipped(self):
        self.store.add(px(1), 3)
        with open(self.path, "a") as f:
            f.write("not json\n" + '{"id":"x","ts":"t","label":3,"pixels":"AAAA"}\n' + '{"id":"y"}\n')
        self.assertEqual(self.store.stats()["total"], 1)

    def test_file_is_private(self):
        self.store.add(px(1), 3)
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)

    def test_empty_store(self):
        X, y, ids = self.store.arrays(range(1, 10))
        self.assertEqual((X.shape, y.shape, ids), ((0, 784), (0,), []))


if __name__ == "__main__":
    unittest.main()
