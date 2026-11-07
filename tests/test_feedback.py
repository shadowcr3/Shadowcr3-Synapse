import os
import stat
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.feedback import (FeedbackError, FeedbackFull, FeedbackStore, HOLDOUT_FRACTION, MIN_HOLDOUT, MIN_TOTAL,
                          holdout_mask, readiness)


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
        self.assertEqual(list(y), [2, 6])
        self.assertEqual(len(ids), 2)

    def test_duplicates_are_ignored(self):
        self.store.add(px(1), 3)
        r = self.store.add(px(1), 3)
        self.assertTrue(r["duplicate"]); self.assertEqual(self.store.stats()["total"], 1)
        self.assertFalse(self.store.add(px(1), 4)["duplicate"])

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


def make(n_per_digit, digits=range(9)):
    ids, labels = [], []
    for d in digits:
        for k in range(n_per_digit):
            ids.append(f"{d:02d}{k:02d}".ljust(16, "a"))
            labels.append(d)
    return ids, np.array(labels)


class TestHoldout(unittest.TestCase):
    def test_split_is_stratified_per_digit(self):
        ids, labels = make(8)
        mask = holdout_mask(ids, labels)
        for d in range(9):
            self.assertEqual(int(mask[labels == d].sum()), round(HOLDOUT_FRACTION * 8))
            self.assertLess(int(mask[labels == d].sum()), 8)

    def test_split_is_deterministic_and_independent_of_order(self):
        ids, labels = make(6)
        a = {i for i, m in zip(ids, holdout_mask(ids, labels)) if m}
        perm = np.random.default_rng(0).permutation(len(ids))
        ids2, labels2 = [ids[i] for i in perm], labels[perm]
        b = {i for i, m in zip(ids2, holdout_mask(ids2, labels2)) if m}
        self.assertEqual(a, b)
        self.assertEqual(a, {i for i, m in zip(ids, holdout_mask(ids, labels)) if m})

    def test_a_held_out_drawing_stays_held_out_as_more_are_added(self):
        ids, labels = make(6)
        before = {i for i, m in zip(ids, holdout_mask(ids, labels)) if m}
        more_ids, more_labels = make(8)
        after = {i for i, m in zip(more_ids, holdout_mask(more_ids, more_labels)) if m}
        self.assertEqual(len(before & set(ids)), len(before))
        self.assertGreaterEqual(len(after), len(before))

    def test_singletons_are_never_held_out_and_training_keeps_at_least_one(self):
        ids, labels = make(1)
        self.assertFalse(holdout_mask(ids, labels).any())
        ids, labels = make(2)
        mask = holdout_mask(ids, labels)
        for d in range(9):
            self.assertEqual(int(mask[labels == d].sum()), 1)

    def test_readiness_thresholds(self):
        ids, labels = make(3)
        r = readiness(ids, labels)
        self.assertFalse(r["ready"]); self.assertTrue(any(str(MIN_TOTAL) in x for x in r["reasons"]))
        ids, labels = make(6)
        r = readiness(ids, labels)
        self.assertTrue(r["ready"], r); self.assertGreaterEqual(r["holdout"], MIN_HOLDOUT)
        self.assertEqual(r["train"] + r["holdout"], r["total"])

    def test_many_samples_of_one_digit_are_not_enough(self):
        ids, labels = make(60, digits=[3])
        r = readiness(ids, labels)
        self.assertFalse(r["ready"]); self.assertTrue(any("cover 1 digits" in x for x in r["reasons"]))

    def test_empty(self):
        self.assertFalse(readiness([], np.array([]))["ready"])

    def test_store_stats_report_readiness(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as t:
            store = FeedbackStore(Path(t) / "s.jsonl")
            for i in range(MIN_TOTAL + 6):
                store.add(px(i), i % 9 + 1)
            r = store.stats()["readiness"]
            self.assertTrue(r["ready"], r)
            self.assertEqual(r["total"], MIN_TOTAL + 6)


if __name__ == "__main__":
    unittest.main()
