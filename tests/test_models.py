import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.models import ModelStore, StoreError
from tests.helpers import make_release, synthetic_dataset


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.ds = synthetic_dataset()
        self.net, self.meta = make_release(self.dir, self.ds)
        self.store = ModelStore(self.dir)

    def add_candidate(self, cid, val_acc=None, split=None, parent=None):
        meta = json.loads(json.dumps(self.meta))
        meta.update(run_id=cid, source="ui-train", parent=parent, created_utc=f"2026-02-0{len(cid) % 9 + 1}T00:00:00+00:00")
        if val_acc is not None:
            meta["evaluation"]["validation_accuracy"] = val_acc
        if split is not None:
            meta["dataset"]["split_sha256"] = split
        self.store.save_candidate(self.net, meta, cid)


class TestStore(StoreCase):
    def test_starts_on_release(self):
        self.assertEqual(self.store.active_id, "release")
        self.assertEqual(self.store.predictor.version_id, "release")
        self.assertEqual([v["id"] for v in self.store.list_versions()], ["release"])

    def test_gate_passes_when_equal_or_better_and_swaps_atomically(self):
        base = self.meta["evaluation"]["validation_accuracy"]
        self.add_candidate("cand-a", val_acc=base)
        before = self.store.predictor
        d = self.store.promote("cand-a")
        self.assertTrue(d["passed"], d)
        self.assertEqual(self.store.active_id, "cand-a")
        self.assertIsNot(self.store.predictor, before)
        self.assertEqual(self.store.history(), ["release"])

    def test_gate_blocks_regressions_and_incomparable_validation_sets(self):
        base = self.meta["evaluation"]["validation_accuracy"]
        self.add_candidate("worse", val_acc=base - 0.05)
        d = self.store.promote("worse")
        self.assertFalse(d["passed"]); self.assertIn("below", d["reasons"][0])
        self.assertEqual(self.store.active_id, "release")
        self.add_candidate("othersplit", val_acc=1.0, split="different")
        self.assertFalse(self.store.promote("othersplit")["passed"])

    def test_gate_blocks_user_holdout_regression(self):
        self.add_candidate("c1", val_acc=self.meta["evaluation"]["validation_accuracy"])
        d = self.store.evaluate_gate("c1", user_before=0.9, user_after=0.8)
        self.assertFalse(d["passed"])

    def test_cannot_promote_active_or_unknown(self):
        self.assertFalse(self.store.promote("release")["passed"])
        with self.assertRaises(StoreError):
            self.store.promote("does-not-exist")

    def test_rollback_and_persistence_across_restarts(self):
        self.add_candidate("cand-a", val_acc=self.meta["evaluation"]["validation_accuracy"])
        self.store.promote("cand-a")
        again = ModelStore(self.dir)                         # simulate a restart
        self.assertEqual(again.active_id, "cand-a")
        self.assertEqual(again.rollback(), "release")
        self.assertEqual(ModelStore(self.dir).active_id, "release")
        with self.assertRaises(StoreError):
            again.rollback()

    def test_registry_pointing_at_missing_model_falls_back(self):
        (self.dir / "registry.json").write_text(json.dumps({"active": "ghost", "history": []}))
        self.assertEqual(ModelStore(self.dir).active_id, "release")
        (self.dir / "registry.json").write_text("not json")
        self.assertEqual(ModelStore(self.dir).active_id, "release")

    def test_id_validation_blocks_path_traversal(self):
        for bad in ("../x", "a/b", "", "A", "x" * 80, "..", "c:\\x"):
            with self.assertRaises(StoreError, msg=bad):
                self.store.get(bad) if False else self.store.promote(bad)
        with self.assertRaises(StoreError):
            self.store.save_candidate(self.net, self.meta, "../escape")

    def test_delete_protects_active_and_rollback_targets(self):
        self.add_candidate("keep-a", val_acc=self.meta["evaluation"]["validation_accuracy"])
        self.add_candidate("keep-b", val_acc=self.meta["evaluation"]["validation_accuracy"])
        self.store.promote("keep-a"); self.store.promote("keep-b")
        for protected in ("keep-b", "keep-a", "release"):
            with self.assertRaises(StoreError):
                self.store.delete(protected)
        self.add_candidate("spare")
        self.store.delete("spare")
        self.assertNotIn("spare", [v["id"] for v in self.store.list_versions()])

    def test_prune_caps_candidates_but_keeps_protected(self):
        from app import models
        old = models.MAX_CANDIDATES
        models.MAX_CANDIDATES = 3
        self.addCleanup(setattr, models, "MAX_CANDIDATES", old)
        self.add_candidate("keep-a", val_acc=self.meta["evaluation"]["validation_accuracy"])
        self.store.promote("keep-a")
        for i in range(6):
            self.add_candidate(f"extra-{i}")
        ids = {p.stem for p in (self.dir / "candidates").glob("*.json")}
        self.assertIn("keep-a", ids)
        self.assertLessEqual(len(ids), 3)

    def test_unreadable_candidate_is_skipped_in_listing(self):
        self.add_candidate("good")
        (self.dir / "candidates" / "broken.json").write_text("{}")
        ids = [v["id"] for v in self.store.list_versions()]
        self.assertIn("good", ids); self.assertNotIn("broken", ids)


if __name__ == "__main__":
    unittest.main()
