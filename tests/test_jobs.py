import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from app.feedback import FeedbackStore
from app.jobs import DataUnavailable, JobBusy, JobError, JobManager, MIN_USER_SAMPLES
from app.models import ModelStore
from training.config import load_config
from tests.helpers import class_pixels, make_release, synthetic_dataset

CFG = load_config(Path(__file__).resolve().parent.parent / "configs" / "default.json")


class JobCase(unittest.TestCase):
    max_seconds = 120.0

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.ds = synthetic_dataset()
        make_release(self.dir, self.ds, epochs=6)
        self.store = ModelStore(self.dir)
        self.feedback = FeedbackStore(self.dir / "fb.jsonl")
        self.mgr = JobManager(self.store, self.feedback, lambda: self.ds, CFG, max_seconds=self.max_seconds)
        self.addCleanup(self._stop)

    def _stop(self):
        self.mgr.cancel(); self.mgr.wait(30)

    def run_job(self, kind, params, timeout=90):
        self.mgr.start(kind, params)
        self.mgr.wait(timeout)
        return self.mgr.status()

    def add_samples(self, n):
        rng = np.random.default_rng(0)
        for i in range(n):
            px = np.array(class_pixels(i % 9, rng), dtype=np.uint8)
            self.feedback.add(px, i % 9 + 1)


class TestTrainJob(JobCase):
    def test_train_job_streams_events_and_saves_a_candidate(self):
        st = self.run_job("train", {"hidden_sizes": [16], "epochs": 3, "batch_size": 32})
        job = st["job"]
        self.assertEqual(job["state"], "done", job)
        types = [e["type"] for e in st["events"]]
        self.assertEqual(types[0], "job_started"); self.assertEqual(types[-1], "job_finished")
        self.assertIn("init", types); self.assertEqual(types.count("epoch"), 3)
        epoch = next(e for e in st["events"] if e["type"] == "epoch")
        self.assertEqual(len(epoch["weight_stats"]), 4)                    # W0 b0 W1 b1
        self.assertEqual(epoch["tiles"]["n"], 16); self.assertEqual(epoch["tiles"]["shape"], [28, 28])
        self.assertIn("deltas", epoch["probe"]["trace"]["backward"])
        res = job["result"]
        self.assertEqual(res["parameter_count"], (784 * 16 + 16) + (16 * 9 + 9))
        self.assertIn(res["candidate_id"], [v["id"] for v in self.store.list_versions()])
        self.assertIn("passed", res["gate"])
        self.assertEqual(self.store.active_id, "release")                  # never auto-promoted

    def test_events_are_json_serialisable_and_ordered(self):
        import json
        st = self.run_job("train", {"hidden_sizes": [8], "epochs": 2})
        json.dumps(st)
        seqs = [e["seq"] for e in st["events"]]
        self.assertEqual(seqs, sorted(seqs)); self.assertEqual(len(set(seqs)), len(seqs))
        later = self.mgr.status(since=seqs[2])
        self.assertEqual([e["seq"] for e in later["events"]], seqs[3:])

    def test_validation_errors(self):
        bad = [{"hidden_sizes": []}, {"hidden_sizes": [1]}, {"hidden_sizes": [513]}, {"hidden_sizes": [8] * 4},
               {"hidden_sizes": "128"}, {"hidden_sizes": [True]}, {"hidden_sizes": [8.5]}, {"epochs": 0},
               {"epochs": 1000}, {"epochs": 1.5}, {"learning_rate": 5}, {"batch_size": 1}, {"nope": 1},
               {"augmentation": "yes"}, {"weight_decay": -1}, {"subset": 1_000_000}]
        for params in bad:
            with self.assertRaises(JobError, msg=str(params)):
                self.mgr.start("train", params)
        with self.assertRaises(JobError):
            self.mgr.start("explode", {})
        with self.assertRaises(JobError):
            self.mgr.start("train", [])
        self.assertFalse(self.mgr.status()["busy"])

    def test_param_cap(self):
        with mock.patch("app.jobs.MAX_PARAMS", 5000):
            with self.assertRaises(JobError):
                self.mgr.start("train", {"hidden_sizes": [64]})

    def test_cancel_stops_job_without_saving_a_candidate(self):
        self.mgr.start("train", {"hidden_sizes": [64, 32], "epochs": 60, "batch_size": 16})
        deadline = time.time() + 30
        while time.time() < deadline and not any(e["type"] == "epoch" for e in self.mgr.status()["events"]):
            time.sleep(0.02)
        self.assertTrue(self.mgr.cancel())
        self.mgr.wait(30)
        st = self.mgr.status()
        self.assertEqual(st["job"]["state"], "cancelled")
        self.assertEqual([v["id"] for v in self.store.list_versions()], ["release"])
        self.assertFalse(self.mgr.cancel())                                # nothing left to cancel

    def test_only_one_job_at_a_time(self):
        self.mgr.start("train", {"hidden_sizes": [64], "epochs": 60, "batch_size": 16})
        with self.assertRaises(JobBusy):
            self.mgr.start("sweep", {})

    def test_failed_job_is_reported_and_manager_survives(self):
        with mock.patch.object(JobManager, "_run_train", side_effect=RuntimeError("boom")):
            st = self.run_job("train", {"hidden_sizes": [8], "epochs": 1})
        self.assertEqual(st["job"]["state"], "failed"); self.assertEqual(st["job"]["error"], "boom")
        self.assertEqual(self.run_job("train", {"hidden_sizes": [8], "epochs": 1})["job"]["state"], "done")

    def test_data_unavailable(self):
        def boom():
            raise DataUnavailable("no data")
        mgr = JobManager(self.store, self.feedback, boom, CFG)
        with self.assertRaises(DataUnavailable):
            mgr.start("train", {})


class TestTimeLimit(JobCase):
    max_seconds = 0.0

    def test_time_limit_cancels(self):
        st = self.run_job("train", {"hidden_sizes": [16], "epochs": 5})
        self.assertEqual(st["job"]["state"], "cancelled")
        self.assertIn("time limit", st["job"]["result"]["reason"])


class TestFinetune(JobCase):
    def test_needs_enough_samples(self):
        self.add_samples(MIN_USER_SAMPLES - 1)
        with self.assertRaises(JobError):
            self.mgr.start("finetune", {})

    def test_finetune_reports_before_after_and_can_auto_promote(self):
        self.add_samples(18)
        st = self.run_job("finetune", {"epochs": 2})
        res = st["job"]["result"]
        self.assertEqual(st["job"]["state"], "done", st["job"])
        self.assertEqual(res["user_samples"] + res["user_holdout"], 18)
        self.assertGreater(res["user_holdout"], 0)
        for k in ("user_holdout_accuracy_before", "user_holdout_accuracy_after",
                  "mnist_validation_accuracy_before", "mnist_validation_accuracy_after"):
            self.assertIsNotNone(res[k], k)
        init = next(e for e in st["events"] if e["type"] == "init")
        self.assertEqual(init["user_samples"], res["user_samples"])
        if res["gate"]["passed"]:
            self.assertTrue(res["promoted"]); self.assertEqual(self.store.active_id, res["candidate_id"])
            cand = self.store.get(res["candidate_id"]).metadata
            self.assertEqual(cand["parent"], "release"); self.assertEqual(cand["source"], "ui-finetune")

    def test_auto_promote_off_never_swaps(self):
        self.add_samples(12)
        res = self.run_job("finetune", {"epochs": 1, "auto_promote": False})["job"]["result"]
        self.assertFalse(res["promoted"]); self.assertEqual(self.store.active_id, "release")

    def test_finetune_does_not_modify_the_served_model(self):
        self.add_samples(12)
        before = [p.copy() for p in self.store.predictor.network.parameters()]
        self.run_job("finetune", {"epochs": 2, "auto_promote": False})
        for a, b in zip(before, self.store.predictor.network.parameters()):
            np.testing.assert_array_equal(a, b)


class TestSweep(JobCase):
    def test_sweep_compares_architectures(self):
        st = self.run_job("sweep", {"architectures": [[8], [16, 8]], "epochs": 2, "subset": 360})
        res = st["job"]["result"]["results"]
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["parameter_count"], (784 * 8 + 8) + (8 * 9 + 9))
        self.assertEqual(res[1]["layer_sizes"], [784, 16, 8, 9])
        self.assertEqual(sum(e["type"] == "sweep_arch_done" for e in st["events"]), 2)
        self.assertEqual([v["id"] for v in self.store.list_versions()], ["release"])     # nothing saved

    def test_sweep_validation(self):
        for bad in ({"architectures": []}, {"architectures": [[8]] * 9}, {"architectures": [[1]]},
                    {"architectures": "x"}, {"epochs": 99}):
            with self.assertRaises(JobError, msg=str(bad)):
                self.mgr.start("sweep", bad)


if __name__ == "__main__":
    unittest.main()
