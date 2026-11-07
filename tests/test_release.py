import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.verify_release import verify

REPO = Path(__file__).resolve().parent.parent
HAVE_RELEASE = (REPO / "models" / "synapse_v1.npz").exists() and (REPO / "experiments" / "ood_eval.json").exists()


@unittest.skipUnless(HAVE_RELEASE, "no release artifacts in this checkout")
class TestReleaseConsistency(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        meta = json.loads((REPO / "models" / "synapse_v1.json").read_text())
        self.run_id = meta["run_id"]
        shutil.copytree(REPO / "models", self.root / "models", ignore=shutil.ignore_patterns("candidates", "registry.json"))
        (self.root / "experiments" / "runs").mkdir(parents=True)
        shutil.copytree(REPO / "experiments" / "runs" / self.run_id, self.root / "experiments" / "runs" / self.run_id)
        for f in ("ood_eval.json", "ood_eval.md", "pipeline_eval.json", "pipeline_eval.md"):
            shutil.copy(REPO / "experiments" / f, self.root / "experiments" / f)
        (self.root / "configs").mkdir()
        shutil.copy(REPO / "configs" / "default.json", self.root / "configs" / "default.json")

    def run_dir(self):
        return self.root / "experiments" / "runs" / self.run_id

    def test_real_release_is_consistent(self):
        self.assertEqual(verify(REPO), [])
        self.assertEqual(verify(self.root), [])

    def test_detects_modified_weights(self):
        with open(self.root / "models" / "synapse_v1.npz", "ab") as f:
            f.write(b"x")
        self.assertTrue(any("npz_sha256" in p for p in verify(self.root)))

    def test_detects_run_log_from_another_run(self):
        p = self.run_dir() / "log.json"
        log = json.loads(p.read_text()); log["run_id"] = "synapse_v1_seed42_20000101T000000Z"
        p.write_text(json.dumps(log))
        self.assertTrue(any("run log is for" in x for x in verify(self.root)))

    def test_detects_metric_disagreement_between_log_and_metadata(self):
        p = self.run_dir() / "log.json"
        log = json.loads(p.read_text()); log["evaluation"]["latency"]["median_ms"] += 0.001
        p.write_text(json.dumps(log))
        self.assertTrue(any("latency" in x for x in verify(self.root)))

    def test_detects_report_with_different_numbers(self):
        p = self.run_dir() / "report.md"
        p.write_text(p.read_text().replace("Test accuracy |", "Test accuracy | 1.0% |", 1))
        self.assertTrue(any("test accuracy" in x for x in verify(self.root)))

    def test_detects_missing_chart_and_missing_report(self):
        (self.run_dir() / "charts" / "confusion_matrix.png").unlink()
        self.assertTrue(any("confusion_matrix.png" in x for x in verify(self.root)))
        (self.run_dir() / "report.md").unlink()
        self.assertTrue(any("report.md is missing" in x for x in verify(self.root)))

    def test_detects_evaluation_for_a_different_model(self):
        p = self.root / "experiments" / "ood_eval.json"
        rep = json.loads(p.read_text()); rep["model_sha256"] = "0" * 64
        p.write_text(json.dumps(rep))
        self.assertTrue(any("OOD evaluation" in x and "different model" in x for x in verify(self.root)))
        (self.root / "experiments" / "pipeline_eval.json").unlink()
        self.assertTrue(any("pipeline evaluation" in x and "missing" in x for x in verify(self.root)))

    def test_detects_config_drift(self):
        p = self.root / "configs" / "default.json"
        cfg = json.loads(p.read_text()); cfg["training"]["epochs"] += 1
        p.write_text(json.dumps(cfg))
        self.assertTrue(any("differs from the config" in x for x in verify(self.root)))

    def test_leftover_runs_and_stray_files_fail_strict_only(self):
        (self.root / "experiments" / "runs" / "synapse_v1_seed1_20000101T000000Z").mkdir()
        (self.root / "models" / "registry.json").write_text("{}")
        strict = verify(self.root)
        self.assertTrue(any("leftover run" in x for x in strict))
        self.assertTrue(any("registry.json" in x for x in strict))
        self.assertEqual(verify(self.root, strict=False), [])

    def test_skip_test_models_are_not_releasable(self):
        p = self.root / "models" / "synapse_v1.json"
        meta = json.loads(p.read_text()); meta["test_evaluated"] = False
        p.write_text(json.dumps(meta))
        self.assertTrue(any("not evaluated on the test set" in x for x in verify(self.root)))

    def test_missing_model_is_reported(self):
        shutil.rmtree(self.root / "models")
        self.assertTrue(any("missing" in x for x in verify(self.root)))


@unittest.skipUnless(HAVE_RELEASE, "no release artifacts in this checkout")
class TestPackaging(unittest.TestCase):
    def test_package_builds_a_verified_zip_with_matching_manifest(self):
        with tempfile.TemporaryDirectory() as out:
            r = subprocess.run([sys.executable, "tools/package_release.py", "--out", out, "--no-data"], cwd=REPO,
                               capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            from app.version import __version__
            zip_path = next(Path(out).glob("*.zip"))
            self.assertEqual(zip_path.name, f"Shadowcr3-Synapse-{__version__}.zip")
            z = zipfile.ZipFile(zip_path)
            names = set(z.namelist())
            man = json.loads(z.read("Shadowcr3-Synapse/MANIFEST.json"))
            meta = json.loads(z.read("Shadowcr3-Synapse/models/synapse_v1.json"))
            self.assertEqual(man["run_id"], meta["run_id"])
            self.assertIn(f"Shadowcr3-Synapse/experiments/runs/{meta['run_id']}/report.md", names)
            self.assertIn("Shadowcr3-Synapse/RELEASE_NOTES.md", names)
            self.assertIn("Shadowcr3-Synapse/LICENSE", names)
            self.assertTrue(z.read("Shadowcr3-Synapse/LICENSE").startswith(b"                    GNU GENERAL PUBLIC LICENSE"))
            notes = z.read("Shadowcr3-Synapse/RELEASE_NOTES.md").decode()
            self.assertIn(meta["run_id"], notes); self.assertIn(meta["npz_sha256"], notes)
            self.assertIn(f"{meta['evaluation']['test_accuracy']:.2%}", notes)
            for forbidden in ("node_modules", "__pycache__", "registry.json", "user_samples", "candidates/"):
                self.assertFalse(any(forbidden in n for n in names), forbidden)
            self.assertEqual({n for n in names if "/runs/" in n and n.count("/") >= 4 and "report.md" in n}, {f"Shadowcr3-Synapse/experiments/runs/{meta['run_id']}/report.md"})

    def test_package_refuses_inconsistent_release(self):
        with tempfile.TemporaryDirectory() as t:
            r = subprocess.run([sys.executable, str(REPO / "tools" / "package_release.py"), "--root", t, "--out", t],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 1); self.assertIn("refusing", r.stderr)


if __name__ == "__main__":
    unittest.main()
