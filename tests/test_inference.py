import base64
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.inference import ModelError, Predictor
from app.preprocessing import parse_pixels
from tests.helpers import class_pixels, make_release, synthetic_dataset


class TestPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.ds = synthetic_dataset()
        cls.net, cls.meta = make_release(cls.dir, cls.ds, epochs=8)
        cls.p = Predictor.from_dir(cls.dir, "synapse_v1")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_prediction_contract(self):
        out = self.p.predict(parse_pixels({"pixels": class_pixels(4)}))
        self.assertEqual(out["digits"], list(range(1, 10)))
        self.assertAlmostEqual(sum(out["scores"]), 1.0, places=5)
        self.assertEqual(out["top_guess"], 5)
        self.assertEqual(out["status"], "ok"); self.assertEqual(out["digit"], 5)
        self.assertGreater(out["confidence"], out["threshold"])
        self.assertGreaterEqual(out["processing_ms"], 0)

    def test_matches_training_time_forward_pass(self):
        px = parse_pixels({"pixels": class_pixels(2)})
        direct = self.net.predict_proba(px.astype(np.float32) / 255)[0]
        np.testing.assert_allclose(self.p.predict(px)["scores"], direct, atol=1e-5)

    def test_low_confidence_is_rejected(self):
        p = Predictor(self.p.network, {**self.p.metadata, "reject": {"threshold": 1.0}})
        out = p.predict(parse_pixels({"pixels": class_pixels(1)}))
        self.assertEqual(out["status"], "low_confidence")
        self.assertIsNone(out["digit"]); self.assertIn(out["top_guess"], range(1, 10))

    def test_no_threshold_always_accepts(self):
        p = Predictor(self.p.network, {**self.p.metadata, "reject": None})
        self.assertEqual(p.predict(parse_pixels({"pixels": class_pixels(1)}))["status"], "ok")

    def test_trace_is_consistent_with_predict(self):
        px = parse_pixels({"pixels": class_pixels(6)})
        tr = self.p.trace(px, label_digit=7, k_edges=10)
        np.testing.assert_allclose(tr["layers"][-1]["probs"], self.p.predict(px)["scores"], atol=1e-3)
        self.assertEqual(tr["backward"]["target"], 6)
        with self.assertRaises(ValueError):
            self.p.trace(px, label_digit=0)

    def test_weights_decode(self):
        w = self.p.weights(0)
        q = np.frombuffer(base64.b64decode(w["data"]), dtype=np.int8).reshape(w["shape"])
        approx = q * np.array(w["scales"])[:, None]
        self.assertLess(np.abs(approx - self.p.network.layers[0].W).max(), max(w["scales"]))
        self.assertEqual(len(w["bias"]), w["shape"][0])
        with self.assertRaises(ValueError):
            self.p.weights(5)

    def test_info_is_whitelisted_and_has_no_paths(self):
        info = self.p.info()
        self.assertNotIn("training_config", info)
        self.assertNotIn(str(self.dir), repr(info))
        self.assertEqual(info["version_id"], "synapse_v1")

    def test_load_failures(self):
        with self.assertRaises(ModelError):
            Predictor.from_dir(self.dir, "nope")
        with self.assertRaises(ModelError):
            Predictor(self.p.network, {**self.p.metadata, "class_mapping": {"0": 1}})
        with self.assertRaises(ModelError):
            Predictor(self.p.network, {**self.p.metadata, "reject": {"threshold": 7}})


class TestIndependence(unittest.TestCase):
    def test_inference_modules_do_not_import_training(self):
        import ast
        for name in ("inference.py", "preprocessing.py"):
            tree = ast.parse((Path(__file__).resolve().parent.parent / "app" / name).read_text())
            mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
            mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            self.assertFalse(any(m.split(".")[0] == "training" for m in mods), name)


if __name__ == "__main__":
    unittest.main()
