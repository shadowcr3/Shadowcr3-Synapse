import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from app import create_app
from app.jobs import DataUnavailable
from app.server import LOOPBACK_HOSTS
from tests.helpers import class_pixels, make_release, synthetic_dataset

REPO = Path(__file__).resolve().parent.parent


class ApiCase(unittest.TestCase):
    read_only = False

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.ds = synthetic_dataset()
        make_release(cls.dir, cls.ds, epochs=8)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.fb = self.dir / f"fb-{id(self)}.jsonl"
        self.models = self.dir / f"m-{id(self)}"
        import shutil
        shutil.copytree(self.dir, self.models, ignore=shutil.ignore_patterns("m-*", "fb-*"))
        self.app = create_app(self.models, allowed_hosts=LOOPBACK_HOSTS, data_provider=lambda: self.ds,
                              feedback_path=self.fb, read_only=self.read_only)
        self.c = self.app.test_client()
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        jobs = self.app.extensions["synapse"]["jobs"]
        jobs.cancel(); jobs.wait(30)

    def px(self, c=2):
        return {"pixels": class_pixels(c)}


class TestService(ApiCase):
    def test_health_info_metrics_versions(self):
        h = self.c.get("/health").json
        self.assertEqual((h["status"], h["model_loaded"], h["read_only"]), ("ok", True, False))
        info = self.c.get("/model/info").json
        self.assertEqual(info["digits"], list(range(1, 10))); self.assertNotIn("training_config", info)
        m = self.c.get("/model/metrics").json
        self.assertIn("history", m); self.assertIn("validation", m)
        v = self.c.get("/model/versions").json
        self.assertEqual(v["active"], "release"); self.assertEqual(len(v["versions"]), 1)

    def test_weights_endpoint(self):
        w = self.c.get("/model/weights/0").json
        self.assertEqual(w["shape"][1], 784); self.assertEqual(len(w["scales"]), w["shape"][0])
        self.assertEqual(self.c.get("/model/weights/9").status_code, 404)
        self.assertEqual(self.c.get("/model/weights/-1").status_code, 404)

    def test_static_files_whitelisted_and_traversal_blocked(self):
        self.assertEqual(self.c.get("/static/preprocess.js").status_code, 200)
        for path in ("/static/..%2Fapp%2Fserver.py", "/static/../app/server.py", "/static/server.py",
                     "/static/index.html", "/static/%2e%2e/models/synapse_v1.npz", "/static/x.py"):
            self.assertEqual(self.c.get(path).status_code, 404, path)

    def test_security_headers_on_every_response(self):
        for r in (self.c.get("/health"), self.c.get("/nope"), self.c.post("/predict", data="x")):
            h = r.headers
            self.assertIn("default-src 'none'", h["Content-Security-Policy"])
            self.assertEqual(h["X-Content-Type-Options"], "nosniff")
            self.assertEqual(h["X-Frame-Options"], "DENY")
            self.assertEqual(h["Cache-Control"], "no-store")
            self.assertNotIn("Access-Control-Allow-Origin", h)

    def test_unknown_routes_and_methods_return_json_errors(self):
        r = self.c.get("/nope"); self.assertEqual((r.status_code, r.json["error"]["code"]), (404, "not_found"))
        r = self.c.get("/predict"); self.assertEqual((r.status_code, r.json["error"]["code"]), (405, "method_not_allowed"))

    def test_host_header_allowlist(self):
        self.assertEqual(self.c.get("/health", headers={"Host": "evil.example"}).status_code, 400)
        self.assertEqual(self.c.get("/health", headers={"Host": "127.0.0.1:5000"}).status_code, 200)
        self.assertEqual(self.c.get("/health", headers={"Host": "[::1]:5000"}).status_code, 200)

    def test_cross_origin_state_changes_blocked(self):
        r = self.c.post("/feedback", json={**self.px(), "label": 3}, headers={"Origin": "http://evil.example"})
        self.assertEqual((r.status_code, r.json["error"]["code"]), (403, "cross_origin"))
        r = self.c.post("/feedback", json={**self.px(), "label": 3}, headers={"Origin": "http://localhost"})
        self.assertEqual(r.status_code, 201)

    def test_internal_errors_do_not_leak(self):
        with mock.patch("app.inference.Predictor.predict", side_effect=RuntimeError("secret /etc/passwd")):
            r = self.c.post("/predict", json=self.px())
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.json["error"]["code"], "internal_error")
        self.assertNotIn("secret", r.get_data(as_text=True)); self.assertNotIn("Traceback", r.get_data(as_text=True))

    def test_works_fully_offline(self):
        def no_network(*a, **k):
            raise OSError("network disabled")
        with mock.patch.object(socket.socket, "connect", no_network), \
             mock.patch("socket.create_connection", no_network), mock.patch("socket.getaddrinfo", no_network):
            self.assertEqual(self.c.post("/predict", json=self.px()).status_code, 200)
            self.assertEqual(self.c.post("/trace", json=self.px()).status_code, 200)


class TestPredictApi(ApiCase):
    def test_predict(self):
        r = self.c.post("/predict", json=self.px(4))
        self.assertEqual(r.status_code, 200)
        b = r.json
        self.assertEqual((b["status"], b["digit"], b["model_id"]), ("ok", 5, "release"))
        self.assertAlmostEqual(sum(b["scores"]), 1.0, places=5)

    def test_invalid_inputs(self):
        cases = [({"pixels": [0] * 783}, 400, "invalid_pixels"), ({"pixels": "x"}, 400, "invalid_pixels"),
                 ({"pixels": [999] * 784}, 400, "invalid_pixels"), ({"pixels": [1.5] * 784}, 400, "invalid_pixels"),
                 ({"pixels": [0] * 784}, 422, "blank_input"), ({"pixels": [200] * 784, "x": 1}, 400, "invalid_pixels"),
                 ([1, 2], 400, "invalid_pixels"), ({}, 400, "invalid_pixels")]
        for body, status, code in cases:
            r = self.c.post("/predict", json=body)
            self.assertEqual((r.status_code, r.json["error"]["code"]), (status, code), str(body)[:50])

    def test_content_type_json_and_size_limits(self):
        r = self.c.post("/predict", data="x", content_type="text/plain")
        self.assertEqual((r.status_code, r.json["error"]["code"]), (415, "unsupported_media_type"))
        r = self.c.post("/predict", data="{not json", content_type="application/json")
        self.assertEqual((r.status_code, r.json["error"]["code"]), (400, "invalid_json"))
        r = self.c.post("/predict", data="[" + "0," * 20000 + "0]", content_type="application/json")
        self.assertEqual((r.status_code, r.json["error"]["code"]), (413, "payload_too_large"))

    def test_trace_with_and_without_label(self):
        r = self.c.post("/trace", json=self.px(3)).json
        self.assertNotIn("backward", r["trace"]); self.assertEqual(r["prediction"]["top_guess"], 4)
        r = self.c.post("/trace", json={**self.px(3), "label": 4, "edges": 12}).json
        self.assertIn("deltas", r["trace"]["backward"]); self.assertLessEqual(len(r["trace"]["edges"]["input_hidden1"]), 12)

    def test_trace_validation(self):
        for extra in ({"label": 0}, {"label": 10}, {"label": "3"}, {"label": True}, {"edges": -1},
                      {"edges": 201}, {"edges": "9"}, {"junk": 1}):
            self.assertEqual(self.c.post("/trace", json={**self.px(), **extra}).status_code, 400, str(extra))
        self.assertEqual(self.c.post("/trace", json={"pixels": [0] * 784}).status_code, 422)


class TestFeedbackApi(ApiCase):
    def test_add_dedupe_stats_clear(self):
        r = self.c.post("/feedback", json={**self.px(1), "label": 2})
        self.assertEqual((r.status_code, r.json["count"]), (201, 1))
        r = self.c.post("/feedback", json={**self.px(1), "label": 2})
        self.assertEqual((r.status_code, r.json["duplicate"]), (200, True))
        self.assertEqual(self.c.get("/feedback/stats").json["per_digit"]["2"], 1)
        self.assertEqual(self.c.delete("/feedback").status_code, 400)
        self.assertEqual(self.c.delete("/feedback?confirm=yes").json["deleted"], 1)
        self.assertEqual(self.c.get("/feedback/stats").json["total"], 0)

    def test_validation(self):
        for body in ({**self.px(), "label": 0}, {**self.px(), "label": 10}, {**self.px(), "label": "3"},
                     {**self.px(), "label": True}, {"pixels": self.px()["pixels"]}, {**self.px(), "label": 3, "x": 1}):
            self.assertEqual(self.c.post("/feedback", json=body).status_code, 400, str(body)[:60])
        self.assertEqual(self.c.post("/feedback", json={"pixels": [0] * 784, "label": 3}).status_code, 422)


class TestJobsApi(ApiCase):
    def _wait(self, timeout=60):
        end = time.time() + timeout
        while time.time() < end:
            st = self.c.get("/jobs/status").json
            if not st["busy"]:
                return st
            time.sleep(0.05)
        self.fail("job did not finish")

    def test_train_promote_rollback_flow(self):
        r = self.c.post("/jobs/train", json={"hidden_sizes": [16], "epochs": 3, "batch_size": 32})
        self.assertEqual(r.status_code, 202)
        self.assertEqual(self.c.post("/jobs/train", json={}).status_code, 409)          # busy
        st = self._wait()
        self.assertEqual(st["job"]["state"], "done")
        cid = st["job"]["result"]["candidate_id"]
        versions = self.c.get("/model/versions").json["versions"]
        self.assertEqual({v["id"] for v in versions}, {"release", cid})
        d = self.c.post("/model/promote", json={"id": cid})
        self.assertIn(d.status_code, (200, 409))
        if d.status_code == 200:
            self.assertEqual(self.c.get("/health").json["active_model"], cid)
            self.assertEqual(self.c.post("/predict", json=self.px()).json["model_id"], cid)
            self.assertEqual(self.c.post("/model/rollback").json["active"], "release")
        self.assertEqual(self.c.post("/model/rollback").status_code, 409 if d.status_code == 409 else 409)

    def test_invalid_job_params_and_kinds(self):
        self.assertEqual(self.c.post("/jobs/train", json={"epochs": 0}).json["error"]["code"], "invalid_job")
        self.assertEqual(self.c.post("/jobs/nope", json={}).status_code, 400)
        self.assertEqual(self.c.post("/jobs/train", data="x", content_type="text/plain").status_code, 415)
        self.assertEqual(self.c.get("/jobs/status?since=abc").status_code, 400)
        self.assertEqual(self.c.post("/jobs/finetune", json={}).json["error"]["code"], "invalid_job")

    def test_cancel(self):
        self.c.post("/jobs/train", json={"hidden_sizes": [64], "epochs": 60, "batch_size": 16})
        self.assertTrue(self.c.post("/jobs/cancel").json["cancelling"])
        self.assertEqual(self._wait()["job"]["state"], "cancelled")

    def test_promote_validation(self):
        for body, code in (({"id": "../x"}, 400), ({"id": "ghost"}, 400), ({}, 400), ({"id": "release"}, 409)):
            self.assertEqual(self.c.post("/model/promote", json=body).status_code, code, str(body))

    def test_data_unavailable_is_503(self):
        def boom():
            raise DataUnavailable("no MNIST")
        app = create_app(self.models, allowed_hosts=LOOPBACK_HOSTS, data_provider=boom, feedback_path=self.fb)
        r = app.test_client().post("/jobs/train", json={})
        self.assertEqual((r.status_code, r.json["error"]["code"]), (503, "data_unavailable"))


class TestReadOnly(ApiCase):
    read_only = True

    def test_writes_blocked_reads_allowed(self):
        for method, path in (("post", "/jobs/train"), ("post", "/jobs/cancel"), ("post", "/feedback"),
                             ("delete", "/feedback?confirm=yes"), ("post", "/model/promote"), ("post", "/model/rollback")):
            r = getattr(self.c, method)(path, json={})
            self.assertEqual((r.status_code, r.json["error"]["code"]), (403, "read_only"), path)
        self.assertEqual(self.c.post("/predict", json=self.px()).status_code, 200)
        self.assertEqual(self.c.post("/trace", json=self.px()).status_code, 200)
        self.assertTrue(self.c.get("/health").json["read_only"])


if __name__ == "__main__":
    unittest.main()


class TestOodEndpoint(ApiCase):
    def _app_with_report(self, report):
        import json
        exp = self.dir / f"exp-{id(self)}"
        exp.mkdir(exist_ok=True)
        if report is not None:
            (exp / "ood_eval.json").write_text(json.dumps(report))
        return create_app(self.models, allowed_hosts=LOOPBACK_HOSTS, data_provider=lambda: self.ds,
                          feedback_path=self.fb, experiments_dir=exp).test_client()

    def test_served_only_when_bound_to_the_active_models_hash(self):
        sha = self.app.extensions["synapse"]["store"].predictor.metadata["npz_sha256"]
        ok = self._app_with_report({"model_sha256": sha, "categories": {}, "per_character": {}})
        self.assertEqual(ok.get("/model/ood").status_code, 200)
        self.assertEqual(self._app_with_report({"model_sha256": "0" * 64}).get("/model/ood").status_code, 404)
        self.assertEqual(self._app_with_report(None).get("/model/ood").status_code, 404)
