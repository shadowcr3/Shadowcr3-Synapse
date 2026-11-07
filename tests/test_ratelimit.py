import tempfile
import unittest
from pathlib import Path

from app import create_app
from app.ratelimit import DEFAULT_RULES, RateLimiter, Rule, classify
from app.server import LOOPBACK_HOSTS
from tests.helpers import class_pixels, make_release, synthetic_dataset


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class TestBuckets(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.rl = RateLimiter({"a": Rule(rate=2.0, burst=3.0), "b": Rule(rate=1.0, burst=1.0)}, max_clients=3, clock=self.clock)

    def test_burst_then_blocked_with_a_sensible_wait(self):
        self.assertEqual([self.rl.check("x", "a")[0] for _ in range(3)], [True, True, True])
        ok, wait = self.rl.check("x", "a")
        self.assertFalse(ok); self.assertAlmostEqual(wait, 0.5, places=6)

    def test_refills_over_time_up_to_the_burst(self):
        for _ in range(3):
            self.rl.check("x", "a")
        self.clock.t += 0.5
        self.assertTrue(self.rl.check("x", "a")[0]); self.assertFalse(self.rl.check("x", "a")[0])
        self.clock.t += 100
        self.assertEqual(sum(self.rl.check("x", "a")[0] for _ in range(10)), 3)

    def test_clients_and_buckets_are_independent(self):
        for _ in range(3):
            self.rl.check("x", "a")
        self.assertFalse(self.rl.check("x", "a")[0])
        self.assertTrue(self.rl.check("y", "a")[0])
        self.assertTrue(self.rl.check("x", "b")[0])

    def test_memory_is_bounded_and_evicts_least_recent(self):
        for c in "wxyz":
            self.rl.check(c, "a")
        self.assertEqual(self.rl.tracked_clients(), 3)
        for _ in range(3):
            self.rl.check("z", "a")
        self.assertFalse(self.rl.check("z", "a")[0])

    def test_unknown_bucket_is_a_programming_error(self):
        with self.assertRaises(KeyError):
            self.rl.check("x", "nope")


class TestClassification(unittest.TestCase):
    def test_routes_map_to_the_intended_bucket(self):
        cases = [("POST", "/predict", "infer"), ("POST", "/trace", "infer"), ("GET", "/jobs/status", "read"),
                 ("POST", "/jobs/train", "job"), ("POST", "/jobs/finetune", "job"), ("POST", "/jobs/cancel", "write"),
                 ("POST", "/feedback", "write"), ("DELETE", "/feedback", "write"), ("POST", "/model/promote", "write"),
                 ("GET", "/model/info", "read"), ("GET", "/health", "read"), ("GET", "/", "static"),
                 ("GET", "/static/app.js", "static"), ("GET", "/feedback/stats", "read")]
        for method, path, bucket in cases:
            self.assertEqual(classify(method, path), bucket, (method, path))
        self.assertTrue(set(DEFAULT_RULES) >= {classify(m, p) for m, p, _ in cases})

    def test_the_drawing_ui_stays_inside_the_default_limits(self):
        rl, clock = RateLimiter(clock=Clock()), None
        clock = rl._clock
        denied = 0
        for step in range(600):
            clock.t += 0.11
            denied += not rl.check("ui", "infer")[0]
            if step % 3 == 0:
                denied += not rl.check("ui", "read")[0]
        self.assertEqual(denied, 0)


class TestApiLimiting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.ds = synthetic_dataset()
        make_release(cls.dir, cls.ds)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def client(self, rules, **kw):
        limiter = RateLimiter(rules, clock=Clock())
        app = create_app(self.dir, allowed_hosts=LOOPBACK_HOSTS, data_provider=lambda: self.ds,
                         feedback_path=self.dir / "fb.jsonl", rate_limit=limiter, **kw)
        return app.test_client(), limiter

    def test_returns_429_with_retry_after_and_a_json_error(self):
        c, _ = self.client({**DEFAULT_RULES, "infer": Rule(1.0, 2.0)})
        body = {"pixels": class_pixels(1)}
        self.assertEqual([c.post("/predict", json=body).status_code for _ in range(2)], [200, 200])
        r = c.post("/predict", json=body)
        self.assertEqual(r.status_code, 429)
        self.assertEqual(r.json["error"]["code"], "rate_limited")
        self.assertGreaterEqual(int(r.headers["Retry-After"]), 1)
        self.assertIn("default-src 'none'", r.headers["Content-Security-Policy"])

    def test_one_flooded_endpoint_class_does_not_block_others(self):
        c, _ = self.client({**DEFAULT_RULES, "infer": Rule(0.001, 1.0)})
        c.post("/predict", json={"pixels": class_pixels(1)})
        self.assertEqual(c.post("/predict", json={"pixels": class_pixels(1)}).status_code, 429)
        self.assertEqual(c.get("/health").status_code, 200)
        self.assertEqual(c.get("/model/info").status_code, 200)

    def test_job_starts_are_tightly_limited(self):
        c, _ = self.client({**DEFAULT_RULES, "job": Rule(0.001, 1.0)})
        first = c.post("/jobs/train", json={"hidden_sizes": [8], "epochs": 1})
        self.assertEqual(first.status_code, 202)
        self.assertEqual(c.post("/jobs/train", json={}).status_code, 429)
        c.post("/jobs/cancel", json={})
        self.app_jobs_wait(c)

    def app_jobs_wait(self, c):
        import time
        for _ in range(300):
            if not c.get("/jobs/status").json["busy"]:
                return
            time.sleep(0.05)

    def test_forwarded_for_cannot_be_used_to_dodge_the_limit(self):
        c, _ = self.client({**DEFAULT_RULES, "infer": Rule(0.001, 1.0)})
        body = {"pixels": class_pixels(1)}
        self.assertEqual(c.post("/predict", json=body, headers={"X-Forwarded-For": "1.1.1.1"}).status_code, 200)
        self.assertEqual(c.post("/predict", json=body, headers={"X-Forwarded-For": "2.2.2.2"}).status_code, 429)

    def test_trusted_proxy_mode_keys_on_the_forwarded_client(self):
        c, _ = self.client({**DEFAULT_RULES, "infer": Rule(0.001, 1.0)}, trust_proxy=1)
        body = {"pixels": class_pixels(1)}
        self.assertEqual(c.post("/predict", json=body, headers={"X-Forwarded-For": "1.1.1.1"}).status_code, 200)
        self.assertEqual(c.post("/predict", json=body, headers={"X-Forwarded-For": "2.2.2.2"}).status_code, 200)
        self.assertEqual(c.post("/predict", json=body, headers={"X-Forwarded-For": "1.1.1.1"}).status_code, 429)

    def test_can_be_disabled(self):
        app = create_app(self.dir, allowed_hosts=LOOPBACK_HOSTS, data_provider=lambda: self.ds,
                         feedback_path=self.dir / "fb.jsonl", rate_limit=False)
        c = app.test_client()
        self.assertTrue(all(c.get("/health").status_code == 200 for _ in range(200)))

    def test_enabled_by_default(self):
        app = create_app(self.dir, allowed_hosts=LOOPBACK_HOSTS, data_provider=lambda: self.ds,
                         feedback_path=self.dir / "fb.jsonl")
        self.assertIsNotNone(app.extensions["synapse"]["limiter"])
        c = app.test_client()
        codes = [c.get("/model/info").status_code for _ in range(60)]
        self.assertIn(429, codes)


if __name__ == "__main__":
    unittest.main()
