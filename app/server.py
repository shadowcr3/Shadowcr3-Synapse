
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, request
from werkzeug.exceptions import HTTPException

from .feedback import FeedbackStore
from .inference import ModelError
from .jobs import DataUnavailable, JobManager
from .models import ModelStore
from .ratelimit import RateLimiter, classify, retry_after_seconds
from .version import __version__
from .routes import bp, error_response

ROOT = Path(__file__).resolve().parent.parent
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}
MAX_BODY_BYTES = 16 * 1024
STATE_CHANGING = {"POST", "PUT", "PATCH", "DELETE"}
WRITE_PREFIXES = ("/jobs/", "/feedback", "/model/promote", "/model/rollback", "/model/versions/")

CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
       "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
log = logging.getLogger("synapse")


def _hostname(host_header: str) -> str:
    if host_header.startswith("["):
        return host_header.split("]")[0] + "]"
    return host_header.rsplit(":", 1)[0] if ":" in host_header else host_header


def make_data_provider(data_cfg: dict):
    from training.dataset import prepare_data
    lock, cache = threading.Lock(), {}

    def provider():
        with lock:
            if "ds" not in cache:
                try:
                    cache["ds"] = prepare_data({**data_cfg, "download": False}, split_seed=data_cfg["split_seed"])
                except (FileNotFoundError, ValueError, OSError) as exc:
                    raise DataUnavailable(
                        f"MNIST training data is not available ({exc}). Run `python tools/fetch_mnist.py` first.") from exc
            return cache["ds"]
    return provider


def create_app(model_dir=None, model_name: str = "synapse_v1", allowed_hosts=None, *, config_path=None,
               data_provider=None, feedback_path=None, read_only: bool = False,
               max_job_seconds: float = 900.0, experiments_dir=None, rate_limit=None, trust_proxy: int = 0) -> Flask:
    from training.config import load_config
    cfg = load_config(config_path or ROOT / "configs" / "default.json")
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES
    app.json.sort_keys = False

    store = ModelStore(model_dir or ROOT / "models", model_name)
    digits = store.predictor.digits
    feedback = FeedbackStore(feedback_path or ROOT / "data" / "user_samples.jsonl", digits=digits)
    jobs = JobManager(store, feedback, data_provider or make_data_provider(cfg["data"]), cfg,
                      max_seconds=max_job_seconds)
    frontend = ROOT / "frontend"
    app.extensions["synapse"] = {
        "store": store, "feedback": feedback, "jobs": jobs, "frontend_dir": frontend, "read_only": read_only,
        "experiments_dir": Path(experiments_dir) if experiments_dir else ROOT / "experiments",
        "static_files": {p.name for p in frontend.iterdir() if p.suffix in (".css", ".js")},
        "started": time.monotonic(),
    }
    allowed = {h.lower() for h in allowed_hosts} if allowed_hosts is not None else None
    limiter = None if rate_limit is False else (rate_limit if isinstance(rate_limit, RateLimiter) else RateLimiter(rate_limit))
    app.extensions["synapse"]["limiter"] = limiter
    if trust_proxy:
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=trust_proxy)

    @app.before_request
    def guards():
        if limiter is not None:
            ok, wait = limiter.check(request.remote_addr or "unknown", classify(request.method, request.path))
            if not ok:
                resp = error_response(429, "rate_limited", "too many requests; slow down", retry_after=round(wait, 2))
                resp.headers["Retry-After"] = str(retry_after_seconds(wait))
                return resp
        if allowed is not None and _hostname(request.host).lower() not in allowed:
            return error_response(400, "bad_host", "unexpected Host header")
        if request.method in STATE_CHANGING:
            origin = request.headers.get("Origin")
            if origin and urlsplit(origin).netloc.lower() != request.host.lower():
                return error_response(403, "cross_origin", "cross-origin requests are not allowed")
            if read_only and request.path.startswith(WRITE_PREFIXES):
                return error_response(403, "read_only", "this server is running in read-only mode")

    @app.after_request
    def security_headers(resp):
        resp.headers["Content-Security-Policy"] = CSP
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.errorhandler(HTTPException)
    def http_error(exc):
        codes = {400: "bad_request", 404: "not_found", 405: "method_not_allowed",
                 413: "payload_too_large", 415: "unsupported_media_type", 429: "rate_limited"}
        messages = {413: f"request body exceeds {MAX_BODY_BYTES} bytes"}
        status = exc.code or 500
        return error_response(status, codes.get(status, "http_error"), messages.get(status, exc.name.lower()))

    @app.errorhandler(Exception)
    def unexpected(exc):
        app.logger.exception("unhandled error")
        return error_response(500, "internal_error", "internal server error")

    app.register_blueprint(bp)
    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Synapse server")
    ap.add_argument("--model-dir", default=str(ROOT / "models"))
    ap.add_argument("--model-name", default="synapse_v1")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--read-only", action="store_true", help="disable training, feedback and model promotion")
    ap.add_argument("--max-job-seconds", type=float, default=900.0)
    ap.add_argument("--dev", action="store_true", help="use Flask's development server instead of waitress")
    ap.add_argument("--no-rate-limit", action="store_true", help="disable per-client rate limiting")
    ap.add_argument("--trust-proxy", type=int, default=0, metavar="N",
                    help="trust N reverse proxies' X-Forwarded-For (only behind a proxy you control)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    loopback = args.host in LOOPBACK_HOSTS
    if not loopback:
        log.warning("binding to %s exposes this unauthenticated server to the network; consider --read-only", args.host)
    try:
        app = create_app(args.model_dir, args.model_name, LOOPBACK_HOSTS if loopback else None,
                         read_only=args.read_only, max_job_seconds=args.max_job_seconds,
                         rate_limit=False if args.no_rate_limit else None, trust_proxy=args.trust_proxy)
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    info = app.extensions["synapse"]["store"].predictor.info()
    print(f"Synapse {__version__}, model {info['version']} ({info['version_id']}) at http://{args.host}:{args.port}"
          f"{'  [read-only]' if args.read_only else ''}  [Ctrl+C to stop]")
    try:
        if args.dev:
            app.run(host=args.host, port=args.port, debug=False, threaded=True)
        else:
            try:
                from waitress import serve
            except ImportError:
                log.warning("waitress is not installed; falling back to the Flask development server")
                app.run(host=args.host, port=args.port, debug=False, threaded=True)
            else:
                serve(app, host=args.host, port=args.port, threads=8, ident="synapse")
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
