import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from waitress import create_server

from app import create_app
from app.ratelimit import DEFAULT_RULES, RateLimiter, Rule
from app.server import LOOPBACK_HOSTS
from tests.helpers import make_release, synthetic_dataset

tmp = Path(tempfile.mkdtemp(prefix="synapse-ui-"))
ds = synthetic_dataset()
make_release(tmp, ds, hidden=(24,), epochs=6)
limiter = RateLimiter({**DEFAULT_RULES, "write": Rule(rate=50.0, burst=200.0), "job": Rule(rate=5.0, burst=20.0)})
app = create_app(tmp, allowed_hosts=LOOPBACK_HOSTS, data_provider=lambda: ds, feedback_path=tmp / "fb.jsonl", rate_limit=limiter)
server = create_server(app, host="127.0.0.1", port=0, threads=8)
print(json.dumps({"port": int(server.effective_port)}), flush=True)
server.run()
