"""Boot a Synapse server on a free port with a tiny synthetic model, for the UI test.
Prints {"port": N} on stdout, then serves until killed."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from waitress import create_server  # noqa: E402

from app import create_app  # noqa: E402
from app.server import LOOPBACK_HOSTS  # noqa: E402
from tests.helpers import make_release, synthetic_dataset  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="synapse-ui-"))
ds = synthetic_dataset()
make_release(tmp, ds, hidden=(24,), epochs=6)
app = create_app(tmp, allowed_hosts=LOOPBACK_HOSTS, data_provider=lambda: ds, feedback_path=tmp / "fb.jsonl")
server = create_server(app, host="127.0.0.1", port=0, threads=8)
print(json.dumps({"port": int(server.effective_port)}), flush=True)
server.run()
