import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    rate: float
    burst: float


DEFAULT_RULES = {
    "infer": Rule(rate=15.0, burst=40.0),
    "read": Rule(rate=10.0, burst=40.0),
    "write": Rule(rate=2.0, burst=10.0),
    "job": Rule(rate=0.1, burst=3.0),
    "static": Rule(rate=30.0, burst=120.0),
}


def classify(method: str, path: str) -> str:
    if path in ("/predict", "/trace"):
        return "infer"
    if path.startswith("/jobs/"):
        if path == "/jobs/status":
            return "read"
        if path == "/jobs/cancel":
            return "write"
        return "job" if method == "POST" else "read"
    if method in ("POST", "PUT", "PATCH", "DELETE"):
        return "write"
    if path == "/" or path.startswith("/static/"):
        return "static"
    return "read"


class RateLimiter:
    def __init__(self, rules=None, max_clients: int = 10_000, clock=time.monotonic) -> None:
        self.rules = dict(DEFAULT_RULES if rules is None else rules)
        self.max_clients = max_clients
        self._clock = clock
        self._lock = threading.Lock()
        self._clients: OrderedDict = OrderedDict()

    def check(self, client: str, bucket: str):
        rule = self.rules[bucket]
        now = self._clock()
        with self._lock:
            state = self._clients.get(client)
            if state is None:
                state = {}
                self._clients[client] = state
                while len(self._clients) > self.max_clients:
                    self._clients.popitem(last=False)
            else:
                self._clients.move_to_end(client)
            tokens, last = state.get(bucket, (rule.burst, now))
            tokens = min(rule.burst, tokens + (now - last) * rule.rate)
            if tokens >= 1.0:
                state[bucket] = (tokens - 1.0, now)
                return True, 0.0
            state[bucket] = (tokens, now)
            return False, (1.0 - tokens) / rule.rate

    def tracked_clients(self) -> int:
        with self._lock:
            return len(self._clients)


def retry_after_seconds(wait: float) -> int:
    return max(1, int(math.ceil(wait)))
