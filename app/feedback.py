"""Local store of user-labelled drawings used to fine-tune the model.

Append-only JSON Lines (one sample per line), process-local lock, fsync on write, 0600 file
mode. Samples are the exact 28x28 pixels the model saw (already validated like /predict input)
plus a digit label 1-9. Duplicates are ignored. Nothing here ever leaves the machine.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .preprocessing import N_PIXELS

log = logging.getLogger("synapse.feedback")
MAX_SAMPLES = 20_000


class FeedbackError(ValueError):
    pass


class FeedbackFull(FeedbackError):
    pass


class FeedbackStore:
    def __init__(self, path, digits=tuple(range(1, 10)), max_samples: int = MAX_SAMPLES) -> None:
        self.path = Path(path)
        self.digits = tuple(digits)
        self.max_samples = max_samples
        self._lock = threading.Lock()

    @staticmethod
    def _sample_id(pixels: np.ndarray, label: int) -> str:
        return hashlib.sha256(pixels.tobytes() + bytes([label])).hexdigest()[:16]

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows, seen = [], set()
        with open(self.path, "r", encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                try:
                    row = json.loads(line)
                    pixels = np.frombuffer(base64.b64decode(row["pixels"], validate=True), dtype=np.uint8)
                    if pixels.size != N_PIXELS or row["label"] not in self.digits or row["id"] in seen:
                        raise ValueError("bad row")
                    seen.add(row["id"])
                    rows.append({"id": row["id"], "ts": row["ts"], "label": row["label"], "pixels": pixels})
                except (ValueError, KeyError, TypeError):
                    log.warning("skipping malformed feedback line %d", n)
        return rows

    def add(self, pixels: np.ndarray, label: int) -> dict:
        if label not in self.digits:
            raise FeedbackError(f"label must be one of {list(self.digits)}")
        if pixels.dtype != np.uint8 or pixels.shape != (N_PIXELS,):
            raise FeedbackError("pixels must be a uint8 array of 784 values")
        sid = self._sample_id(pixels, label)
        with self._lock:
            rows = self._read()
            if any(r["id"] == sid for r in rows):
                return {"id": sid, "duplicate": True, "count": len(rows)}
            if len(rows) >= self.max_samples:
                raise FeedbackFull(f"sample limit of {self.max_samples} reached; clear some samples first")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps({"id": sid, "ts": datetime.now(timezone.utc).isoformat(), "label": label,
                               "pixels": base64.b64encode(pixels.tobytes()).decode("ascii")})
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            return {"id": sid, "duplicate": False, "count": len(rows) + 1}

    def stats(self) -> dict:
        with self._lock:
            rows = self._read()
        per = {str(d): 0 for d in self.digits}
        for r in rows:
            per[str(r["label"])] += 1
        return {"total": len(rows), "per_digit": per, "max_samples": self.max_samples}

    def arrays(self, classes):
        """(X uint8 (N, 784), y class indices (N,), ids). `classes` maps index -> digit."""
        index = {int(c): i for i, c in enumerate(classes)}
        with self._lock:
            rows = [r for r in self._read() if r["label"] in index]
        if not rows:
            return np.empty((0, N_PIXELS), np.uint8), np.empty(0, np.int64), []
        return (np.stack([r["pixels"] for r in rows]), np.array([index[r["label"]] for r in rows], dtype=np.int64),
                [r["id"] for r in rows])

    def delete(self, sample_id: str) -> bool:
        with self._lock:
            rows = self._read()
            kept = [r for r in rows if r["id"] != sample_id]
            if len(kept) == len(rows):
                return False
            self._rewrite(kept)
            return True

    def clear(self) -> int:
        with self._lock:
            n = len(self._read())
            self._rewrite([])
            return n

    def _rewrite(self, rows: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "ts": r["ts"], "label": r["label"],
                                    "pixels": base64.b64encode(r["pixels"].tobytes()).decode("ascii")}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
