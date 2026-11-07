
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path

from .inference import ModelError, Predictor

log = logging.getLogger("synapse.models")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
RELEASE_ID = "release"
MAX_CANDIDATES = 30


class StoreError(ValueError):
    pass


class ModelStore:
    def __init__(self, model_dir, release_name: str = "synapse_v1") -> None:
        self.model_dir = Path(model_dir)
        self.release_name = release_name
        self.candidates_dir = self.model_dir / "candidates"
        self.registry_path = self.model_dir / "registry.json"
        self._lock = threading.RLock()
        self._cache: dict[str, Predictor] = {RELEASE_ID: Predictor.from_dir(self.model_dir, release_name, RELEASE_ID)}
        reg = self._read_registry()
        self._history: list[str] = [h for h in reg.get("history", []) if h == RELEASE_ID or self._exists(h)]
        active = reg.get("active", RELEASE_ID)
        if active != RELEASE_ID:
            try:
                self._cache[active] = self._load_candidate(active)
            except (StoreError, ModelError) as exc:
                log.warning("registry points at unusable model %r (%s); using the release model", active, exc)
                active, self._history = RELEASE_ID, []
        self._active_id = active

    @property
    def predictor(self) -> Predictor:
        return self._cache[self._active_id]

    @property
    def active_id(self) -> str:
        return self._active_id

    def _read_registry(self) -> dict:
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                reg = json.load(f)
            return reg if isinstance(reg, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_registry(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.model_dir, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"active": self._active_id, "history": self._history}, f, indent=2)
        os.chmod(tmp, 0o644)
        os.replace(tmp, self.registry_path)

    @staticmethod
    def _check_id(cid: str) -> str:
        if not isinstance(cid, str) or not ID_RE.match(cid):
            raise StoreError("invalid model id")
        return cid

    def _exists(self, cid: str) -> bool:
        return cid == RELEASE_ID or (ID_RE.match(cid) is not None and (self.candidates_dir / f"{cid}.json").exists())

    def _load_candidate(self, cid: str) -> Predictor:
        self._check_id(cid)
        if not self._exists(cid):
            raise StoreError("no such model")
        return Predictor.from_dir(self.candidates_dir, cid, cid)

    def save_candidate(self, network, metadata: dict, cid: str) -> None:
        self._check_id(cid)
        with self._lock:
            network.save(self.candidates_dir, cid, metadata)
            self._prune()

    def get(self, cid: str) -> Predictor:
        with self._lock:
            if cid not in self._cache:
                self._cache[cid] = self._load_candidate(cid)
            return self._cache[cid]

    def _prune(self) -> None:
        files = sorted(self.candidates_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        protected = {self._active_id, *self._history}
        excess = len(files) - MAX_CANDIDATES
        for f in files:
            if excess <= 0:
                break
            if f.stem in protected:
                continue
            for ext in (".json", ".npz"):
                (self.candidates_dir / f"{f.stem}{ext}").unlink(missing_ok=True)
            self._cache.pop(f.stem, None)
            excess -= 1

    def delete(self, cid: str) -> None:
        with self._lock:
            self._check_id(cid)
            if cid == self._active_id or cid in self._history:
                raise StoreError("cannot delete the active model or one kept for rollback")
            if not self._exists(cid):
                raise StoreError("no such model")
            for ext in (".json", ".npz"):
                (self.candidates_dir / f"{cid}{ext}").unlink(missing_ok=True)
            self._cache.pop(cid, None)

    def list_versions(self) -> list[dict]:
        with self._lock:
            ids = [RELEASE_ID] + [p.stem for p in self.candidates_dir.glob("*.json") if ID_RE.match(p.stem)]
            out = []
            for cid in ids:
                try:
                    info = self.get(cid).info()
                except (StoreError, ModelError) as exc:
                    log.warning("skipping unreadable model %s: %s", cid, exc)
                    continue
                out.append({"id": cid, "active": cid == self._active_id, "kind": "release" if cid == RELEASE_ID else "candidate",
                            "created_utc": info["created_utc"], "source": info["source"], "parent": info["parent"],
                            "layer_sizes": info["architecture"]["layer_sizes"], "parameter_count": info["parameter_count"],
                            "validation_accuracy": info["evaluation"].get("validation_accuracy"),
                            "test_accuracy": info["evaluation"].get("test_accuracy"),
                            "test_evaluated": info["test_evaluated"]})
            out.sort(key=lambda v: (v["kind"] != "release", v["created_utc"] or ""), reverse=False)
            return out

    def evaluate_gate(self, cid: str, tolerance: float = 0.003) -> dict:
        cand, active = self.get(cid), self.predictor
        c_ev, a_ev = cand.metadata.get("evaluation", {}), active.metadata.get("evaluation", {})
        c_acc, a_acc = c_ev.get("validation_accuracy"), a_ev.get("validation_accuracy")
        same_val = (cand.metadata.get("dataset", {}).get("split_sha256")
                    == active.metadata.get("dataset", {}).get("split_sha256"))
        reasons = []
        if cid == self._active_id:
            reasons.append("already the active model")
        if not same_val:
            reasons.append("validation sets differ, so accuracies are not comparable")
        if c_acc is None or a_acc is None:
            reasons.append("missing validation accuracy")
        elif c_acc < a_acc - tolerance:
            reasons.append(f"validation accuracy {c_acc:.4f} is more than {tolerance:.3f} below the active model's {a_acc:.4f}")
        if cand.metadata.get("source") == "ui-finetune":
            reasons += self._user_holdout_reasons(cand, c_ev.get("user_holdout"))
        return {"candidate_id": cid, "passed": not reasons, "reasons": reasons, "tolerance": tolerance,
                "candidate_validation_accuracy": c_acc, "active_validation_accuracy": a_acc,
                "active_id": self._active_id}

    def _user_holdout_reasons(self, cand, uh) -> list:
        from .feedback import MIN_HOLDOUT, MIN_HOLDOUT_DIGITS
        out = []
        if not isinstance(uh, dict) or uh.get("accuracy_before") is None or uh.get("accuracy_after") is None:
            return ["no held-out drawings of yours were used to check this model"]
        if uh.get("n", 0) < MIN_HOLDOUT or uh.get("digits", 0) < MIN_HOLDOUT_DIGITS:
            out.append(f"the held-out check was too small ({uh.get('n', 0)} drawings over {uh.get('digits', 0)} digits)")
        if uh["accuracy_after"] < uh["accuracy_before"]:
            out.append("accuracy on your held-out drawings got worse")
        if cand.metadata.get("parent") != self._active_id:
            out.append("it was fine-tuned from a different model than the one that is active now")
        return out

    def promote(self, cid: str, tolerance: float = 0.003) -> dict:
        with self._lock:
            self._check_id(cid)
            decision = self.evaluate_gate(cid, tolerance)
            if decision["passed"]:
                self._history.append(self._active_id)
                self._history = self._history[-10:]
                self._active_id = cid
                self._write_registry()
                log.info("promoted %s (history %s)", cid, self._history)
            return decision

    def rollback(self) -> str:
        with self._lock:
            if not self._history:
                raise StoreError("nothing to roll back to")
            self._active_id = self._history.pop()
            self.get(self._active_id)
            self._write_registry()
            log.info("rolled back to %s", self._active_id)
            return self._active_id

    def history(self) -> list[str]:
        return list(self._history)
