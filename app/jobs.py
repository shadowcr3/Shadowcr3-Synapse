
from __future__ import annotations

import base64
import logging
import secrets
import threading
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np

from neural_network import Adam, Network
from .feedback import MIN_TOTAL as MIN_USER_SAMPLES, holdout_mask, readiness
from neural_network.introspection import quantize_rows_int8, trace
from training import evaluator
from training.augmentation import AugmentConfig
from training.dataset import normalize_pixels
from training.reporting import metrics_report
from training.trainer import TrainerHooks, TrainSettings, train

log = logging.getLogger("synapse.jobs")

MAX_EPOCHS = 60
MAX_PARAMS = 1_500_000
MAX_HIDDEN_LAYERS = 3
HIDDEN_MIN, HIDDEN_MAX = 2, 512
MAX_ARCHS = 8
N_TILES = 24


class JobError(ValueError):
    pass


class JobBusy(RuntimeError):
    pass


class DataUnavailable(RuntimeError):
    pass


def _num(params, name, lo, hi, default, kind=float):
    v = params.get(name, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (kind is int and int(v) != v):
        raise JobError(f"{name} must be a number")
    v = kind(v)
    if not lo <= v <= hi:
        raise JobError(f"{name} must be between {lo} and {hi}")
    return v


def _bool(params, name, default):
    v = params.get(name, default)
    if not isinstance(v, bool):
        raise JobError(f"{name} must be true or false")
    return v


def _hidden(value, n_classes) -> list[int]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_HIDDEN_LAYERS:
        raise JobError(f"hidden sizes must be a list of 1 to {MAX_HIDDEN_LAYERS} integers")
    if any(isinstance(h, bool) or not isinstance(h, int) or not HIDDEN_MIN <= h <= HIDDEN_MAX for h in value):
        raise JobError(f"each hidden size must be an integer from {HIDDEN_MIN} to {HIDDEN_MAX}")
    sizes = [784, *value, n_classes]
    if sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:])) > MAX_PARAMS:
        raise JobError(f"network too large (limit {MAX_PARAMS:,} parameters)")
    return list(value)


def _unknown(params, allowed):
    extra = set(params) - allowed
    if extra:
        raise JobError(f"unknown parameter(s): {sorted(extra)}")


class _Hooks(TrainerHooks):
    report_every = 15

    def __init__(self, mgr: "JobManager", net: Network, ds, *, light: bool, arch_index=None, arch_total=None):
        self.m, self.net, self.ds, self.light = mgr, net, ds, light
        self.arch_index, self.arch_total = arch_index, arch_total
        self._prev = net.get_state()
        self._last_emit = 0.0

    def _ctx(self) -> dict:
        return {} if self.arch_index is None else {"arch_index": self.arch_index, "arch_total": self.arch_total}

    def on_batch(self, info: dict) -> None:
        now = time.monotonic()
        if now - self._last_emit >= 0.12:
            self._last_emit = now
            self.m._emit("batch", **self._ctx(), **info)

    def on_epoch(self, info: dict) -> None:
        cur = self.net.get_state()
        stats = [{"mean_abs_delta": float(np.mean(np.abs(c - p))),
                  "rel_change": float(np.linalg.norm(c - p) / max(np.linalg.norm(p), 1e-12))}
                 for c, p in zip(cur, self._prev)]
        self._prev = cur
        payload = {**self._ctx(), **info, "weight_stats": stats, "names": self.net.parameter_names()}
        if not self.light:
            payload["tiles"] = self.m.tiles(self.net)
            payload["probe"] = self.m.probe(self.net, self.ds)
        self.m._progress.update({"epoch": info["epoch"], "epochs": info["epochs"], **self._ctx()})
        self.m._emit("epoch", **payload)

    def should_stop(self) -> bool:
        if time.monotonic() > self.m._deadline:
            self.m._timed_out = True
            return True
        return self.m._cancel.is_set()


class JobManager:
    def __init__(self, store, feedback, data_provider, config: dict, *, max_events: int = 6000,
                 max_seconds: float = 900.0, gate_tolerance: float = 0.003) -> None:
        self.store, self.feedback, self._data_provider, self.cfg = store, feedback, data_provider, config
        self.max_seconds, self.gate_tolerance = max_seconds, gate_tolerance
        self._lock = threading.RLock()
        self._events: deque = deque(maxlen=max_events)
        self._seq = 0
        self._job: dict | None = None
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._timed_out = False
        self._deadline = 0.0
        self._t0 = time.monotonic()
        self._progress: dict = {}

    def _emit(self, type_: str, **data) -> None:
        with self._lock:
            self._seq += 1
            self._events.append({"seq": self._seq, "type": type_, "t": round(time.monotonic() - self._t0, 3),
                                 "job_id": self._job["id"] if self._job else None, **data})

    def status(self, since: int = 0) -> dict:
        with self._lock:
            events = [e for e in self._events if e["seq"] > since]
            dropped = bool(self._events) and self._events[0]["seq"] > since + 1 and since > 0
            job = None if self._job is None else {**self._job, "progress": dict(self._progress)}
            return {"job": job, "events": events, "next_seq": self._seq, "dropped": dropped,
                    "busy": bool(self._thread and self._thread.is_alive())}

    def _data(self):
        return self._data_provider()

    @staticmethod
    def tiles(net: Network, n: int = N_TILES) -> dict:
        q, scale = quantize_rows_int8(net.layers[0].W[:n])
        return {"layer": "hidden1", "n": int(q.shape[0]), "shape": [28, 28],
                "scales": [float(s) for s in scale], "data": base64.b64encode(q.tobytes()).decode("ascii")}

    @staticmethod
    def probe(net: Network, ds) -> dict:
        want = 6 if len(ds.classes) > 6 else 0
        i = int(np.flatnonzero(ds.y_val == want)[0])
        return {"label_index": int(ds.y_val[i]), "pixels": [int(v) for v in ds.X_val[i]],
                "trace": trace(net, normalize_pixels(ds.X_val[i]), target=int(ds.y_val[i]), k_edges=40)}

    def _augment(self, enabled: bool) -> AugmentConfig:
        d = dict(self.cfg["training"]["augmentation"])
        d["enabled"] = enabled
        return AugmentConfig.from_dict(d)

    def _finalize_candidate(self, net, ds, history, *, source, parent, params, started, extra_eval=None) -> dict:
        classes = list(ds.classes)
        val = evaluator.evaluate_classifier(net, ds.X_val, ds.y_val, classes)
        rej_cfg = self.cfg["reject"]
        reject = None
        if rej_cfg["enabled"]:
            thr = evaluator.select_reject_threshold(evaluator.max_softmax_scores(net, ds.X_val),
                                                    rej_cfg["target_in_distribution_accept"])
            reject = evaluator.evaluate_reject(net, ds.X_val, ds.y_val, ds.X_ood_val, thr)
        latency = evaluator.measure_latency(net, ds.X_val, repeats=300)
        parent_meta = self.store.predictor.metadata
        cid = f"c{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(2)}"
        meta = {
            "model_name": "candidate", "model_version": "ui", "run_id": cid, "source": source, "parent": parent,
            "created_utc": datetime.now(timezone.utc).isoformat(), "test_evaluated": False,
            "class_mapping": {str(i): int(c) for i, c in enumerate(classes)},
            "preprocessing": parent_meta.get("preprocessing"),
            "reject": ({"method": "max_softmax", "threshold": reject["threshold"],
                        "target_in_distribution_accept": rej_cfg["target_in_distribution_accept"],
                        "selected_on": "in-distribution validation split"} if reject else None),
            "training_config": {"layer_sizes": list(net.layer_sizes), "job": params},
            "best_epoch": history["best_epoch"], "dataset": ds.info,
            "evaluation": {"validation_accuracy": val["accuracy"], "validation_loss": val["loss"],
                           "test_accuracy": None, "test_macro_f1": None, "test_ece": None, "test_loss": None,
                           "latency": latency, "reject_validation": reject, "reject_test": None,
                           **(extra_eval or {})},
            "metrics_report": metrics_report(history, val, None, reject, None, latency, ds),
        }
        self.store.save_candidate(net, meta, cid)
        return {"candidate_id": cid, "validation_accuracy": val["accuracy"], "validation_loss": val["loss"],
                "parameter_count": net.num_parameters(), "layer_sizes": list(net.layer_sizes),
                "best_epoch": history["best_epoch"], "seconds": round(time.monotonic() - started, 1)}

    def start(self, kind: str, params: dict) -> dict:
        if not isinstance(params, dict):
            raise JobError("parameters must be a JSON object")
        runners = {"train": (self._validate_train, self._run_train),
                   "finetune": (self._validate_finetune, self._run_finetune),
                   "sweep": (self._validate_sweep, self._run_sweep)}
        if kind not in runners:
            raise JobError(f"unknown job kind {kind!r}")
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise JobBusy("a job is already running")
            ds = self._data()
            validate, run = runners[kind]
            params = validate(params, ds)
            self._cancel.clear()
            self._timed_out = False
            self._t0 = time.monotonic()
            self._deadline = self._t0 + self.max_seconds
            self._progress = {}
            self._job = {"id": f"j{secrets.token_hex(4)}", "kind": kind, "state": "running",
                         "started_utc": datetime.now(timezone.utc).isoformat(), "finished_utc": None,
                         "params": params, "result": None, "error": None}
            self._thread = threading.Thread(target=self._worker, args=(run, ds, params), daemon=True)
            self._thread.start()
            return dict(self._job)

    def cancel(self) -> bool:
        with self._lock:
            running = bool(self._thread and self._thread.is_alive())
            if running:
                self._cancel.set()
            return running

    def wait(self, timeout: float | None = None) -> None:
        t = self._thread
        if t:
            t.join(timeout)

    def _worker(self, run, ds, params) -> None:
        job = self._job
        self._emit("job_started", kind=job["kind"], params=params)
        try:
            result = run(ds, params, time.monotonic())
            state = "cancelled" if (self._cancel.is_set() or self._timed_out) else "done"
            if state == "cancelled":
                result = {"reason": "time limit reached" if self._timed_out else "cancelled by user"}
        except Exception as exc:
            log.exception("job failed")
            state, result = "failed", None
            job["error"] = str(exc) or exc.__class__.__name__
        with self._lock:
            job.update(state=state, result=result, finished_utc=datetime.now(timezone.utc).isoformat())
        self._emit("job_finished", state=state, result=result, error=job["error"])

    def _validate_train(self, p, ds) -> dict:
        _unknown(p, {"hidden_sizes", "epochs", "batch_size", "learning_rate", "augmentation",
                     "weight_decay", "seed", "subset"})
        return {"hidden_sizes": _hidden(p.get("hidden_sizes", [128, 64]), len(ds.classes)),
                "epochs": _num(p, "epochs", 1, MAX_EPOCHS, 8, int),
                "batch_size": _num(p, "batch_size", 16, 512, 64, int),
                "learning_rate": _num(p, "learning_rate", 1e-5, 0.1, 1e-3),
                "augmentation": _bool(p, "augmentation", True),
                "weight_decay": _num(p, "weight_decay", 0.0, 0.01, 1e-4),
                "seed": _num(p, "seed", 0, 2**31 - 1, 42, int),
                "subset": _num(p, "subset", min(500, len(ds.X_train)), len(ds.X_train), len(ds.X_train), int)}

    def _validate_finetune(self, p, ds) -> dict:
        _unknown(p, {"epochs", "learning_rate", "auto_promote"})
        _, y, ids = self.feedback.arrays(ds.classes)
        ready = readiness(ids, y)
        if not ready["ready"]:
            raise JobError("not enough saved drawings to fine-tune safely: " + "; ".join(ready["reasons"]))
        return {"epochs": _num(p, "epochs", 1, 20, 4, int),
                "learning_rate": _num(p, "learning_rate", 1e-6, 1e-2, 3e-4),
                "auto_promote": _bool(p, "auto_promote", True)}

    def _validate_sweep(self, p, ds) -> dict:
        _unknown(p, {"architectures", "epochs", "subset", "batch_size", "learning_rate", "augmentation"})
        archs = p.get("architectures", [[16], [64], [128, 64], [256, 128]])
        if not isinstance(archs, list) or not 1 <= len(archs) <= MAX_ARCHS:
            raise JobError(f"architectures must be a list of 1 to {MAX_ARCHS} hidden-size lists")
        return {"architectures": [_hidden(a, len(ds.classes)) for a in archs],
                "epochs": _num(p, "epochs", 1, 20, 5, int),
                "subset": _num(p, "subset", min(500, len(ds.X_train)), len(ds.X_train), min(15000, len(ds.X_train)), int),
                "batch_size": _num(p, "batch_size", 16, 512, 64, int),
                "learning_rate": _num(p, "learning_rate", 1e-5, 0.1, 1e-3),
                "augmentation": _bool(p, "augmentation", False)}

    def _settings(self, epochs, batch, lr, augment, schedule="cosine") -> TrainSettings:
        return TrainSettings(epochs=epochs, batch_size=batch, learning_rate=lr, lr_schedule=schedule,
                             augment=self._augment(augment), prefetch=True)

    @staticmethod
    def _subset(ds, n, seed=0):
        idx = np.sort(np.random.default_rng(seed).permutation(len(ds.X_train))[:n])
        return ds.X_train[idx], ds.y_train[idx]

    def _run_train(self, ds, p, started) -> dict:
        init_ss, shuffle_ss, aug_ss = np.random.SeedSequence(p["seed"]).spawn(3)
        net = Network([784, *p["hidden_sizes"], len(ds.classes)], seed=init_ss, dtype="float32")
        X, y = self._subset(ds, p["subset"])
        hooks = _Hooks(self, net, ds, light=False)
        self._emit("init", layer_sizes=list(net.layer_sizes), parameter_count=net.num_parameters(),
                   names=net.parameter_names(), tiles=self.tiles(net), probe=self.probe(net, ds),
                   total_epochs=p["epochs"])
        opt = Adam(p["learning_rate"], weight_decay=p["weight_decay"])
        history = train(net, opt, X, y, ds.X_val, ds.y_val,
                        settings=self._settings(p["epochs"], p["batch_size"], p["learning_rate"], p["augmentation"]),
                        shuffle_seed=shuffle_ss, augment_seed=aug_ss, hooks=hooks, log=lambda *_: None)
        if history["cancelled"]:
            return {}
        result = self._finalize_candidate(net, ds, history, source="ui-train", parent=None, params=p, started=started)
        result["gate"] = self.store.evaluate_gate(result["candidate_id"], self.gate_tolerance)
        return result

    def _run_finetune(self, ds, p, started) -> dict:
        active, parent = self.store.predictor, self.store.active_id
        Xu, yu, ids = self.feedback.arrays(ds.classes)
        hold = holdout_mask(ids, yu)
        Xu_tr, yu_tr, Xu_va, yu_va = Xu[~hold], yu[~hold], Xu[hold], yu[hold]

        net = Network(active.network.layer_sizes, seed=0, dtype=active.network.dtype)
        net.set_state(active.network.get_state())
        user_before = evaluator.evaluate_loss_accuracy(net, Xu_va, yu_va)[1]
        _, mnist_before = evaluator.evaluate_loss_accuracy(net, ds.X_val, ds.y_val)

        rng = np.random.default_rng(0)
        n_replay = min(len(ds.X_train), max(2000, 20 * len(yu_tr)))
        replay = rng.choice(len(ds.X_train), size=n_replay, replace=False)
        rep = int(np.clip(np.ceil(0.5 * n_replay / len(yu_tr)), 1, 20))
        X = np.concatenate([ds.X_train[replay], np.repeat(Xu_tr, rep, axis=0)])
        y = np.concatenate([ds.y_train[replay], np.repeat(yu_tr, rep)])

        self._emit("init", layer_sizes=list(net.layer_sizes), parameter_count=net.num_parameters(),
                   names=net.parameter_names(), tiles=self.tiles(net), probe=self.probe(net, ds),
                   total_epochs=p["epochs"], user_samples=int(len(yu_tr)), replay_samples=int(n_replay), repeat=rep)
        hooks = _Hooks(self, net, ds, light=False)
        opt = Adam(p["learning_rate"], weight_decay=self.cfg["training"]["weight_decay"])
        history = train(net, opt, X, y, ds.X_val, ds.y_val,
                        settings=self._settings(p["epochs"], 64, p["learning_rate"], True, "constant"),
                        shuffle_seed=1, augment_seed=2, hooks=hooks, log=lambda *_: None)
        if history["cancelled"]:
            return {}
        user_after = evaluator.evaluate_loss_accuracy(net, Xu_va, yu_va)[1]
        _, mnist_after = evaluator.evaluate_loss_accuracy(net, ds.X_val, ds.y_val)
        holdout = {"n": int(len(yu_va)), "digits": int(len(np.unique(yu_va))),
                   "accuracy_before": user_before, "accuracy_after": user_after,
                   "per_digit_n": {str(ds.classes[int(c)]): int((yu_va == c).sum()) for c in np.unique(yu_va)}}
        result = self._finalize_candidate(net, ds, history, source="ui-finetune", parent=parent, params=p,
                                          started=started, extra_eval={"user_holdout": holdout})
        cid = result["candidate_id"]
        result.update(user_samples=int(len(yu_tr)), user_holdout=int(len(yu_va)),
                      user_holdout_accuracy_before=user_before, user_holdout_accuracy_after=user_after,
                      mnist_validation_accuracy_before=mnist_before, mnist_validation_accuracy_after=mnist_after)
        gate = self.store.evaluate_gate(cid, self.gate_tolerance)
        result["gate"], result["promoted"] = gate, False
        if p["auto_promote"] and gate["passed"]:
            result["gate"] = self.store.promote(cid, self.gate_tolerance)
            result["promoted"] = result["gate"]["passed"]
        return result

    def _run_sweep(self, ds, p, started) -> dict:
        X, y = self._subset(ds, p["subset"])
        results = []
        total = len(p["architectures"])
        self._emit("init", total_epochs=p["epochs"], arch_total=total, subset=int(len(y)))
        for i, hidden in enumerate(p["architectures"]):
            if self._cancel.is_set() or time.monotonic() > self._deadline:
                break
            net = Network([784, *hidden, len(ds.classes)], seed=np.random.SeedSequence(7), dtype="float32")
            t0 = time.monotonic()
            hist = train(net, Adam(p["learning_rate"]), X, y, ds.X_val, ds.y_val,
                         settings=self._settings(p["epochs"], p["batch_size"], p["learning_rate"],
                                                 p["augmentation"], "constant"),
                         shuffle_seed=np.random.SeedSequence(8), augment_seed=np.random.SeedSequence(9),
                         hooks=_Hooks(self, net, ds, light=True, arch_index=i, arch_total=total),
                         log=lambda *_: None)
            if hist["cancelled"]:
                break
            b = hist["best_epoch"] - 1
            r = {"hidden_sizes": hidden, "layer_sizes": list(net.layer_sizes), "parameter_count": net.num_parameters(),
                 "val_acc": hist["val_acc"][b], "val_loss": hist["val_loss"][b], "train_acc": hist["train_acc"][b],
                 "best_epoch": hist["best_epoch"], "seconds": round(time.monotonic() - t0, 1),
                 "history": {k: hist[k] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}}
            results.append(r)
            self._emit("sweep_arch_done", index=i, result=r)
        return {"results": results, "subset": int(len(y)), "epochs": p["epochs"]}
