
from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from neural_network.backpropagation import backward
from neural_network.loss import softmax_cross_entropy

from .augmentation import AugmentConfig, augment_batch
from .dataset import normalize_pixels
from .evaluator import evaluate_loss_accuracy


class TrainerHooks:
    report_every = 25

    def on_batch(self, info: dict) -> None: ...
    def on_epoch(self, info: dict) -> None: ...
    def should_stop(self) -> bool:
        return False


@dataclass
class TrainSettings:
    epochs: int
    batch_size: int
    learning_rate: float
    lr_schedule: str = "constant"
    lr_min_factor: float = 0.05
    selection_metric: str = "val_loss"
    patience: int = 0
    augment: AugmentConfig = AugmentConfig()
    prefetch: bool = True


def _batches(X, y, order, batch_size, dtype, rng, augment, prefetch):
    def produce():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb = (augment_batch(X[idx], rng, augment, dtype) if augment.enabled
                  else normalize_pixels(X[idx], dtype))
            yield xb, y[idx]

    if not (prefetch and augment.enabled):
        yield from produce()
        return
    q: queue.Queue = queue.Queue(maxsize=8)
    stop = threading.Event()
    DONE = object()

    def worker():
        try:
            for item in produce():
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
            q.put(DONE)
        except BaseException as exc:
            q.put(exc)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        t.join(timeout=2)


def _lr_at(settings: TrainSettings, step: int, total: int) -> float:
    if settings.lr_schedule == "constant" or total <= 1:
        return settings.learning_rate
    floor = settings.learning_rate * settings.lr_min_factor
    progress = min(1.0, step / (total - 1))
    return floor + 0.5 * (settings.learning_rate - floor) * (1.0 + math.cos(math.pi * progress))


def train(network, optimizer, X_train, y_train, X_val, y_val, *, settings: TrainSettings,
          shuffle_seed, augment_seed=None, hooks: TrainerHooks | None = None, log=print) -> dict:
    hooks = hooks or TrainerHooks()
    rng = np.random.default_rng(shuffle_seed)
    aug_rng = np.random.default_rng(augment_seed if augment_seed is not None else shuffle_seed)
    n = len(X_train)
    steps_per_epoch = math.ceil(n / settings.batch_size)
    total_steps = steps_per_epoch * settings.epochs
    keys = ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_seconds")
    history = {k: [] for k in keys}
    best_key, best_state, best_epoch, since_best = None, None, 0, 0
    step, cancelled, stopped_early = 0, False, False

    for epoch in range(1, settings.epochs + 1):
        t0 = time.perf_counter()
        order = rng.permutation(n)
        loss_sum, correct, seen = 0.0, 0, 0
        win_loss, win_correct, win_seen = 0.0, 0, 0
        for xb, yb in _batches(X_train, y_train, order, settings.batch_size, network.dtype,
                               aug_rng, settings.augment, settings.prefetch):
            optimizer.lr = _lr_at(settings, step, total_steps)
            logits, cache = network.forward(xb)
            loss, dlogits = softmax_cross_entropy(logits, yb)
            if not np.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, step {step}")
            grads = backward(network, cache, dlogits)
            optimizer.step(network.parameters(), grads)
            ok = int(np.sum(logits.argmax(axis=1) == yb))
            b = len(yb)
            loss_sum += loss * b; correct += ok; seen += b
            win_loss += loss * b; win_correct += ok; win_seen += b
            step += 1
            if step % hooks.report_every == 0:
                hooks.on_batch({
                    "epoch": epoch, "step": step, "total_steps": total_steps,
                    "step_in_epoch": step - (epoch - 1) * steps_per_epoch,
                    "steps_per_epoch": steps_per_epoch,
                    "loss": win_loss / win_seen, "acc": win_correct / win_seen,
                    "lr": optimizer.lr,
                    "grad_norms": [float(np.linalg.norm(g)) for g in grads],
                })
                win_loss, win_correct, win_seen = 0.0, 0, 0
                if hooks.should_stop():
                    cancelled = True
                    break
        if cancelled:
            break

        val_loss, val_acc = evaluate_loss_accuracy(network, X_val, y_val)
        row = {"train_loss": loss_sum / seen, "train_acc": correct / seen, "val_loss": val_loss,
               "val_acc": val_acc, "lr": optimizer.lr, "epoch_seconds": time.perf_counter() - t0}
        for k in keys:
            history[k].append(row[k])

        key = (val_loss,) if settings.selection_metric == "val_loss" else (-val_acc, val_loss)
        improved = best_key is None or key < best_key
        if improved:
            best_key, best_state, best_epoch, since_best = key, network.get_state(), epoch, 0
        else:
            since_best += 1
        log(f"epoch {epoch:3d}/{settings.epochs}  train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} "
            f"| val loss {val_loss:.4f} acc {val_acc:.4f} | lr {row['lr']:.2e} | {row['epoch_seconds']:.1f}s"
            + (" *" if improved else ""))
        hooks.on_epoch({"epoch": epoch, "epochs": settings.epochs, "best_epoch": best_epoch,
                        "improved": improved, **row})
        if hooks.should_stop():
            cancelled = True
            break
        if settings.patience and since_best >= settings.patience:
            stopped_early = True
            log(f"early stopping: no improvement for {settings.patience} epochs")
            break

    if best_state is not None:
        network.set_state(best_state)
    history.update(best_epoch=best_epoch, cancelled=cancelled, stopped_early=stopped_early,
                   epochs_run=len(history["train_loss"]))
    return history
