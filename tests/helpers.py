
from __future__ import annotations

import numpy as np

from neural_network import Adam, Network
from training import evaluator
from training.dataset import Dataset
from training.reporting import metrics_report
from training.trainer import TrainSettings, train

CLASSES = tuple(range(1, 10))


def _images(n_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 40, (n_per_class * 9, 784)).astype(np.uint8)
    y = np.repeat(np.arange(9), n_per_class)
    for i, c in enumerate(y):
        X[i, c * 80:(c + 1) * 80] = rng.integers(150, 256, 80)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def synthetic_dataset(seed: int = 0) -> Dataset:
    Xtr, ytr = _images(40, seed + 1)
    Xva, yva = _images(12, seed + 2)
    Xte, yte = _images(12, seed + 3)
    rng = np.random.default_rng(seed + 4)
    ood = rng.integers(0, 256, (60, 784)).astype(np.uint8)
    info = {"classes": list(CLASSES), "excluded_digits": [0], "val_fraction": 0.1,
            "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte), "n_ood_val": 60, "n_ood_test": 60,
            "split_sha256": "synthetic-split", "file_sha256": {}}
    return Dataset(Xtr, ytr, Xva, yva, Xte, yte, ood, ood.copy(), CLASSES, info)


def make_release(model_dir, ds: Dataset, hidden=(24,), epochs: int = 6, name: str = "synapse_v1", seed: int = 0):
    net = Network([784, *hidden, 9], seed=seed, dtype="float32")
    hist = train(net, Adam(0.003), ds.X_train, ds.y_train, ds.X_val, ds.y_val,
                 settings=TrainSettings(epochs=epochs, batch_size=32, learning_rate=0.003, prefetch=False),
                 shuffle_seed=seed, log=lambda *_: None)
    val = evaluator.evaluate_classifier(net, ds.X_val, ds.y_val, list(CLASSES))
    thr = evaluator.select_reject_threshold(evaluator.max_softmax_scores(net, ds.X_val), 0.95)
    rej = evaluator.evaluate_reject(net, ds.X_val, ds.y_val, ds.X_ood_val, thr)
    lat = evaluator.measure_latency(net, ds.X_val, repeats=20, warmup=2)
    meta = {
        "model_name": name, "model_version": "test", "run_id": f"{name}_test", "created_utc": "2026-01-01T00:00:00+00:00",
        "test_evaluated": False, "class_mapping": {str(i): c for i, c in enumerate(CLASSES)},
        "preprocessing": {"input_shape": [28, 28], "flatten": "row-major", "scale": "uint8 / 255.0"},
        "reject": {"method": "max_softmax", "threshold": thr, "target_in_distribution_accept": 0.95,
                   "selected_on": "validation"},
        "best_epoch": hist["best_epoch"], "dataset": ds.info,
        "evaluation": {"validation_accuracy": val["accuracy"], "validation_loss": val["loss"], "test_accuracy": None,
                       "latency": lat, "reject_validation": rej},
        "metrics_report": metrics_report(hist, val, None, rej, None, lat, ds),
    }
    net.save(model_dir, name, meta)
    return net, meta


def class_pixels(c: int, rng=None) -> list[int]:
    rng = rng or np.random.default_rng(c)
    px = rng.integers(0, 30, 784)
    px[c * 80:(c + 1) * 80] = rng.integers(180, 256, 80)
    return [int(v) for v in px]
