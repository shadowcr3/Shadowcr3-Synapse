"""Evaluation: accuracy, per-class metrics, calibration, reject rule, latency, charts."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from neural_network.activations import softmax
from neural_network.loss import cross_entropy_from_logits

from . import metrics
from .dataset import normalize_pixels


def batched_logits(network, X_uint8: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    out = np.empty((len(X_uint8), network.layer_sizes[-1]), dtype=network.dtype)
    for s in range(0, len(X_uint8), batch_size):
        out[s:s + batch_size] = network.forward(normalize_pixels(X_uint8[s:s + batch_size], network.dtype))[0]
    return out


def evaluate_loss_accuracy(network, X_uint8, y, batch_size: int = 4096):
    logits = batched_logits(network, X_uint8, batch_size)
    return cross_entropy_from_logits(logits, y), metrics.accuracy(y, logits.argmax(axis=1))


def evaluate_classifier(network, X_uint8, y, classes, n_bins: int = 15) -> dict:
    logits = batched_logits(network, X_uint8)
    probs = softmax(logits)
    pred = probs.argmax(axis=1)
    cm = metrics.confusion_matrix(y, pred, len(classes))
    precision, recall, f1, support = metrics.precision_recall_f1(cm)
    ece, bins = metrics.expected_calibration_error(probs, y, n_bins)
    return {
        "n": int(len(y)),
        "accuracy": metrics.accuracy(y, pred),
        "loss": cross_entropy_from_logits(logits, y),
        "macro_f1": float(f1.mean()),
        "ece": ece,
        "calibration_bins": bins,
        "confusion_matrix": cm.tolist(),
        "per_class": {str(c): {"precision": float(precision[i]), "recall": float(recall[i]),
                               "f1": float(f1[i]), "support": int(support[i])}
                      for i, c in enumerate(classes)},
    }


# ----- reject rule: max-softmax score ----------------------------------------------
def max_softmax_scores(network, X_uint8) -> np.ndarray:
    return softmax(batched_logits(network, X_uint8)).max(axis=1)


def select_reject_threshold(scores_in_distribution: np.ndarray, target_accept: float) -> float:
    """Largest threshold t such that at least `target_accept` of in-distribution
    validation samples satisfy score >= t. Uses in-distribution data only."""
    if not 0.0 < target_accept <= 1.0:
        raise ValueError("target_accept must be in (0, 1]")
    return float(np.quantile(scores_in_distribution, 1.0 - target_accept, method="lower"))


def evaluate_reject(network, X_in, y_in, X_out, threshold: float) -> dict:
    """Accept a sample iff its max softmax probability >= threshold."""
    s_in, s_out = max_softmax_scores(network, X_in), max_softmax_scores(network, X_out)
    pred_in = batched_logits(network, X_in).argmax(axis=1)
    accepted = s_in >= threshold
    return {
        "threshold": float(threshold),
        "in_distribution_accept_rate": float(accepted.mean()),
        "accuracy_on_accepted": float(np.mean(pred_in[accepted] == y_in[accepted])) if accepted.any() else None,
        "accuracy_all": metrics.accuracy(y_in, pred_in),
        "out_of_distribution_reject_rate": float(np.mean(s_out < threshold)),
        "auroc_in_vs_out": metrics.roc_auc(s_in, s_out),
        "n_in": int(len(s_in)), "n_out": int(len(s_out)),
    }


def measure_latency(network, X_uint8, repeats: int = 1000, warmup: int = 50, seed: int = 0) -> dict:
    """Single-sample latency of normalize + forward + softmax (no HTTP, no preprocessing)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X_uint8), size=repeats + warmup)
    times = []
    for k, i in enumerate(idx):
        t0 = time.perf_counter()
        network.predict_proba(normalize_pixels(X_uint8[i:i + 1], network.dtype))
        t1 = time.perf_counter()
        if k >= warmup:
            times.append((t1 - t0) * 1000.0)
    t = np.array(times)
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
            "mean_ms": float(t.mean()), "repeats": repeats}


# ----- charts (optional: requires matplotlib) --------------------------------------
def save_charts(history: dict, test_eval: dict, classes, out_dir) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(epochs, history["train_loss"], label="train")
    ax[0].plot(epochs, history["val_loss"], label="validation")
    ax[0].axvline(history["best_epoch"], ls=":", c="gray", label="selected epoch")
    ax[0].set(xlabel="epoch", ylabel="cross-entropy", title="Loss"); ax[0].legend()
    ax[1].plot(epochs, history["train_acc"], label="train")
    ax[1].plot(epochs, history["val_acc"], label="validation")
    ax[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy"); ax[1].legend()
    fig.tight_layout(); p = out_dir / "training_curves.png"; fig.savefig(p, dpi=130); plt.close(fig)
    written.append(str(p))

    cm = np.array(test_eval["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes,
           xlabel="predicted", ylabel="true", title="Test confusion matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); p = out_dir / "confusion_matrix.png"; fig.savefig(p, dpi=130); plt.close(fig)
    written.append(str(p))

    bins = [b for b in test_eval["calibration_bins"] if b["count"]]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", c="gray", label="perfect calibration")
    ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", label="model")
    ax.set(xlabel="mean confidence", ylabel="accuracy",
           title=f"Reliability (test, ECE={test_eval['ece']:.4f})"); ax.legend()
    fig.tight_layout(); p = out_dir / "reliability.png"; fig.savefig(p, dpi=130); plt.close(fig)
    written.append(str(p))
    return written
