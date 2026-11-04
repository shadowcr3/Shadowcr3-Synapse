"""Cross-entropy loss, including the fused softmax + cross-entropy used for training."""

from __future__ import annotations

import numpy as np

from .activations import log_softmax


def cross_entropy(probs: np.ndarray, onehot: np.ndarray, eps: float = 1e-12) -> float:
    """Mean over the batch of L = -sum_i y_i log(P_i). Probabilities are clipped to
    [eps, 1] only to avoid log(0); this reference form is used for testing."""
    p = np.clip(probs, eps, 1.0)
    return float(-np.sum(onehot * np.log(p)) / probs.shape[0])


def _check_targets(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets)
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError(f"targets must have shape ({logits.shape[0]},), got {targets.shape}")
    if targets.dtype.kind not in "iu":
        raise TypeError("targets must be integer class indices")
    if targets.size and (targets.min() < 0 or targets.max() >= logits.shape[1]):
        raise ValueError(f"target index out of range [0, {logits.shape[1] - 1}]")
    return targets


def cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    """Mean cross-entropy from raw logits and integer targets (loss only)."""
    targets = _check_targets(logits, targets)
    logp = log_softmax(logits)
    return float(-np.mean(logp[np.arange(len(targets)), targets]))


def softmax_cross_entropy(logits: np.ndarray, targets: np.ndarray):
    """Fused softmax + cross-entropy. Returns (mean loss, dL/dlogits).

    With P = softmax(z) and one-hot Y, the gradient collapses to (P - Y) / N.
    """
    targets = _check_targets(logits, targets)
    n = len(targets)
    logp = log_softmax(logits)
    rows = np.arange(n)
    loss = float(-np.mean(logp[rows, targets]))
    dlogits = np.exp(logp)
    dlogits[rows, targets] -= 1.0
    dlogits /= n
    return loss, dlogits
