"""Activation functions. All operate on NumPy arrays and never mutate their input."""

from __future__ import annotations

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    """ReLU(z) = max(0, z)."""
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    """dReLU/dz: 1 where z > 0, else 0 (the subgradient 0 is used at z == 0)."""
    return (z > 0).astype(z.dtype)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax: subtract the per-row max before exponentiating.

    softmax(z) == softmax(z - c) for any constant c, so this changes nothing
    mathematically but prevents exp() overflow for large logits.
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """log(softmax(z)) computed as (z - max) - log(sum(exp(z - max))).

    Avoids the log(0) that occurs when a probability underflows to zero.
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
