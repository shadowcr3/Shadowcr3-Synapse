"""Dense (fully connected) layer: parameters, initialisation, and layer-local math.

Convention: a batch X has shape (N, in_features) and the weight matrix W has shape
(out_features, in_features), so the batched form of z = Wx + b is Z = X @ W.T + b.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Initializer = Callable[[int, int, np.random.Generator, np.dtype], np.ndarray]


def he_normal(fan_in: int, fan_out: int, rng: np.random.Generator, dtype) -> np.ndarray:
    """He initialisation, N(0, 2/fan_in): keeps ReLU activation variance stable."""
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_out, fan_in)).astype(dtype)


def xavier_normal(fan_in: int, fan_out: int, rng: np.random.Generator, dtype) -> np.ndarray:
    """Glorot/Xavier initialisation, N(0, 2/(fan_in+fan_out)): suited to the linear
    logit layer, where small initial logits give a near-uniform starting softmax."""
    return rng.normal(0.0, np.sqrt(2.0 / (fan_in + fan_out)), size=(fan_out, fan_in)).astype(dtype)


class Dense:
    """Affine transform z = W a + b (the activation is applied by the Network)."""

    def __init__(self, in_features: int, out_features: int, initializer: Initializer,
                 rng: np.random.Generator, dtype=np.float64) -> None:
        self.W = initializer(in_features, out_features, rng, dtype)
        self.b = np.zeros(out_features, dtype=dtype)

    def forward(self, a_prev: np.ndarray) -> np.ndarray:
        """(N, in) -> (N, out)."""
        return a_prev @ self.W.T + self.b

    def backward(self, dz: np.ndarray, a_prev: np.ndarray):
        """Given dL/dZ (N, out) and the layer input (N, in), return (dW, db, dA_prev).

        dW = dZ^T A_prev   (out, in)
        db = sum_n dZ      (out,)
        dA_prev = dZ W     (N, in)

        dz is expected to already include the 1/N of the mean loss, so no further
        batch scaling happens here.
        """
        dW = dz.T @ a_prev
        db = dz.sum(axis=0)
        da_prev = dz @ self.W
        return dW, db, da_prev
