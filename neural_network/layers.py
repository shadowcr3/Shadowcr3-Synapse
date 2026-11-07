
from __future__ import annotations

from typing import Callable

import numpy as np

Initializer = Callable[[int, int, np.random.Generator, np.dtype], np.ndarray]


def he_normal(fan_in: int, fan_out: int, rng: np.random.Generator, dtype) -> np.ndarray:
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_out, fan_in)).astype(dtype)


def xavier_normal(fan_in: int, fan_out: int, rng: np.random.Generator, dtype) -> np.ndarray:
    return rng.normal(0.0, np.sqrt(2.0 / (fan_in + fan_out)), size=(fan_out, fan_in)).astype(dtype)


class Dense:
    def __init__(self, in_features: int, out_features: int, initializer: Initializer,
                 rng: np.random.Generator, dtype=np.float64) -> None:
        self.W = initializer(in_features, out_features, rng, dtype)
        self.b = np.zeros(out_features, dtype=dtype)

    def forward(self, a_prev: np.ndarray) -> np.ndarray:
        return a_prev @ self.W.T + self.b

    def backward(self, dz: np.ndarray, a_prev: np.ndarray):
        dW = dz.T @ a_prev
        db = dz.sum(axis=0)
        da_prev = dz @ self.W
        return dW, db, da_prev
