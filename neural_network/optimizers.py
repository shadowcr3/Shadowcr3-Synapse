
from __future__ import annotations

from typing import Sequence

import numpy as np


def _check(params: Sequence[np.ndarray], grads: Sequence[np.ndarray]) -> None:
    if len(params) != len(grads):
        raise ValueError(f"{len(params)} parameters but {len(grads)} gradients")
    for p, g in zip(params, grads):
        if p.shape != g.shape:
            raise ValueError(f"parameter shape {p.shape} != gradient shape {g.shape}")


class SGD:
    def __init__(self, learning_rate: float = 0.01) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.lr = learning_rate

    def step(self, params: Sequence[np.ndarray], grads: Sequence[np.ndarray]) -> None:
        _check(params, grads)
        for p, g in zip(params, grads):
            p -= self.lr * g


class Adam:
    def __init__(self, learning_rate: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.0) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        if weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        self.lr, self.beta1, self.beta2, self.eps = learning_rate, beta1, beta2, epsilon
        self.weight_decay = weight_decay
        self.t = 0
        self._m: list[np.ndarray] | None = None
        self._v: list[np.ndarray] | None = None
        self._tmp: list[np.ndarray] | None = None

    def step(self, params: Sequence[np.ndarray], grads: Sequence[np.ndarray]) -> None:
        _check(params, grads)
        if self._m is None:
            self._m = [np.zeros_like(p) for p in params]
            self._v = [np.zeros_like(p) for p in params]
            self._tmp = [np.empty_like(p) for p in params]
        elif len(self._m) != len(params):
            raise ValueError("parameter list changed between steps")
        self.t += 1
        c2 = np.sqrt(1.0 - self.beta2 ** self.t)
        lr_t = self.lr * c2 / (1.0 - self.beta1 ** self.t)
        eps_hat = self.eps * c2
        b1, b2 = self.beta1, self.beta2
        decay = 1.0 - self.lr * self.weight_decay
        for p, g, m, v, tmp in zip(params, grads, self._m, self._v, self._tmp):
            if self.weight_decay and p.ndim == 2:
                p *= decay
            m *= b1
            np.multiply(g, 1.0 - b1, out=tmp)
            m += tmp
            v *= b2
            np.multiply(g, g, out=tmp)
            tmp *= 1.0 - b2
            v += tmp
            np.sqrt(v, out=tmp)
            tmp += eps_hat
            np.divide(m, tmp, out=tmp)
            tmp *= lr_t
            p -= tmp
