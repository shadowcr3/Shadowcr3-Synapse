"""Backpropagation (chain rule through the layer stack) and numerical gradient checking."""

from __future__ import annotations

import warnings

import numpy as np

from .activations import relu_grad
from .loss import cross_entropy_from_logits, softmax_cross_entropy


def backward(network, cache, dlogits: np.ndarray) -> list[np.ndarray]:
    """Propagate dL/dlogits back through the network.

    For layer l with input A_{l-1} and pre-activation Z_l:
        dW_l, db_l, dA_{l-1} = layer.backward(dZ_l, A_{l-1})
        dZ_{l-1} = dA_{l-1} * ReLU'(Z_{l-1})        (hidden layers only)

    Returns gradients aligned with network.parameters(): [dW0, db0, dW1, db1, ...].
    """
    grads: list[np.ndarray] = []
    dz = dlogits
    for l in range(len(network.layers) - 1, -1, -1):
        a_prev, _ = cache[l]
        dW, db, da_prev = network.layers[l].backward(dz, a_prev)
        grads.append(db)
        grads.append(dW)
        if l > 0:
            dz = da_prev * relu_grad(cache[l - 1][1])
    grads.reverse()
    return grads


def gradient_check(network, X: np.ndarray, y: np.ndarray, eps: float = 1e-6,
                   max_per_tensor: int | None = None,
                   rng: np.random.Generator | None = None) -> list[dict]:
    """Compare analytic gradients with central finite differences.

    For each parameter tensor, returns the relative error
        ||g_analytic - g_numeric|| / (||g_analytic|| + ||g_numeric||)
    over all entries, or over a random sample of `max_per_tensor` entries (use this for
    large tensors). Requires float64: finite differences with eps ~ 1e-6 are meaningless
    in float32.
    """
    if network.dtype != np.float64:
        raise ValueError("gradient_check requires a float64 network")
    rng = rng or np.random.default_rng(0)

    logits, cache = network.forward(X)
    _, dlogits = softmax_cross_entropy(logits, y)
    analytic = backward(network, cache, dlogits)

    # ReLU is not differentiable at 0. If a hidden pre-activation lies within a few eps of
    # the kink, finite differences straddle it and disagree with the subgradient used by
    # backpropagation. That is a property of the test point, not a backprop bug.
    near_kink = sum(int(np.sum(np.abs(z) < 10 * eps)) for _, z in cache[:-1])
    if near_kink:
        warnings.warn(f"{near_kink} hidden pre-activation(s) within 10*eps of the ReLU kink; "
                      "gradient-check errors may be spurious (use non-zero biases or other data)",
                      RuntimeWarning, stacklevel=2)

    def loss() -> float:
        return cross_entropy_from_logits(network.predict_logits(X), y)

    results = []
    for name, p, g in zip(network.parameter_names(), network.parameters(), analytic):
        n = p.size
        if max_per_tensor is not None and n > max_per_tensor:
            positions = rng.choice(n, size=max_per_tensor, replace=False)
        else:
            positions = np.arange(n)
        flat = p.reshape(-1)  # view: p is C-contiguous
        numeric = np.empty(len(positions))
        for k, pos in enumerate(positions):
            original = flat[pos]
            flat[pos] = original + eps
            loss_plus = loss()
            flat[pos] = original - eps
            loss_minus = loss()
            flat[pos] = original
            numeric[k] = (loss_plus - loss_minus) / (2.0 * eps)
        ana = g.reshape(-1)[positions]
        denom = max(np.linalg.norm(ana) + np.linalg.norm(numeric), 1e-12)
        results.append({
            "name": name,
            "relative_error": float(np.linalg.norm(ana - numeric) / denom),
            "entries_checked": int(len(positions)),
        })
    return results
