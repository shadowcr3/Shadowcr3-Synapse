
from __future__ import annotations

import numpy as np

from .activations import relu, relu_grad, softmax
from .loss import softmax_cross_entropy

LAYER_NAMES = ("hidden1", "hidden2", "hidden3")


def layer_names(network) -> list[str]:
    n_hidden = len(network.layers) - 1
    return [f"hidden{i + 1}" for i in range(n_hidden)] + ["output"]


def _top_edges(contrib: np.ndarray, k: int) -> list[list]:
    if k <= 0:
        return []
    flat = np.abs(contrib).ravel()
    k = min(k, flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(-flat[idx])]
    dst, src = np.unravel_index(idx, contrib.shape)
    return [[int(s), int(d), round(float(contrib[d, s]), 5)] for s, d in zip(src, dst)]


def trace(network, pixels01, target: int | None = None, k_edges: int = 80) -> dict:
    x = np.asarray(pixels01, dtype=np.float64).reshape(1, -1)
    if x.shape[1] != network.layer_sizes[0]:
        raise ValueError(f"expected {network.layer_sizes[0]} inputs")
    W = [layer.W.astype(np.float64) for layer in network.layers]
    b = [layer.b.astype(np.float64) for layer in network.layers]
    names = layer_names(network)

    acts, pres = [x], []
    for i in range(len(W)):
        z = acts[-1] @ W[i].T + b[i]
        pres.append(z)
        acts.append(relu(z) if i < len(W) - 1 else z)
    logits = pres[-1][0]
    probs = softmax(pres[-1])[0]

    r = lambda a: np.round(a, 4).tolist()
    layers = [{"name": names[i], "size": int(W[i].shape[0]), "activation": r(acts[i + 1][0])}
              for i in range(len(W) - 1)]
    layers.append({"name": "output", "size": int(W[-1].shape[0]), "logits": r(logits), "probs": r(probs)})
    edges = {f"{'input' if i == 0 else names[i - 1]}_{names[i]}":
             _top_edges(W[i] * acts[i][0][None, :], k_edges) for i in range(len(W))}
    out = {"layers": layers, "edges": edges, "predicted": int(np.argmax(probs))}

    if target is not None:
        if not 0 <= target < W[-1].shape[0]:
            raise ValueError("target out of range")
        loss, d_logits = softmax_cross_entropy(pres[-1], np.array([target]))
        deltas = [None] * len(W)
        deltas[-1] = d_logits[0]
        for i in range(len(W) - 2, -1, -1):
            deltas[i] = (deltas[i + 1] @ W[i + 1]) * relu_grad(pres[i][0])
        back_edges = {f"{names[i]}_{'input' if i == 0 else names[i - 1]}":
                      _top_edges((W[i] * deltas[i][:, None]), k_edges) for i in range(len(W))}
        out["backward"] = {
            "target": int(target), "loss": round(float(loss), 6),
            "deltas": {names[i]: r(deltas[i]) for i in range(len(W))},
            "delta_norms": {names[i]: round(float(np.linalg.norm(deltas[i])), 6) for i in range(len(W))},
            "edges": back_edges,
        }
    return out


def quantize_rows_int8(W: np.ndarray):
    W = np.asarray(W, dtype=np.float64)
    scale = np.maximum(np.abs(W).max(axis=1), 1e-12) / 127.0
    q = np.clip(np.rint(W / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale
