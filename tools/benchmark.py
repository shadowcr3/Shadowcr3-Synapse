#!/usr/bin/env python3

from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neural_network import Adam, Network, backward, softmax_cross_entropy
from training.augmentation import AugmentConfig, augment_batch
from training.dataset import load_mnist, normalize_pixels


def step_times(dtype, X, y, steps=400, bs=64, augment=False):
    net, opt = Network([784, 128, 64, 9], seed=0, dtype=dtype), Adam(1e-3, weight_decay=1e-4)
    order = np.random.default_rng(0).permutation(len(X))
    rng, cfg = np.random.default_rng(1), AugmentConfig(enabled=augment)
    parts = dict(data=0.0, forward=0.0, loss=0.0, backward=0.0, optimizer=0.0)
    for s in range(steps):
        idx = order[s * bs:(s + 1) * bs]
        t0 = time.perf_counter()
        xb = augment_batch(X[idx], rng, cfg, dtype) if augment else normalize_pixels(X[idx], dtype)
        yb = y[idx]
        t1 = time.perf_counter(); logits, cache = net.forward(xb)
        t2 = time.perf_counter(); _, dl = softmax_cross_entropy(logits, yb)
        t3 = time.perf_counter(); g = backward(net, cache, dl)
        t4 = time.perf_counter(); opt.step(net.parameters(), g)
        t5 = time.perf_counter()
        for k, (a, b) in zip(parts, ((t0, t1), (t1, t2), (t2, t3), (t3, t4), (t4, t5))):
            parts[k] += b - a
    total = sum(parts.values())
    return total / steps * 1000, {k: v / total for k, v in parts.items()}


def main() -> int:
    raw = load_mnist(ROOT / "data" / "mnist", download=False)
    X = raw["x_train"].reshape(-1, 784)
    y = (raw["y_train"] % 9).astype(np.int64)
    assert 400 * 64 <= len(X)
    steps_per_epoch = int(np.ceil(48670 / 64))
    rows, shares = [], {}
    for label, dt, aug in (("float64, no augmentation", np.float64, False), ("float32, no augmentation", np.float32, False),
                           ("float32, with augmentation (release config)", np.float32, True)):
        ms, parts = step_times(dt, X, y, augment=aug)
        rows.append((label, ms, ms * steps_per_epoch / 1000))
        shares[label] = parts

    net = Network([784, 128, 64, 9], seed=0, dtype=np.float32)
    x1 = normalize_pixels(X[:1], np.float32)
    for _ in range(100):
        net.predict_proba(x1)
    t = []
    for _ in range(3000):
        t0 = time.perf_counter(); net.predict_proba(x1); t.append((time.perf_counter() - t0) * 1000)
    big = normalize_pixels(X[:4096], np.float32)
    t0 = time.perf_counter(); net.predict_proba(big); batch_ms = (time.perf_counter() - t0) * 1000

    cpu = platform.processor() or platform.machine()
    lines = ["# Performance", "", "Measured by `tools/benchmark.py` on the machine below. Treat as relative, not absolute: other hardware, BLAS builds and thread counts differ.", "",
             f"- Python {platform.python_version()}, NumPy {np.__version__}, {platform.platform()}, {cpu}, {os.cpu_count()} logical CPU(s)", "",
             "## Training step (architecture 784-128-64-9, batch 64, Adam + weight decay)", "",
             "| Configuration | ms / step | est. s / epoch (48,670 samples) |", "|---|---|---|"]
    lines += [f"| {l} | {ms:.2f} | {ep:.1f} |" for l, ms, ep in rows]
    lines += ["", "Share of a step spent in each phase:", "", "| Configuration | data | forward | loss | backward | optimizer |", "|---|---|---|---|---|---|"]
    for l, p in shares.items():
        lines.append(f"| {l} | " + " | ".join(f"{p[k]:.0%}" for k in ('data', 'forward', 'loss', 'backward', 'optimizer')) + " |")
    lines += ["", "Augmentation runs on a helper thread during real training (`prefetch`), so on a multi-core machine its cost overlaps with the step; the step-time table above runs it inline (worst case).", "",
              "## Inference (float32)", "", f"- Single sample (normalise + forward + softmax): median {np.median(t):.3f} ms, p95 {np.percentile(t, 95):.3f} ms",
              f"- Batch of 4,096 samples: {batch_ms:.1f} ms ({batch_ms / 4096 * 1000:.1f} us / sample)", "",
              "## Efficiency measures in the code", "",
              "- float32 training and inference (half the memory traffic; float64 is kept for gradient checks).",
              "- Adam with bias correction folded into the step size and in-place updates on preallocated scratch buffers (no temporaries per step).",
              "- Fused softmax + cross-entropy (`P - Y`) with a log-sum-exp formulation.",
              "- Augmentation: separable coordinate construction, one flat gather, lerp interpolation; prefetched on a helper thread with an ordered RNG stream so results are identical with or without it.",
              "- Images stay uint8 in memory (about 4x smaller than float32) and are converted per batch.", ""]
    out = ROOT / "docs" / "PERFORMANCE.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
