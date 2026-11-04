# Performance

Measured by `tools/benchmark.py` on the machine below. Treat as relative, not absolute: other hardware, BLAS builds and thread counts differ.

- Python 3.12.3, NumPy 2.4.4, Linux-6.18.44-fc-v233-x86_64-with-glibc2.39, x86_64, 1 logical CPU(s)

## Training step (architecture 784-128-64-9, batch 64, Adam + weight decay)

| Configuration | ms / step | est. s / epoch (48,670 samples) |
|---|---|---|
| float64, no augmentation | 1.85 | 1.4 |
| float32, no augmentation | 1.17 | 0.9 |
| float32, with augmentation (release config) | 2.17 | 1.7 |

Share of a step spent in each phase:

| Configuration | data | forward | loss | backward | optimizer |
|---|---|---|---|---|---|
| float64, no augmentation | 6% | 21% | 4% | 30% | 39% |
| float32, no augmentation | 4% | 20% | 7% | 25% | 44% |
| float32, with augmentation (release config) | 44% | 12% | 4% | 15% | 25% |

Augmentation runs on a helper thread during real training (`prefetch`), so on a multi-core machine its cost overlaps with the step; the step-time table above runs it inline (worst case).

## Inference (float32)

- Single sample (normalise + forward + softmax): median 0.027 ms, p95 0.046 ms
- Batch of 4,096 samples: 14.1 ms (3.4 us / sample)

## Efficiency measures in the code

- float32 training and inference (half the memory traffic; float64 is kept for gradient checks).
- Adam with bias correction folded into the step size and in-place updates on preallocated scratch buffers (no temporaries per step).
- Fused softmax + cross-entropy (`P - Y`) with a log-sum-exp formulation.
- Augmentation: separable coordinate construction, one flat gather, lerp interpolation; prefetched on a helper thread with an ordered RNG stream so results are identical with or without it.
- Images stay uint8 in memory (about 4x smaller than float32) and are converted per batch.
