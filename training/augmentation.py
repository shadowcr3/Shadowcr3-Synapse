"""Vectorised on-the-fly data augmentation (NumPy only).

Each sample receives an independent random affine transform (rotation, isotropic scale,
shear, translation) applied by inverse mapping with bilinear interpolation, plus optional
stroke thickening. Fresh variants are drawn every epoch, so the network never sees exactly
the same pixels twice: more effective data without storing any.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIDE = 28
_CENTRE = (SIDE - 1) / 2.0


@dataclass(frozen=True)
class AugmentConfig:
    enabled: bool = False
    rotation_deg: float = 12.0
    scale: float = 0.10          # scale factor drawn from [1 - scale, 1 + scale]
    shear: float = 0.10
    translate_px: float = 2.0
    thicken_prob: float = 0.20   # probability of a 1-px stroke dilation

    @classmethod
    def from_dict(cls, d: dict) -> "AugmentConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


_AX = np.arange(SIDE, dtype=np.float32)
_PAD = SIDE + 2          # padded side: one zero cell on every edge (black background)


def augment_batch(x_uint8: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig,
                  dtype=np.float32) -> np.ndarray:
    """(B, 784) uint8 -> (B, 784) in [0, 1] of `dtype`, randomly transformed per cfg."""
    b = x_uint8.shape[0]
    if not cfg.enabled:
        out = x_uint8.astype(np.float32)
        out *= np.float32(1.0 / 255.0)
        return out.astype(dtype, copy=False)

    f32 = np.float32
    theta = np.deg2rad(rng.uniform(-cfg.rotation_deg, cfg.rotation_deg, b)).astype(f32)
    s = rng.uniform(1.0 - cfg.scale, 1.0 + cfg.scale, b).astype(f32)
    sh = rng.uniform(-cfg.shear, cfg.shear, b).astype(f32)
    tx = rng.uniform(-cfg.translate_px, cfg.translate_px, b).astype(f32)
    ty = rng.uniform(-cfg.translate_px, cfg.translate_px, b).astype(f32)

    # Forward map A = s * R(theta) * [[1, sh], [0, 1]]; sampling needs its inverse.
    c, sn = np.cos(theta), np.sin(theta)
    a11, a12, a21, a22 = s * c, s * (c * sh - sn), s * sn, s * (sn * sh + c)
    det = a11 * a22 - a12 * a21
    i11, i12, i21, i22 = a22 / det, -a12 / det, -a21 / det, a11 / det

    # src = Ainv @ ([x, y] - centre - t) + centre, built as (B,1,28) + (B,28,1) outer sums.
    cen = f32(_CENTRE)
    ox = (_AX[None, :] - cen - tx[:, None])           # (B, 28): x offsets
    oy = (_AX[None, :] - cen - ty[:, None])           # (B, 28): y offsets
    src_x = (i11[:, None] * ox)[:, None, :] + (i12[:, None] * oy)[:, :, None] + cen
    src_y = (i21[:, None] * ox)[:, None, :] + (i22[:, None] * oy)[:, :, None] + cen
    hi = f32(SIDE - 1e-3)                              # keeps x0 + 1 inside the padded frame
    np.clip(src_x, -1.0, hi, out=src_x)               # outside -> the zero border
    np.clip(src_y, -1.0, hi, out=src_y)

    x0 = np.floor(src_x)
    y0 = np.floor(src_y)
    fx = src_x - x0
    fy = src_y - y0
    # flat index of the top-left tap inside the zero-padded (B, 30, 30) array
    base = (np.arange(b, dtype=np.int32) * (_PAD * _PAD))[:, None, None]
    idx = base + (y0.astype(np.int32) + 1) * _PAD + (x0.astype(np.int32) + 1)

    padded = np.zeros((b, _PAD, _PAD), dtype=f32)
    padded[:, 1:-1, 1:-1] = x_uint8.reshape(b, SIDE, SIDE)
    flat = padded.reshape(-1)
    t00 = np.take(flat, idx)
    t01 = np.take(flat, idx + 1)
    t10 = np.take(flat, idx + _PAD)
    t11 = np.take(flat, idx + _PAD + 1)
    t01 -= t00
    t01 *= fx
    t01 += t00                                         # top row lerp
    t11 -= t10
    t11 *= fx
    t11 += t10                                         # bottom row lerp
    t11 -= t01
    t11 *= fy
    t11 += t01                                         # vertical lerp
    out = t11
    out *= f32(1.0 / 255.0)

    if cfg.thicken_prob > 0:
        pick = np.flatnonzero(rng.random(b) < cfg.thicken_prob)
        if pick.size:
            sub = out[pick]
            p = np.zeros((pick.size, _PAD, _PAD), dtype=f32)
            p[:, 1:-1, 1:-1] = sub
            out[pick] = np.maximum.reduce(
                [sub, p[:, :-2, 1:-1], p[:, 2:, 1:-1], p[:, 1:-1, :-2], p[:, 1:-1, 2:]])
    np.clip(out, 0.0, 1.0, out=out)
    return out.reshape(b, SIDE * SIDE).astype(dtype, copy=False)
