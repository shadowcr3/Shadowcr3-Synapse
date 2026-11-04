"""Input validation and MNIST-style normalisation.

The browser downsamples the canvas to 28x28 (frontend/preprocess.js) and sends 784 raw
pixels. `parse_pixels` is the server-side gate for that payload. `mnist_style_normalize`
is the reference implementation of the browser algorithm, used by tests and tools to prove
the JavaScript port matches and to measure robustness; it is not on the request path.

MNIST convention (LeCun et al.): crop the digit, scale it to fit a 20x20 box preserving the
aspect ratio, place it in a 28x28 field, and translate so its centre of mass is at the
field centre. Bounding-box centring alone does not reproduce this.
"""

from __future__ import annotations

import numpy as np

SIDE = 28
N_PIXELS = SIDE * SIDE
BOX = 20
INK_THRESHOLD = 32          # pixel value (0-255) counted as "ink"
MIN_BBOX_FRACTION = 0.08    # drawing must span >= 8% of the source canvas side
MIN_INK_PIXELS = 12         # of the 784 normalised pixels
CENTRE = (SIDE - 1) / 2.0   # 13.5 in pixel-index coordinates


class InputError(ValueError):
    code = "invalid_pixels"


class BlankInputError(InputError):
    code = "blank_input"


def parse_pixels(payload) -> np.ndarray:
    """Validate a /predict payload; return a uint8 array of shape (784,).

    Accepts exactly {"pixels": [784 integers in 0..255]}. Rejects everything else,
    including bools, floats, wrong lengths, extra keys, and blank drawings.
    """
    if not isinstance(payload, dict):
        raise InputError('request body must be a JSON object: {"pixels": [...]}')
    unknown = set(payload) - {"pixels"}
    if unknown:
        raise InputError(f"unexpected field(s): {sorted(unknown)}")
    pixels = payload.get("pixels")
    if not isinstance(pixels, list):
        raise InputError('"pixels" must be an array')
    if len(pixels) != N_PIXELS:
        raise InputError(f'"pixels" must contain exactly {N_PIXELS} values, got {len(pixels)}')
    for v in pixels:
        if isinstance(v, bool) or not isinstance(v, int):
            raise InputError('"pixels" values must be integers')
        if not 0 <= v <= 255:
            raise InputError('"pixels" values must be in 0..255')
    arr = np.array(pixels, dtype=np.uint8)
    if int(np.count_nonzero(arr >= INK_THRESHOLD)) < MIN_INK_PIXELS:
        raise BlankInputError("the canvas is empty or the drawing is too small")
    return arr


def to_model_input(pixels: np.ndarray) -> np.ndarray:
    """uint8 (784,) -> float64 (1, 784) in [0, 1]; identical to training normalisation."""
    return pixels.astype(np.float64).reshape(1, N_PIXELS) / 255.0


def _area_weights(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) matrix; row i holds the overlap fraction of output cell i with each
    input pixel (exact box/area filter; rows sum to 1)."""
    s = n_in / n_out
    w = np.zeros((n_out, n_in))
    for i in range(n_out):
        lo, hi = i * s, (i + 1) * s
        for j in range(int(np.floor(lo)), min(n_in, int(np.ceil(hi)))):
            w[i, j] = max(0.0, min(hi, j + 1) - max(lo, j)) / s
    return w


def mnist_style_normalize(img, ink_threshold: float = INK_THRESHOLD,
                          min_bbox_fraction: float = MIN_BBOX_FRACTION) -> np.ndarray:
    """Grayscale (H, W) array, ink bright on dark (0..255) -> uint8 (28, 28) MNIST-style.

    Steps: bounding box of ink -> crop -> scale max side to 20 px (area filter) -> stretch
    peak to 255 -> centre in 28x28 -> shift so the centre of mass is at (13.5, 13.5),
    clamped so no ink leaves the frame. Raises BlankInputError for empty/tiny drawings.
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2:
        raise InputError("image must be 2-D")
    ys, xs = np.nonzero(img > ink_threshold)
    if ys.size == 0:
        raise BlankInputError("the canvas is empty")
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = int(r1 - r0 + 1), int(c1 - c0 + 1)
    if max(h, w) < min_bbox_fraction * max(img.shape):
        raise BlankInputError("the drawing is too small")

    crop = img[r0:r1 + 1, c0:c1 + 1]
    scale = BOX / max(h, w)
    nh, nw = max(1, int(np.floor(h * scale + 0.5))), max(1, int(np.floor(w * scale + 0.5)))
    res = _area_weights(h, nh) @ crop @ _area_weights(w, nw).T
    peak = res.max()
    if peak > 0:
        res = res * (255.0 / peak)

    top, left = (SIDE - nh) // 2, (SIDE - nw) // 2
    field = np.zeros((SIDE, SIDE))
    field[top:top + nh, left:left + nw] = res
    total = field.sum()
    idx = np.arange(SIDE)
    cy, cx = (field.sum(axis=1) * idx).sum() / total, (field.sum(axis=0) * idx).sum() / total
    dy = int(np.floor(CENTRE - cy + 0.5))
    dx = int(np.floor(CENTRE - cx + 0.5))
    dy = max(-top, min(dy, SIDE - top - nh))
    dx = max(-left, min(dx, SIDE - left - nw))

    out = np.zeros((SIDE, SIDE))
    out[top + dy:top + dy + nh, left + dx:left + dx + nw] = res
    return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)
