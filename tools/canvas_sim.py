
from __future__ import annotations

import numpy as np

SIZE = 280


def _bilinear_weights(n_in: int, n_out: int) -> np.ndarray:
    w = np.zeros((n_out, n_in))
    for i in range(n_out):
        x = min(max((i + 0.5) * n_in / n_out - 0.5, 0.0), n_in - 1.0)
        j = int(np.floor(x)); f = x - j; j2 = min(j + 1, n_in - 1)
        w[i, j] += 1 - f; w[i, j2] += f
    return w


def resize_bilinear(img: np.ndarray, nh: int, nw: int) -> np.ndarray:
    return _bilinear_weights(img.shape[0], nh) @ img @ _bilinear_weights(img.shape[1], nw).T


def dilate(img: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return img
    h, w = img.shape
    p = np.pad(img, ((0, 0), (r, r)))
    out = np.max([p[:, i:i + w] for i in range(2 * r + 1)], axis=0)
    p = np.pad(out, ((r, r), (0, 0)))
    return np.max([p[i:i + h, :] for i in range(2 * r + 1)], axis=0)


def place(patch: np.ndarray, rng: np.random.Generator, size: int = SIZE) -> np.ndarray:
    ph, pw = patch.shape
    ph, pw = min(ph, size), min(pw, size)
    canvas = np.zeros((size, size))
    top, left = int(rng.integers(0, size - ph + 1)), int(rng.integers(0, size - pw + 1))
    canvas[top:top + ph, left:left + pw] = patch[:ph, :pw]
    return canvas


def digit_to_canvas(img28: np.ndarray, rng: np.random.Generator, side=(100, 240), thickness: int = 0) -> np.ndarray:
    ys, xs = np.nonzero(img28 > 20)
    crop = img28[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(float)
    h, w = crop.shape
    s = rng.uniform(*side) / max(h, w)
    patch = resize_bilinear(crop, max(2, int(round(h * s))), max(2, int(round(w * s))))
    return place(np.clip(dilate(patch, thickness), 0, 255), rng)


def draw_polyline(points, width: float, size: int = SIZE) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size].astype(float)
    dist = np.full((size, size), np.inf)
    pts = np.asarray(points, dtype=float)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / L2, 0, 1) if L2 > 0 else 0.0
        dist = np.minimum(dist, np.hypot(xx - (x0 + t * dx), yy - (y0 + t * dy)))
    return 255.0 * np.clip(width / 2 + 0.5 - dist, 0, 1)


def scribble(rng: np.random.Generator) -> np.ndarray:
    n = int(rng.integers(3, 9))
    pts = rng.uniform(25, SIZE - 25, (n, 2))
    return draw_polyline(pts, rng.uniform(14, 26))


def _ellipse(cx, cy, rx, ry, n=48, a0=0.0, a1=2 * np.pi):
    a = np.linspace(a0, a1, n)
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)


SHAPES = ("circle", "square", "triangle", "cross", "star", "line", "zigzag", "wave", "spiral", "check")


def shape(kind: str, rng: np.random.Generator) -> np.ndarray:
    c, r = SIZE / 2 + rng.uniform(-30, 30, 2), rng.uniform(55, 110)
    w = rng.uniform(14, 24)
    if kind == "circle":
        pts = _ellipse(*c, r, r * rng.uniform(0.7, 1.0))
    elif kind == "square":
        pts = np.array([[-r, -r], [r, -r], [r, r], [-r, r], [-r, -r]]) + c
    elif kind == "triangle":
        pts = np.array([[0, -r], [r, r], [-r, r], [0, -r]]) + c
    elif kind == "cross":
        return np.maximum(draw_polyline([c + [-r, 0], c + [r, 0]], w), draw_polyline([c + [0, -r], c + [0, r]], w))
    elif kind == "star":
        k = np.arange(11) * 4 * np.pi / 5
        pts = np.stack([c[0] + r * np.sin(k), c[1] - r * np.cos(k)], 1)
    elif kind == "line":
        a = rng.uniform(0, np.pi)
        pts = np.array([c + r * np.array([np.cos(a), np.sin(a)]), c - r * np.array([np.cos(a), np.sin(a)])])
    elif kind == "zigzag":
        pts = np.array([[c[0] - r + i * r / 2, c[1] + (r / 2 if i % 2 else -r / 2)] for i in range(5)])
    elif kind == "wave":
        x = np.linspace(c[0] - r, c[0] + r, 40)
        pts = np.stack([x, c[1] + r / 3 * np.sin((x - x[0]) / r * 3 * np.pi)], 1)
    elif kind == "spiral":
        t = np.linspace(0, 4 * np.pi, 90)
        pts = np.stack([c[0] + r * t / (4 * np.pi) * np.cos(t), c[1] + r * t / (4 * np.pi) * np.sin(t)], 1)
    else:
        pts = np.array([c + [-r, 0], c + [-r / 3, r / 2], c + [r, -r]])
    return draw_polyline(np.clip(pts, 5, SIZE - 5), w)


def noise(rng: np.random.Generator) -> np.ndarray:
    canvas = np.zeros((SIZE, SIZE))
    yy, xx = np.mgrid[:SIZE, :SIZE]
    for _ in range(int(rng.integers(25, 90))):
        cy, cx, r = rng.uniform(0, SIZE), rng.uniform(0, SIZE), rng.uniform(3, 9)
        canvas = np.maximum(canvas, 255.0 * np.clip(r + 0.5 - np.hypot(yy - cy, xx - cx), 0, 1))
    return canvas


def dot(rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[:SIZE, :SIZE]
    cy, cx, r = rng.uniform(20, SIZE - 20), rng.uniform(20, SIZE - 20), rng.uniform(3, 9)
    return 255.0 * np.clip(r + 0.5 - np.hypot(yy - cy, xx - cx), 0, 1)


FONTS = (("DejaVu Sans", "normal", "normal"), ("DejaVu Sans", "bold", "normal"), ("DejaVu Serif", "normal", "normal"),
         ("DejaVu Serif", "bold", "italic"), ("DejaVu Sans Mono", "normal", "normal"), ("DejaVu Sans", "normal", "oblique"))


def glyph(char: str, font, rng: np.random.Generator) -> np.ndarray:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    family, weight, style = font
    fig = Figure(figsize=(2.8, 2.8), dpi=100, facecolor="black")
    FigureCanvasAgg(fig)
    fig.text(0.5, 0.5, char, color="white", family=family, weight=weight, style=style,
             fontsize=float(rng.uniform(110, 170)), ha="center", va="center")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., 0].astype(float)
    img = img[:SIZE, :SIZE] if img.shape[0] >= SIZE else np.pad(img, ((0, SIZE - img.shape[0]), (0, SIZE - img.shape[1])))
    return np.clip(dilate(img, int(rng.integers(2, 5))), 0, 255)
