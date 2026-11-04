import json
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np

from app.preprocessing import (BlankInputError, InputError, SIDE, _area_weights, mnist_style_normalize,
                               parse_pixels, to_model_input)

REPO = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")


def bar(x0=130, x1=150, y0=40, y1=240, size=280):
    c = np.zeros((size, size)); c[y0:y1, x0:x1] = 255
    return c


def ring(cx=140, cy=140, r=80, w=18, size=280):
    yy, xx = np.mgrid[:size, :size]
    d = np.hypot(yy - cy, xx - cx)
    return np.where(np.abs(d - r) < w / 2, 255.0, 0.0)


def com(img):
    img = img.astype(float); t = img.sum()
    return (img.sum(1) * np.arange(SIDE)).sum() / t, (img.sum(0) * np.arange(SIDE)).sum() / t


def payload(fill=200, n=40):
    px = [0] * 784
    for i in range(n):
        px[i] = fill
    return {"pixels": px}


class TestParsePixels(unittest.TestCase):
    def test_valid(self):
        arr = parse_pixels(payload())
        self.assertEqual((arr.shape, arr.dtype), ((784,), np.uint8))

    def test_rejects_malformed_payloads(self):
        bad = [None, [], "x", 5, {}, {"pixels": None}, {"pixels": "abc"}, {"pixels": [0] * 783},
               {"pixels": [0] * 785}, {"pixels": [300] + [0] * 783}, {"pixels": [-1] + [0] * 783},
               {"pixels": [1.5] + [0] * 783}, {"pixels": [True] + [0] * 783}, {"pixels": ["1"] * 784},
               {"pixels": [None] * 784}, {"pixels": [200] * 784, "extra": 1}]
        for b in bad:
            with self.assertRaises(InputError, msg=repr(b)[:60]):
                parse_pixels(b)

    def test_blank_and_tiny_are_blank_errors(self):
        for fill, n in ((0, 0), (200, 5), (10, 500)):          # empty / too little ink / too faint
            with self.assertRaises(BlankInputError):
                parse_pixels(payload(fill, n))
        self.assertEqual(BlankInputError.code, "blank_input")

    def test_model_input_scaling(self):
        x = to_model_input(parse_pixels(payload(255, 40)))
        self.assertEqual((x.shape, x.dtype), ((1, 784), np.float64))
        self.assertEqual(x.max(), 1.0)


class TestNormalize(unittest.TestCase):
    def test_area_weights_rows_sum_to_one(self):
        for n_in, n_out in ((200, 20), (20, 20), (7, 20), (280, 13)):
            np.testing.assert_allclose(_area_weights(n_in, n_out).sum(1), 1.0, atol=1e-12)

    def test_output_shape_dtype_and_peak(self):
        out = mnist_style_normalize(ring())
        self.assertEqual((out.shape, out.dtype, out.max()), ((SIDE, SIDE), np.uint8, 255))

    def test_fits_20px_box_and_centres_by_mass(self):
        out = mnist_style_normalize(bar())
        ys, xs = np.nonzero(out)
        self.assertEqual(ys.max() - ys.min() + 1, 20)
        cy, cx = com(out)
        self.assertAlmostEqual(cy, 13.5, delta=0.6); self.assertAlmostEqual(cx, 13.5, delta=0.6)

    def test_position_and_size_invariance(self):
        ref = mnist_style_normalize(ring())
        for img in (ring(cx=100, cy=90), ring(cx=190, cy=180, r=60, w=14), ring(r=100, w=24)):
            out = mnist_style_normalize(img).astype(float)
            self.assertLess(np.abs(out - ref).mean(), 12.0)

    def test_aspect_ratio_is_preserved(self):
        out = mnist_style_normalize(bar(x0=135, x1=145))        # thin tall bar, like a "1"
        ys, xs = np.nonzero(out)
        self.assertGreater((ys.max() - ys.min() + 1) / (xs.max() - xs.min() + 1), 4)

    def test_blank_and_too_small_rejected(self):
        with self.assertRaises(BlankInputError):
            mnist_style_normalize(np.zeros((280, 280)))
        tiny = np.zeros((280, 280)); tiny[100:106, 100:106] = 255
        with self.assertRaises(BlankInputError):
            mnist_style_normalize(tiny)

    def test_edge_touching_drawings_stay_in_frame(self):
        out = mnist_style_normalize(bar(x0=0, x1=20, y0=0, y1=280))
        self.assertGreater(out.sum(), 0)
        self.assertEqual(out.shape, (28, 28))

    def test_rejects_non_2d(self):
        with self.assertRaises(InputError):
            mnist_style_normalize(np.zeros((3, 28, 28)))


@unittest.skipUnless(NODE, "node not installed")
class TestJsParity(unittest.TestCase):
    """frontend/preprocess.js must produce the same pixels as the Python reference."""

    @staticmethod
    def _canvases():
        rng = np.random.default_rng(0)
        out = [bar(), bar(x0=20, x1=40, y0=10, y1=120), ring(), ring(cx=60, cy=200, r=40, w=12),
               bar(x0=0, x1=20, y0=0, y1=280)]
        for _ in range(8):                                     # random thick polylines
            c = np.zeros((280, 280)); yy, xx = np.mgrid[:280, :280]
            pts = rng.integers(40, 240, (4, 2))
            for (y0, x0), (y1, x1) in zip(pts[:-1], pts[1:]):
                for t in np.linspace(0, 1, 40):
                    cy, cx = y0 + t * (y1 - y0), x0 + t * (x1 - x0)
                    c[np.hypot(yy - cy, xx - cx) < 9] = 255
            out.append(c)
        return out

    def _run_js(self, canvases):
        req = {"width": 280, "height": 280,
               "images_b64": [__import__("base64").b64encode(c.astype(np.uint8).tobytes()).decode() for c in canvases]}
        res = subprocess.run([NODE, str(REPO / "tools" / "js_preprocess_cli.js")], input=json.dumps(req),
                             capture_output=True, text=True, check=True, timeout=60)
        return json.loads(res.stdout)

    def test_pixel_parity(self):
        canvases = self._canvases()
        for c, js in zip(canvases, self._run_js(canvases)):
            py = mnist_style_normalize(c)
            self.assertIsNone(js["error"])
            diff = np.abs(np.array(js["pixels"], dtype=int).reshape(28, 28) - py.astype(int))
            self.assertLessEqual(diff.max(), 1)                # rounding at .5 boundaries only

    def test_blank_and_tiny_parity(self):
        tiny = np.zeros((280, 280)); tiny[100:106, 100:106] = 255
        blank, small = self._run_js([np.zeros((280, 280)), tiny])
        self.assertEqual((blank["error"], blank["pixels"]), ("blank", None))
        self.assertEqual((small["error"], small["pixels"]), ("too_small", None))


if __name__ == "__main__":
    unittest.main()
