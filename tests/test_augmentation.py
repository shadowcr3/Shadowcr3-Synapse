import unittest

import numpy as np

from training.augmentation import AugmentConfig, augment_batch
from training.dataset import normalize_pixels


def digits(n=32, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 28, 28), dtype=np.uint8)
    x[:, 6:22, 12:16] = rng.integers(180, 256, (n, 16, 4))
    return x.reshape(n, 784)


class TestAugmentation(unittest.TestCase):
    def test_disabled_is_plain_normalisation(self):
        x = digits()
        out = augment_batch(x, np.random.default_rng(0), AugmentConfig(enabled=False), np.float64)
        np.testing.assert_allclose(out, normalize_pixels(x), atol=1e-7)

    def test_zero_strength_is_identity(self):
        cfg = AugmentConfig(True, 0, 0, 0, 0, 0)
        x = digits()
        out = augment_batch(x, np.random.default_rng(0), cfg, np.float64)
        np.testing.assert_allclose(out, normalize_pixels(x), atol=1e-6)

    def test_integer_translation_matches_roll(self):
        cfg = AugmentConfig(True, 0, 0, 0, 0, 0)
        x = digits(4)
        shifted = augment_batch(x, np.random.default_rng(1), AugmentConfig(True, 0, 0, 0, 2.0, 0), np.float64)
        base = normalize_pixels(x).reshape(-1, 28, 28)
        com = lambda im: np.array([(im.sum(1) * np.arange(28)).sum(), (im.sum(0) * np.arange(28)).sum()]) / im.sum()
        moved = [np.linalg.norm(com(a) - com(b)) for a, b in zip(shifted.reshape(-1, 28, 28), base)]
        self.assertTrue(all(m <= 2 * np.sqrt(2) + 1e-3 for m in moved))
        self.assertGreater(max(moved), 0.2)

    def test_shape_range_dtype_determinism(self):
        cfg = AugmentConfig(True)
        x = digits(64)
        a = augment_batch(x, np.random.default_rng(5), cfg, np.float32)
        b = augment_batch(x, np.random.default_rng(5), cfg, np.float32)
        c = augment_batch(x, np.random.default_rng(6), cfg, np.float32)
        self.assertEqual((a.shape, a.dtype), ((64, 784), np.float32))
        self.assertTrue(a.min() >= 0.0 and a.max() <= 1.0)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_samples_get_independent_transforms(self):
        x = np.repeat(digits(1), 32, axis=0)
        out = augment_batch(x, np.random.default_rng(0), AugmentConfig(True), np.float32)
        self.assertGreater(np.unique(out.round(4), axis=0).shape[0], 28)

    def test_ink_roughly_preserved_and_stays_in_frame(self):
        cfg = AugmentConfig(True, thicken_prob=0.0)
        x = digits(200)
        out = augment_batch(x, np.random.default_rng(2), cfg, np.float64)
        ratio = out.sum(1) / normalize_pixels(x).sum(1)
        self.assertTrue(0.7 < np.median(ratio) < 1.4)

    def test_thickening_adds_ink(self):
        x = digits(100)
        off = augment_batch(x, np.random.default_rng(3), AugmentConfig(True, 0, 0, 0, 0, 0.0), np.float64)
        on = augment_batch(x, np.random.default_rng(3), AugmentConfig(True, 0, 0, 0, 0, 1.0), np.float64)
        self.assertTrue(np.all(on.sum(1) > off.sum(1)))


if __name__ == "__main__":
    unittest.main()
