import unittest

import numpy as np

from neural_network.activations import softmax
from neural_network.loss import cross_entropy, cross_entropy_from_logits, softmax_cross_entropy


class TestLoss(unittest.TestCase):
    def test_uniform_prediction_gives_ln_k(self):
        logits = np.zeros((5, 9))
        y = np.array([0, 1, 2, 3, 8])
        self.assertAlmostEqual(cross_entropy_from_logits(logits, y), np.log(9))

    def test_hand_computed_value(self):
        probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        onehot = np.array([[1, 0, 0], [0, 1, 0]])
        self.assertAlmostEqual(cross_entropy(probs, onehot), -(np.log(0.7) + np.log(0.8)) / 2)

    def test_fused_matches_reference(self):
        rng = np.random.default_rng(0)
        logits, y = rng.normal(size=(6, 9)), rng.integers(0, 9, 6)
        onehot = np.eye(9)[y]
        loss, _ = softmax_cross_entropy(logits, y)
        self.assertAlmostEqual(loss, cross_entropy(softmax(logits), onehot))

    def test_gradient_is_p_minus_y_over_n(self):
        rng = np.random.default_rng(1)
        logits, y = rng.normal(size=(6, 9)), rng.integers(0, 9, 6)
        _, g = softmax_cross_entropy(logits, y)
        np.testing.assert_allclose(g, (softmax(logits) - np.eye(9)[y]) / 6)
        np.testing.assert_allclose(g.sum(axis=1), 0.0, atol=1e-15)

    def test_confident_wrong_logits_do_not_overflow(self):
        loss, g = softmax_cross_entropy(np.array([[2000.0, 0.0]]), np.array([1]))
        self.assertTrue(np.isfinite(loss) and np.all(np.isfinite(g)))
        self.assertAlmostEqual(loss, 2000.0)

    def test_invalid_targets_rejected(self):
        logits = np.zeros((3, 4))
        with self.assertRaises(ValueError):
            softmax_cross_entropy(logits, np.array([0, 1, 4]))
        with self.assertRaises(ValueError):
            softmax_cross_entropy(logits, np.array([0, -1, 2]))
        with self.assertRaises(ValueError):
            softmax_cross_entropy(logits, np.array([0, 1]))
        with self.assertRaises(TypeError):
            softmax_cross_entropy(logits, np.array([0.0, 1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
