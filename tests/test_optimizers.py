import unittest

import numpy as np

from neural_network import SGD, Adam


class TestOptimizers(unittest.TestCase):
    def test_sgd_step(self):
        p = [np.array([1.0, 2.0])]
        SGD(0.1).step(p, [np.array([10.0, -10.0])])
        np.testing.assert_allclose(p[0], [0.0, 3.0])

    def test_adam_matches_scalar_reference(self):
        lr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8
        grads = [0.5, -0.25, 0.75]
        theta, m, v = 1.0, 0.0, 0.0
        p, opt = [np.array([1.0])], Adam(lr, b1, b2, eps)
        for t, g in enumerate(grads, start=1):
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            theta -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
            opt.step(p, [np.array([g])])
            self.assertAlmostEqual(float(p[0][0]), theta, places=14)

    def test_adam_first_step_is_about_lr(self):
        p = [np.zeros(3)]
        Adam(0.001).step(p, [np.array([5.0, -0.2, 100.0])])
        np.testing.assert_allclose(p[0], [-0.001, 0.001, -0.001], rtol=1e-6)

    def test_adam_minimises_quadratic(self):
        p = [np.array([5.0, -3.0])]
        opt = Adam(0.1)
        for _ in range(500):
            opt.step(p, [2 * p[0]])
        np.testing.assert_allclose(p[0], 0.0, atol=1e-2)

    def test_shape_and_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            Adam().step([np.zeros(3)], [np.zeros(4)])
        with self.assertRaises(ValueError):
            SGD().step([np.zeros(3)], [])

    def test_invalid_hyperparameters(self):
        with self.assertRaises(ValueError):
            Adam(learning_rate=0)
        with self.assertRaises(ValueError):
            Adam(beta1=1.0)


if __name__ == "__main__":
    unittest.main()
