import unittest
import warnings
from unittest import mock

import numpy as np

from neural_network import Network, backward, gradient_check
from neural_network import backpropagation
from neural_network.activations import softmax
from neural_network.loss import cross_entropy_from_logits, softmax_cross_entropy

TOL = 1e-6


def _randomised_net(sizes, seed):
    net = Network(sizes, seed=seed)
    rng = np.random.default_rng(seed + 100)
    for layer in net.layers:
        layer.b[...] = rng.normal(0.0, 0.5, layer.b.shape)
    return net


class TestGradients(unittest.TestCase):
    def _check(self, net, X, y, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            results = gradient_check(net, X, y, **kw)
        for r in results:
            self.assertLess(r["relative_error"], TOL, msg=str(r))
        return results

    def test_small_network_all_parameters(self):
        rng = np.random.default_rng(0)
        net = _randomised_net([6, 5, 4, 3], seed=2)
        results = self._check(net, rng.normal(size=(7, 6)), rng.integers(0, 3, 7))
        self.assertEqual(sum(r["entries_checked"] for r in results), net.num_parameters())

    def test_deeper_network(self):
        rng = np.random.default_rng(1)
        net = _randomised_net([5, 6, 6, 5, 4], seed=3)
        self._check(net, rng.normal(size=(9, 5)), rng.integers(0, 4, 9))

    def test_full_size_architecture_sampled(self):
        rng = np.random.default_rng(2)
        net = _randomised_net([784, 128, 64, 9], seed=4)
        X = rng.random((8, 784))
        self._check(net, X, rng.integers(0, 9, 8), max_per_tensor=30, rng=rng)

    def test_logit_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(3)
        logits, y = rng.normal(size=(4, 9)), rng.integers(0, 9, 4)
        _, g = softmax_cross_entropy(logits, y)
        eps, num = 1e-6, np.zeros_like(logits)
        for i in range(logits.shape[0]):
            for j in range(logits.shape[1]):
                lp, lm = logits.copy(), logits.copy()
                lp[i, j] += eps; lm[i, j] -= eps
                num[i, j] = (cross_entropy_from_logits(lp, y) - cross_entropy_from_logits(lm, y)) / (2 * eps)
        np.testing.assert_allclose(g, num, atol=1e-8)

    def test_checker_detects_a_wrong_gradient(self):
        rng = np.random.default_rng(4)
        net = _randomised_net([6, 5, 4, 3], seed=5)
        X, y = rng.normal(size=(7, 6)), rng.integers(0, 3, 7)
        real_backward = backpropagation.backward
        with mock.patch.object(backpropagation, "backward",
                               side_effect=lambda n, c, d: [g * 1.1 for g in real_backward(n, c, d)]):
            results = gradient_check(net, X, y)
        self.assertTrue(all(r["relative_error"] > 1e-2 for r in results))

    def test_checker_warns_on_relu_kink(self):
        net = Network([4, 3, 2], seed=0)
        X = np.zeros((2, 4))
        with self.assertWarns(RuntimeWarning):
            gradient_check(net, X, np.array([0, 1]))

    def test_checker_requires_float64(self):
        net = Network([4, 3, 2], seed=0, dtype=np.float32)
        with self.assertRaises(ValueError):
            gradient_check(net, np.zeros((2, 4)), np.array([0, 1]))

    def test_parameters_restored_after_check(self):
        rng = np.random.default_rng(5)
        net = _randomised_net([6, 5, 3], seed=6)
        before = net.get_state()
        gradient_check(net, rng.normal(size=(4, 6)), rng.integers(0, 3, 4))
        for a, b in zip(before, net.parameters()):
            np.testing.assert_array_equal(a, b)

    def test_gradients_align_with_parameters(self):
        rng = np.random.default_rng(6)
        net = _randomised_net([6, 5, 3], seed=7)
        logits, cache = net.forward(rng.normal(size=(4, 6)))
        _, dl = softmax_cross_entropy(logits, rng.integers(0, 3, 4))
        for g, p in zip(backward(net, cache, dl), net.parameters()):
            self.assertEqual(g.shape, p.shape)


if __name__ == "__main__":
    unittest.main()
