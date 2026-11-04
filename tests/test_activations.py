import unittest

import numpy as np

from neural_network.activations import log_softmax, relu, relu_grad, softmax


class TestActivations(unittest.TestCase):
    def test_relu_values(self):
        z = np.array([[-2.0, -0.0, 0.0, 0.5, 3.0]])
        np.testing.assert_array_equal(relu(z), [[0.0, 0.0, 0.0, 0.5, 3.0]])

    def test_relu_does_not_mutate_input(self):
        z = np.array([-1.0, 2.0]); relu(z)
        np.testing.assert_array_equal(z, [-1.0, 2.0])

    def test_relu_grad(self):
        z = np.array([-1.0, 0.0, 1e-12, 2.0])
        np.testing.assert_array_equal(relu_grad(z), [0.0, 0.0, 1.0, 1.0])

    def test_softmax_rows_sum_to_one(self):
        z = np.random.default_rng(0).normal(size=(50, 9)) * 5
        np.testing.assert_allclose(softmax(z).sum(axis=1), 1.0, atol=1e-12)

    def test_softmax_known_value(self):
        np.testing.assert_allclose(softmax(np.array([[0.0, 0.0]])), [[0.5, 0.5]])
        p = softmax(np.array([[1.0, 2.0, 3.0]]))
        e = np.exp([1.0, 2.0, 3.0])
        np.testing.assert_allclose(p, [e / e.sum()])

    def test_softmax_stable_for_huge_logits(self):
        p = softmax(np.array([[1000.0, 1001.0, 999.0]]))
        self.assertTrue(np.all(np.isfinite(p)))
        np.testing.assert_allclose(p.sum(), 1.0)
        self.assertEqual(p.argmax(), 1)

    def test_softmax_shift_invariant(self):
        z = np.random.default_rng(1).normal(size=(4, 9))
        np.testing.assert_allclose(softmax(z), softmax(z + 123.0), atol=1e-12)

    def test_log_softmax_matches_log_of_softmax_and_survives_underflow(self):
        z = np.random.default_rng(2).normal(size=(4, 9))
        np.testing.assert_allclose(log_softmax(z), np.log(softmax(z)), atol=1e-12)
        extreme = np.array([[0.0, -2000.0]])
        self.assertTrue(np.all(np.isfinite(log_softmax(extreme))))


if __name__ == "__main__":
    unittest.main()
