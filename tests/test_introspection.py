import unittest

import numpy as np

from neural_network import Network, backward, softmax_cross_entropy
from neural_network.introspection import layer_names, quantize_rows_int8, trace


def net_and_x(seed=0):
    net = Network([784, 16, 8, 9], seed=seed, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for layer in net.layers:
        layer.b[...] = rng.normal(0, 0.3, layer.b.shape)
    return net, rng.random(784)


class TestTrace(unittest.TestCase):
    def test_forward_matches_network(self):
        net, x = net_and_x()
        t = trace(net, x)
        np.testing.assert_allclose(t["layers"][-1]["probs"], net.predict_proba(x)[0], atol=1e-4)
        self.assertEqual([l["name"] for l in t["layers"]], ["hidden1", "hidden2", "output"])
        self.assertEqual(t["predicted"], int(net.predict(x)[0]))
        self.assertTrue(all(v >= 0 for v in t["layers"][0]["activation"]))

    def test_backward_deltas_match_backprop(self):
        net, x = net_and_x(1)
        y = 4
        t = trace(net, x, target=y)
        logits, cache = net.forward(x)
        _, dl = softmax_cross_entropy(logits, np.array([y]))
        grads = backward(net, cache, dl)
        np.testing.assert_allclose(t["backward"]["deltas"]["output"], grads[5], atol=1e-4)
        np.testing.assert_allclose(t["backward"]["deltas"]["hidden2"], grads[3], atol=1e-4)
        np.testing.assert_allclose(t["backward"]["deltas"]["hidden1"], grads[1], atol=1e-4)

    def test_output_delta_is_p_minus_y(self):
        net, x = net_and_x(2)
        t = trace(net, x, target=7)
        expected = net.predict_proba(x)[0].copy()
        expected[7] -= 1
        np.testing.assert_allclose(t["backward"]["deltas"]["output"], expected, atol=1e-4)
        self.assertAlmostEqual(sum(t["backward"]["deltas"]["output"]), 0.0, places=3)

    def test_dead_relu_units_get_zero_delta(self):
        net, x = net_and_x(3)
        t = trace(net, x, target=1)
        for name in ("hidden1", "hidden2"):
            act = np.array(next(l for l in t["layers"] if l["name"] == name)["activation"])
            delta = np.array(t["backward"]["deltas"][name])
            self.assertTrue(np.all(delta[act == 0] == 0))

    def test_edges_are_sorted_bounded_and_in_range(self):
        net, x = net_and_x()
        t = trace(net, x, target=2, k_edges=25)
        sizes = {"input": 784, "hidden1": 16, "hidden2": 8, "output": 9}
        for key, edges in {**t["edges"], **t["backward"]["edges"]}.items():
            a, b = key.split("_")
            self.assertLessEqual(len(edges), 25)
            mags = [abs(e[2]) for e in edges]
            self.assertEqual(mags, sorted(mags, reverse=True))
            for src, dst, _ in edges:
                self.assertLess(src, sizes[a] if key in t["edges"] else sizes[b])
                self.assertLess(dst, sizes[b] if key in t["edges"] else sizes[a])

    def test_does_not_mutate_network_and_validates_input(self):
        net, x = net_and_x()
        before = net.get_state()
        trace(net, x, target=3)
        for a, b in zip(before, net.parameters()):
            np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            trace(net, x[:100])
        with self.assertRaises(ValueError):
            trace(net, x, target=9)

    def test_quantisation_roundtrip_error_is_small(self):
        W = np.random.default_rng(0).normal(size=(5, 50))
        q, s = quantize_rows_int8(W)
        self.assertEqual(q.dtype, np.int8)
        self.assertLess(np.abs(q * s[:, None] - W).max(), s.max())
        self.assertEqual(layer_names(Network([4, 3, 2, 2])), ["hidden1", "hidden2", "output"])


if __name__ == "__main__":
    unittest.main()
