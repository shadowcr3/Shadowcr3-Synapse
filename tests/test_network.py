import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neural_network import Adam, Network, backward, softmax_cross_entropy


class TestNetwork(unittest.TestCase):
    def test_parameter_count_of_synapse_v1(self):
        net = Network([784, 128, 64, 9])
        expected = (784 * 128 + 128) + (128 * 64 + 64) + (64 * 9 + 9)
        self.assertEqual(expected, 109_321)
        self.assertEqual(net.num_parameters(), expected)

    def test_shapes_and_probabilities(self):
        net = Network([784, 128, 64, 9], seed=1)
        X = np.random.default_rng(0).random((10, 784))
        logits, cache = net.forward(X)
        self.assertEqual(logits.shape, (10, 9))
        self.assertEqual(len(cache), 3)
        probs = net.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)
        self.assertTrue(np.all(probs >= 0))

    def test_single_sample_1d_input_accepted(self):
        net = Network([784, 8, 9], seed=1)
        self.assertEqual(net.predict_proba(np.zeros(784)).shape, (1, 9))

    def test_initialisation_statistics_and_zero_biases(self):
        net = Network([784, 128, 64, 9], seed=3)
        self.assertAlmostEqual(net.layers[0].W.std(), np.sqrt(2 / 784), delta=0.05 * np.sqrt(2 / 784))
        self.assertAlmostEqual(net.layers[1].W.std(), np.sqrt(2 / 128), delta=0.06 * np.sqrt(2 / 128))
        self.assertAlmostEqual(net.layers[2].W.std(), np.sqrt(2 / (64 + 9)), delta=0.15 * np.sqrt(2 / 73))
        self.assertTrue(all(np.all(layer.b == 0) for layer in net.layers))

    def test_same_seed_same_weights_different_seed_different(self):
        a, b, c = Network([20, 8, 3], seed=7), Network([20, 8, 3], seed=7), Network([20, 8, 3], seed=8)
        for x, y in zip(a.parameters(), b.parameters()):
            np.testing.assert_array_equal(x, y)
        self.assertFalse(np.array_equal(a.layers[0].W, c.layers[0].W))

    def test_input_validation(self):
        net = Network([4, 3, 2])
        with self.assertRaises(ValueError):
            net.forward(np.zeros((2, 5)))
        with self.assertRaises(ValueError):
            net.forward(np.array([[np.nan, 0, 0, 0]]))
        with self.assertRaises(ValueError):
            net.forward(np.array([[np.inf, 0, 0, 0]]))
        with self.assertRaises(TypeError):
            net.forward(np.array([["a", "b", "c", "d"]]))
        with self.assertRaises(ValueError):
            net.forward(np.zeros((2, 4, 1)))

    def test_invalid_architecture(self):
        with self.assertRaises(ValueError):
            Network([10])
        with self.assertRaises(ValueError):
            Network([10, 0, 3])

    def test_state_roundtrip_and_validation(self):
        net = Network([6, 4, 3], seed=1)
        saved = net.get_state()
        for p in net.parameters():
            p += 1.0
        net.set_state(saved)
        for a, b in zip(saved, net.parameters()):
            np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            net.set_state(saved[:-1])

    def test_learns_xor(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0])
        net, opt = Network([2, 8, 2], seed=0), Adam(0.05)
        first = softmax_cross_entropy(net.forward(X)[0], y)[0]
        for _ in range(400):
            logits, cache = net.forward(X)
            _, dl = softmax_cross_entropy(logits, y)
            opt.step(net.parameters(), backward(net, cache, dl))
        last = softmax_cross_entropy(net.forward(X)[0], y)[0]
        self.assertLess(last, first)
        self.assertLess(last, 0.05)
        np.testing.assert_array_equal(net.predict(X), y)

    def test_single_neuron_learns_linearly_separable_data(self):
        # M1: a one-layer network (a single softmax unit pair) separates AND.
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 0, 0, 1])
        net, opt = Network([2, 2], seed=0), Adam(0.1)
        for _ in range(300):
            logits, cache = net.forward(X)
            _, dl = softmax_cross_entropy(logits, y)
            opt.step(net.parameters(), backward(net, cache, dl))
        np.testing.assert_array_equal(net.predict(X), y)


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_save_load_predictions_identical(self):
        net = Network([784, 32, 9], seed=5)
        net.save(self.dir, "m", {"class_mapping": {"0": 1}})
        loaded, meta = Network.load(self.dir, "m")
        X = np.random.default_rng(0).random((20, 784))
        np.testing.assert_array_equal(net.predict_logits(X), loaded.predict_logits(X))
        self.assertEqual(meta["parameter_count"], net.num_parameters())
        self.assertEqual(meta["architecture"]["layer_sizes"], [784, 32, 9])
        self.assertEqual(meta["class_mapping"], {"0": 1})

    def test_tampered_weights_rejected(self):
        net = Network([6, 4, 3], seed=1)
        npz, _ = net.save(self.dir, "m")
        other = Network([6, 4, 3], seed=2)
        np.savez_compressed(npz, **dict(zip(other.parameter_names(), other.parameters())))
        with self.assertRaises(ValueError):
            Network.load(self.dir, "m")

    def test_architecture_mismatch_rejected(self):
        net = Network([6, 4, 3], seed=1)
        _, js = net.save(self.dir, "m")
        meta = json.loads(js.read_text())
        meta["architecture"]["layer_sizes"] = [6, 5, 3]
        js.write_text(json.dumps(meta))
        with self.assertRaises(ValueError):
            Network.load(self.dir, "m")

    def test_non_finite_weights_rejected(self):
        net = Network([6, 4, 3], seed=1)
        net.layers[0].W[0, 0] = np.nan
        net.save(self.dir, "m")
        with self.assertRaises(ValueError):
            Network.load(self.dir, "m")

    def test_pickled_npz_is_refused(self):
        net = Network([6, 4, 3], seed=1)
        npz, js = net.save(self.dir, "m")
        np.savez(npz, **{n: np.array([object()], dtype=object) for n in net.parameter_names()})
        import hashlib
        meta = json.loads(js.read_text())
        meta["npz_sha256"] = hashlib.sha256(Path(npz).read_bytes()).hexdigest()
        js.write_text(json.dumps(meta))
        with self.assertRaises(ValueError):       # numpy refuses object arrays when allow_pickle=False
            Network.load(self.dir, "m")


if __name__ == "__main__":
    unittest.main()
