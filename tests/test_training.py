import unittest

import numpy as np

from neural_network import Adam, Network
from training.config import apply_overrides, load_config, validate_config
from training.trainer import TrainSettings, train
from training.evaluator import evaluate_loss_accuracy
from training.trainer import train


def synthetic(n_per_class, seed, n_classes=9):
    """uint8 'images' where class c lights up its own block of pixels, plus noise."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 40, (n_per_class * n_classes, 784)).astype(np.uint8)
    y = np.repeat(np.arange(n_classes), n_per_class)
    for i, c in enumerate(y):
        X[i, c * 80:(c + 1) * 80] = rng.integers(150, 256, 80)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def run(seed=0, epochs=5):
    Xtr, ytr = synthetic(40, 1)
    Xva, yva = synthetic(10, 2)
    net = Network([784, 32, 9], seed=seed)
    hist = train(net, Adam(0.001), Xtr, ytr, Xva, yva,
                 settings=TrainSettings(epochs=epochs, batch_size=32, learning_rate=0.001, prefetch=False),
                 shuffle_seed=seed, log=lambda *_: None)
    return net, hist, (Xva, yva)


class TestTraining(unittest.TestCase):
    def test_loss_decreases_and_model_learns(self):
        _, hist, _ = run()
        self.assertLess(hist["train_loss"][-1], hist["train_loss"][0])
        self.assertLess(hist["val_loss"][-1], hist["val_loss"][0])
        self.assertGreater(hist["val_acc"][-1], 0.95)

    def test_training_is_reproducible(self):
        n1, h1, _ = run(seed=3)
        n2, h2, _ = run(seed=3)
        self.assertEqual(h1["train_loss"], h2["train_loss"])
        for a, b in zip(n1.parameters(), n2.parameters()):
            np.testing.assert_array_equal(a, b)

    def test_best_validation_checkpoint_is_restored(self):
        net, hist, (Xva, yva) = run()
        restored_loss, _ = evaluate_loss_accuracy(net, Xva, yva)
        self.assertAlmostEqual(restored_loss, min(hist["val_loss"]), places=12)
        self.assertEqual(hist["val_loss"][hist["best_epoch"] - 1], min(hist["val_loss"]))

    def test_non_finite_loss_aborts(self):
        Xtr, ytr = synthetic(10, 1)
        net = Network([784, 8, 9], seed=0)
        net.layers[0].W[0, 0] = np.nan    # poisoned weight -> NaN logits -> NaN loss
        with self.assertRaises(FloatingPointError):
            train(net, Adam(), Xtr, ytr, Xtr, ytr,
                  settings=TrainSettings(epochs=1, batch_size=16, learning_rate=0.001),
                  shuffle_seed=0, log=lambda *_: None)


class TestConfig(unittest.TestCase):
    def test_default_config_is_valid_and_matches_spec(self):
        cfg = load_config("configs/default.json")
        self.assertEqual(cfg["model"]["layer_sizes"], [784, 128, 64, 9])
        t = cfg["training"]
        self.assertEqual((t["batch_size"], t["learning_rate"], t["seed"]), (64, 0.001, 42))
        self.assertTrue(t["augmentation"]["enabled"])

    def test_unknown_missing_and_bad_values_rejected(self):
        import copy
        base = load_config("configs/default.json")
        bad = copy.deepcopy(base); bad["training"]["typo_key"] = 1
        with self.assertRaises(ValueError):
            validate_config(bad)
        bad = copy.deepcopy(base); del bad["reject"]
        with self.assertRaises(ValueError):
            validate_config(bad)
        bad = copy.deepcopy(base); bad["training"]["epochs"] = True
        with self.assertRaises(ValueError):
            validate_config(bad)
        bad = copy.deepcopy(base); bad["model"]["layer_sizes"] = [784, 128, 10]
        with self.assertRaises(ValueError):
            validate_config(bad)
        bad = copy.deepcopy(base); bad["data"]["classes"] = [1, 1, 2]
        with self.assertRaises(ValueError):
            validate_config(bad)

    def test_overrides_do_not_mutate_original(self):
        base = load_config("configs/default.json")
        new = apply_overrides(base, ["training.epochs=2", "training.augmentation.enabled=false"])
        self.assertEqual(new["training"]["epochs"], 2)
        self.assertFalse(new["training"]["augmentation"]["enabled"])
        self.assertEqual(base["training"]["epochs"], 30)
        with self.assertRaises(ValueError):
            apply_overrides(base, ["training.nope=1"])
        with self.assertRaises(ValueError):
            apply_overrides(base, ["training.epochs=0"])


if __name__ == "__main__":
    unittest.main()


class TestTrainerFeatures(unittest.TestCase):
    def _data(self):
        return synthetic(40, 1), synthetic(10, 2)

    def test_hooks_report_progress_and_can_cancel(self):
        from training.trainer import TrainerHooks

        class H(TrainerHooks):
            report_every = 2
            def __init__(self): self.batches, self.epochs = [], []
            def on_batch(self, info): self.batches.append(info)
            def on_epoch(self, info): self.epochs.append(info)
            def should_stop(self): return len(self.batches) >= 5

        (Xtr, ytr), (Xva, yva) = self._data()
        h = H()
        net = Network([784, 16, 9], seed=0)
        hist = train(net, Adam(0.001), Xtr, ytr, Xva, yva,
                     settings=TrainSettings(epochs=10, batch_size=16, learning_rate=0.001, prefetch=False),
                     shuffle_seed=0, hooks=h, log=lambda *_: None)
        self.assertTrue(hist["cancelled"])
        self.assertEqual(len(h.batches), 5)
        self.assertEqual(len(h.batches[0]["grad_norms"]), len(net.parameters()))
        self.assertLess(hist["epochs_run"], 10)

    def test_early_stopping_and_val_acc_selection(self):
        (Xtr, ytr), (Xva, yva) = self._data()
        net = Network([784, 16, 9], seed=0)
        hist = train(net, Adam(0.001), Xtr, ytr, Xva, yva,
                     settings=TrainSettings(epochs=60, batch_size=16, learning_rate=0.001, patience=2,
                                            selection_metric="val_acc", prefetch=False),
                     shuffle_seed=0, log=lambda *_: None)
        self.assertTrue(hist["stopped_early"] or hist["epochs_run"] == 60)
        self.assertEqual(max(hist["val_acc"]), hist["val_acc"][hist["best_epoch"] - 1])

    def test_cosine_schedule_decays_to_floor(self):
        from training.trainer import _lr_at
        s = TrainSettings(epochs=4, batch_size=8, learning_rate=1.0, lr_schedule="cosine", lr_min_factor=0.1)
        self.assertAlmostEqual(_lr_at(s, 0, 100), 1.0)
        self.assertAlmostEqual(_lr_at(s, 99, 100), 0.1)
        self.assertTrue(_lr_at(s, 50, 100) < 1.0 and _lr_at(s, 50, 100) > 0.1)

    def test_prefetch_does_not_change_results(self):
        from training.augmentation import AugmentConfig
        (Xtr, ytr), (Xva, yva) = self._data()
        out = []
        for prefetch in (False, True):
            net = Network([784, 16, 9], seed=0)
            train(net, Adam(0.001), Xtr, ytr, Xva, yva,
                  settings=TrainSettings(epochs=2, batch_size=16, learning_rate=0.001,
                                         augment=AugmentConfig(True), prefetch=prefetch),
                  shuffle_seed=1, augment_seed=2, log=lambda *_: None)
            out.append(net.get_state())
        for a, b in zip(*out):
            np.testing.assert_array_equal(a, b)

    def test_float32_training_works(self):
        (Xtr, ytr), (Xva, yva) = self._data()
        net = Network([784, 16, 9], seed=0, dtype=np.float32)
        hist = train(net, Adam(0.001), Xtr, ytr, Xva, yva,
                     settings=TrainSettings(epochs=3, batch_size=16, learning_rate=0.001, prefetch=False),
                     shuffle_seed=0, log=lambda *_: None)
        self.assertEqual(net.parameters()[0].dtype, np.float32)
        self.assertLess(hist["train_loss"][-1], hist["train_loss"][0])
