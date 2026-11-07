import itertools
import unittest

import numpy as np

from training import evaluator, metrics


class TestMetrics(unittest.TestCase):
    def test_accuracy_and_confusion_matrix(self):
        yt, yp = np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 1, 2, 0])
        self.assertAlmostEqual(metrics.accuracy(yt, yp), 4 / 6)
        np.testing.assert_array_equal(metrics.confusion_matrix(yt, yp, 3),
                                      [[1, 1, 0], [0, 2, 0], [1, 0, 1]])

    def test_precision_recall_f1_hand_computed(self):
        cm = np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        p, r, f1, support = metrics.precision_recall_f1(cm)
        np.testing.assert_allclose(p, [1 / 2, 2 / 3, 1.0])
        np.testing.assert_allclose(r, [1 / 2, 1.0, 1 / 2])
        np.testing.assert_allclose(f1, [1 / 2, 0.8, 2 / 3])
        np.testing.assert_array_equal(support, [2, 2, 2])

    def test_prf_zero_division_is_zero(self):
        cm = np.array([[0, 0], [1, 1]])
        p, r, f1, _ = metrics.precision_recall_f1(cm)
        self.assertEqual((p[0], r[0], f1[0]), (0.0, 0.0, 0.0))

    def test_ece_perfect_and_overconfident(self):
        probs = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(metrics.expected_calibration_error(probs, np.array([0, 1]))[0], 0.0)
        self.assertAlmostEqual(metrics.expected_calibration_error(probs, np.array([1, 0]))[0], 1.0)

    def test_ece_calibrated_example(self):
        probs = np.tile([0.8, 0.2], (10, 1))
        y = np.array([0] * 8 + [1] * 2)
        self.assertAlmostEqual(metrics.expected_calibration_error(probs, y)[0], 0.0, places=12)

    def test_auroc_cases(self):
        self.assertEqual(metrics.roc_auc(np.array([0.9, 0.8]), np.array([0.1, 0.2])), 1.0)
        self.assertEqual(metrics.roc_auc(np.array([0.1, 0.2]), np.array([0.9, 0.8])), 0.0)
        self.assertEqual(metrics.roc_auc(np.array([0.5, 0.5]), np.array([0.5, 0.5])), 0.5)

    def test_auroc_matches_brute_force(self):
        rng = np.random.default_rng(0)
        pos, neg = np.round(rng.random(40), 1), np.round(rng.random(55), 1)
        brute = np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a, b in itertools.product(pos, neg)])
        self.assertAlmostEqual(metrics.roc_auc(pos, neg), brute, places=12)


class TestRejectRule(unittest.TestCase):
    def test_threshold_accepts_at_least_target_fraction(self):
        scores = np.linspace(0.3, 1.0, 1000)
        for target in (0.5, 0.9, 0.95, 1.0):
            t = evaluator.select_reject_threshold(scores, target)
            self.assertGreaterEqual(np.mean(scores >= t), target)
        self.assertLess(np.mean(scores >= evaluator.select_reject_threshold(scores, 0.9)), 0.91)

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            evaluator.select_reject_threshold(np.array([0.5]), 0.0)
        with self.assertRaises(ValueError):
            evaluator.select_reject_threshold(np.array([0.5]), 1.5)


if __name__ == "__main__":
    unittest.main()
