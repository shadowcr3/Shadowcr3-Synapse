"""Classification and reject-rule metrics (pure NumPy)."""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def precision_recall_f1(cm: np.ndarray):
    """Per-class (precision, recall, f1, support); 0 where the denominator is 0."""
    tp = np.diag(cm).astype(float)
    pred_total, true_total = cm.sum(axis=0).astype(float), cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred_total, out=np.zeros_like(tp), where=pred_total > 0)
    recall = np.divide(tp, true_total, out=np.zeros_like(tp), where=true_total > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, true_total.astype(np.int64)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15):
    """ECE with equal-width confidence bins: sum_b (n_b/N) * |acc_b - conf_b|.
    Returns (ece, bins) where bins lists per-bin count, mean confidence, accuracy."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece, bins = 0.0, []
    for b in range(n_bins):
        m = which == b
        n = int(m.sum())
        if n:
            c, a = float(conf[m].mean()), float(correct[m].mean())
            ece += (n / len(conf)) * abs(a - c)
        else:
            c = a = None
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": n,
                     "confidence": c, "accuracy": a})
    return float(ece), bins


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=float)
    bounds = np.concatenate(([0], np.flatnonzero(xs[1:] != xs[:-1]) + 1, [len(x)]))
    for start, end in zip(bounds[:-1], bounds[1:]):
        ranks[order[start:end]] = (start + 1 + end) / 2.0
    return ranks


def roc_auc(scores_positive: np.ndarray, scores_negative: np.ndarray) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie) via the rank-sum (Mann-Whitney U) identity."""
    n_pos, n_neg = len(scores_positive), len(scores_negative)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("both groups need at least one sample")
    ranks = _average_ranks(np.concatenate([scores_positive, scores_negative]))
    u = ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))
