"""Evaluation metrics.

The important piece here is ``matched_root_error``: because roots are an
unordered set, a plain element-wise MSE penalizes a prediction that has the
right roots in the wrong order. This metric matches predicted roots to true
roots via optimal assignment (Hungarian algorithm) before measuring distance,
which is the fair way to score this task. See SPEC.md section 7.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _to_complex(vec: np.ndarray) -> np.ndarray:
    """Turn a length-10 [re, im, re, im, ...] vector into 5 complex roots."""
    vec = np.asarray(vec)
    return vec[0::2] + 1j * vec[1::2]


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Plain element-wise MSE (order-sensitive). Reported for comparison."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def matched_root_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-root distance after optimal predicted<->true assignment.

    Accepts single samples (length-10) or batches (N x 10). Returns the mean
    matched distance in the complex plane, in the same units as the roots.
    """
    y_true = np.atleast_2d(y_true)
    y_pred = np.atleast_2d(y_pred)
    total = 0.0
    for t, p in zip(y_true, y_pred):
        tc, pc = _to_complex(t), _to_complex(p)
        cost = np.abs(pc[:, None] - tc[None, :])  # 5x5 distances
        row, col = linear_sum_assignment(cost)
        total += cost[row, col].mean()
    return float(total / len(y_true))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return both the naive and the fair metric."""
    return {
        "mse": mse(y_true, y_pred),
        "matched_root_error": matched_root_error(y_true, y_pred),
    }
