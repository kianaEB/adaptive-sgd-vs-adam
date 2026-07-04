"""Tests for src.evaluate — these run without TensorFlow."""

import numpy as np

from src.evaluate import matched_root_error, mse


def test_identical_has_zero_error():
    y = np.array([1.0, 0.0, -2.0, 0.5, 3.0, -0.5, 0.0, 1.0, 4.0, 0.0])
    assert matched_root_error(y, y) == 0.0
    assert mse(y, y) == 0.0


def test_permutation_is_penalized_by_mse_but_not_matched():
    # Same set of roots, different order.
    true = np.array([1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0])
    pred = np.array([5.0, 0.0, 4.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 0.0])
    assert mse(true, pred) > 0.0            # order-sensitive metric is fooled
    assert matched_root_error(true, pred) < 1e-9  # fair metric sees they match


def test_batch_input():
    true = np.tile(np.arange(10.0), (4, 1))
    pred = true + 0.1
    err = matched_root_error(true, pred)
    assert err > 0.0
