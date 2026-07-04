"""Tests for src.data — these run without TensorFlow."""

import numpy as np

from src.data import N_OUTPUTS, generate_polynomial_data


def test_shapes():
    X, Y = generate_polynomial_data(num_samples=20, seed=0)
    assert X.shape == (20, 6)
    assert Y.shape == (20, N_OUTPUTS)


def test_leading_coefficient_nonzero():
    X, _ = generate_polynomial_data(num_samples=50, seed=1)
    assert np.all(X[:, 5] >= 1.0)  # x^5 coefficient constrained to [1, 10]


def test_reproducible_with_seed():
    X1, Y1 = generate_polynomial_data(num_samples=10, seed=7)
    X2, Y2 = generate_polynomial_data(num_samples=10, seed=7)
    assert np.allclose(X1, X2)
    assert np.allclose(Y1, Y2)


def test_sorted_targets_are_nondecreasing_in_real_part():
    _, Y = generate_polynomial_data(num_samples=30, sort_roots=True, seed=3)
    real_parts = Y[:, 0::2]
    # each row's real parts should be sorted ascending
    assert np.all(np.diff(real_parts, axis=1) >= -1e-9)


def test_no_deprecation_warnings():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # turn warnings into failures
        generate_polynomial_data(num_samples=5, seed=0)


def test_add_label_noise_zero_is_identity():
    import numpy as np

    from src.data import add_label_noise, generate_polynomial_data

    _, Y = generate_polynomial_data(num_samples=10, seed=0)
    assert np.array_equal(add_label_noise(Y, 0.0), Y)


def test_add_label_noise_changes_targets_and_is_seeded():
    import numpy as np

    from src.data import add_label_noise, generate_polynomial_data

    _, Y = generate_polynomial_data(num_samples=10, seed=0)
    noisy1 = add_label_noise(Y, 0.1, seed=1)
    noisy2 = add_label_noise(Y, 0.1, seed=1)
    assert not np.array_equal(noisy1, Y)      # noise actually applied
    assert np.array_equal(noisy1, noisy2)      # reproducible with a seed
