"""Polynomial data generation and target construction.

This is the *testbed* for the study, not the point of the project. Degree-5
polynomial-root regression is a convenient task with a controllable difficulty
knob (label noise), a smooth-ish loss surface, and a fair evaluation metric
(see src/evaluate.py). The study itself lives in experiments/regime_study.py.

Fixes vs. the original notebook:
  * targets are built explicitly (the original passed a generator to np.hstack,
    which raised a deprecation warning).
  * optional sorted targets, and an explicit label-noise injector so we can dial
    up the noisiness of the training loss signal.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import Polynomial

DEGREE = 5
N_OUTPUTS = 2 * DEGREE  # real + imag per root


def _roots_to_target(coefs: np.ndarray, sort_roots: bool = True) -> np.ndarray:
    """Return a length-10 vector [re0, im0, re1, im1, ...] for one polynomial."""
    roots = Polynomial(coefs).roots()
    if sort_roots:
        roots = roots[np.lexsort((roots.imag, roots.real))]
    target = np.empty(N_OUTPUTS, dtype=np.float64)
    target[0::2] = roots.real
    target[1::2] = roots.imag
    return target


def generate_polynomial_data(
    num_samples: int = 70_000,
    sort_roots: bool = True,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random degree-5 polynomials and their roots.

    Returns
    -------
    X : (num_samples, 6) coefficients.
    Y : (num_samples, 10) interleaved real/imag root parts.
    """
    rng = np.random.default_rng(seed)
    coefficients = rng.uniform(-10, 10, size=(num_samples, 6))
    coefficients[:, 5] = rng.uniform(1, 10, size=num_samples)  # keep it degree 5
    roots = np.array(
        [_roots_to_target(c, sort_roots=sort_roots) for c in coefficients]
    )
    return coefficients, roots


def holdout_split(
    n: int, n_val: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically split ``n`` row indices into (fit, val).

    The validation rows are held *out* of the fit set entirely so that model
    selection scores unseen inputs with clean targets -- no label noise, no test
    leakage. Returns sorted index arrays. Used by the study to carve a clean
    validation set from training data (the test set stays wholly separate).
    """
    if not 0 <= n_val < n:
        raise ValueError(f"n_val must be in [0, {n}); got {n_val}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_idx = np.sort(perm[:n_val])
    fit_idx = np.sort(perm[n_val:])
    return fit_idx, val_idx


def add_label_noise(
    Y: np.ndarray, noise_std: float, seed: int | None = None
) -> np.ndarray:
    """Add zero-mean Gaussian noise to targets.

    This is the study's difficulty knob. Applied to TRAINING targets only, it
    makes the loss signal noisier -- which is precisely the condition under
    which AdaptiveSGD's "revert when loss goes up" rule is expected to struggle,
    because a single noisy evaluation can trigger a needless revert.
    """
    if noise_std <= 0:
        return Y
    rng = np.random.default_rng(seed)
    return Y + rng.normal(0.0, noise_std, size=Y.shape)


if __name__ == "__main__":
    X, Y = generate_polynomial_data(num_samples=5, seed=0)
    print("X shape:", X.shape, "Y shape:", Y.shape)
    print("First target:", np.round(Y[0], 3))
