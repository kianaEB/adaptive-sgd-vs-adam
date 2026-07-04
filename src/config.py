"""Central configuration and reproducibility helpers.

This project is a *controlled study*: when does adaptive backtracking
(AdaptiveSGD) hold up, and when do per-parameter moment estimates (Adam) win --
as a function of batch size and label noise? The grids below define that sweep.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass
class Config:
    # data / testbed
    n_train: int = 70_000
    n_test: int = 30_000
    # Clean rows held out of the fit set for fair, identical model selection
    # across every optimizer (carved from train; the test set stays separate).
    n_val: int = 5_000
    sort_roots: bool = True
    seed: int = 42

    # model (held fixed across the whole study on purpose)
    hidden_units: int = 64

    # AdaptiveSGD hyperparameters
    initial_lr: float = 1e-3
    grow: float = 1.5
    shrink: float = 0.4
    lr_min: float = 1e-6
    lr_max: float = 1.0
    patience: int = 1

    # ---- the study sweep -------------------------------------------------
    # We vary two things and hold everything else constant.
    # batch size: full-batch -> tiny. (None means full-batch.)
    batch_sizes: tuple = (None, 4096, 512, 64)
    # label-noise std added to NORMALIZED training targets only (test stays clean).
    noise_levels: tuple = (0.0, 0.05, 0.10, 0.20)
    # optimizers to compare.
    optimizers: tuple = ("adaptive_sgd", "adam", "sgd", "rmsprop")

    # Compute budget. To isolate the batch-size variable, fix the number of
    # gradient *updates* (steps), not epochs -- smaller batches would otherwise
    # get many more updates per epoch, confounding the comparison. See SPEC.md
    # section 6 for this design decision.
    max_steps: int = 4000
    # Accept/revert (and model-selection) cadence, in gradient steps. Fixing this
    # across batch sizes gives every run the same number of control decisions
    # (max_steps / eval_every), so the backtracking granularity does not vary
    # with batch size. See SPEC.md section 6.
    eval_every: int = 10

    def make_smoke(self) -> "Config":
        """Shrink everything for a fast end-to-end check."""
        self.n_train, self.n_test, self.n_val = 2000, 500, 200
        self.batch_sizes = (None, 128)
        self.noise_levels = (0.0, 0.1)
        self.optimizers = ("adaptive_sgd", "adam")
        self.max_steps = 200
        self.eval_every = 5
        return self


def set_seed(seed: int) -> None:
    """Seed python, numpy, and (if available) tensorflow for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass
