"""Tests for the AdaptiveSGD decision policy and the held-out split.

The pure pieces (the accept/revert policy, the val split) are tested WITHOUT
TensorFlow, so they run in the light test environment. A single integration test
exercises the real training loops and is skipped when TF is not installed.
"""

import numpy as np
import pytest

from src.data import holdout_split
from src.optimizers import apply_backtracking_decision

HP = dict(grow=1.5, shrink=0.4, lr_min=1e-6, lr_max=1.0, patience=1)


# --------------------------------------------------------------------------- #
# apply_backtracking_decision (pure, no TF)                                    #
# --------------------------------------------------------------------------- #
def test_improvement_grows_lr_and_snapshots():
    d = apply_backtracking_decision(0.5, best_loss=1.0, fails=3, lr=0.1, **HP)
    assert d.action == "improved"
    assert d.lr == pytest.approx(0.15)  # 0.1 * grow
    assert d.best_loss == 0.5
    assert d.fails == 0
    assert d.snapshot_best and not d.restore_best


def test_grow_is_clamped_to_lr_max():
    d = apply_backtracking_decision(0.1, best_loss=1.0, fails=0, lr=0.9, **HP)
    assert d.lr == 1.0  # 0.9 * 1.5 = 1.35 -> clamped to lr_max


def test_revert_shrinks_lr_and_restores():
    # patience=1 -> the first non-improvement reverts immediately
    d = apply_backtracking_decision(2.0, best_loss=1.0, fails=0, lr=0.1, **HP)
    assert d.action == "reverted"
    assert d.lr == pytest.approx(0.04)  # 0.1 * shrink
    assert d.best_loss == 1.0           # best is unchanged on a revert
    assert d.fails == 0
    assert d.restore_best and not d.snapshot_best


def test_shrink_is_clamped_to_lr_min():
    d = apply_backtracking_decision(2.0, best_loss=1.0, fails=0, lr=1e-6, **HP)
    assert d.lr == 1e-6  # already at the floor, cannot go lower


def test_patience_tolerates_before_reverting():
    hp = {**HP, "patience": 2}
    # first miss: tolerate (no revert, lr unchanged, fails increments)
    d1 = apply_backtracking_decision(2.0, best_loss=1.0, fails=0, lr=0.1, **hp)
    assert d1.action == "no-improve"
    assert d1.fails == 1
    assert d1.lr == 0.1
    assert not d1.restore_best
    # second consecutive miss: now revert
    d2 = apply_backtracking_decision(2.0, best_loss=1.0, fails=d1.fails, lr=0.1, **hp)
    assert d2.action == "reverted"
    assert d2.restore_best


# --------------------------------------------------------------------------- #
# holdout_split (pure, no TF)                                                  #
# --------------------------------------------------------------------------- #
def test_holdout_split_partitions_disjointly():
    fit, val = holdout_split(100, 20, seed=0)
    assert len(val) == 20
    assert len(fit) == 80
    assert set(fit).isdisjoint(val)                       # no row is in both
    assert sorted([*fit, *val]) == list(range(100))       # together they cover all
    assert list(fit) == sorted(fit) and list(val) == sorted(val)


def test_holdout_split_is_seeded():
    assert np.array_equal(holdout_split(50, 10, seed=7)[1], holdout_split(50, 10, seed=7)[1])
    assert not np.array_equal(holdout_split(50, 10, seed=7)[1], holdout_split(50, 10, seed=8)[1])


def test_holdout_split_rejects_bad_sizes():
    with pytest.raises(ValueError):
        holdout_split(10, 10, seed=0)   # n_val must leave at least one fit row
    with pytest.raises(ValueError):
        holdout_split(10, -1, seed=0)


# --------------------------------------------------------------------------- #
# integration: the real training loops (needs TensorFlow)                      #
# --------------------------------------------------------------------------- #
def test_training_loops_run_and_respect_budget():
    pytest.importorskip("tensorflow")
    from src.baselines import build_keras_optimizer, train_baseline
    from src.models import build_model
    from src.optimizers import AdaptiveSGD

    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(64, 6)).astype("float32")
    Y = rng.uniform(-1, 1, size=(64, 10)).astype("float32")
    Xv, Yv = X[:16], Y[:16]
    keys = {"loss", "lr", "n_reverts", "steps", "epochs", "best_val_loss", "final_lr"}

    model = build_model(hidden_units=8)
    hist = AdaptiveSGD(model=model, initial_lr=1e-2, lr_min=1e-6, lr_max=1.0).train(
        X, Y, batch_size=8, max_steps=20, eval_every=5, X_val=Xv, Y_val=Yv, verbose=False,
    )
    assert keys <= set(hist)
    assert hist["steps"] == 20                       # fixed compute budget honored
    assert isinstance(hist["n_reverts"], int) and hist["n_reverts"] >= 0
    assert 1e-6 <= hist["final_lr"] <= 1.0           # LR clamp guardrail holds
    assert len(hist["loss"]) == len(hist["lr"]) > 0

    model2 = build_model(hidden_units=8)
    opt = build_keras_optimizer("adam", learning_rate=1e-2)
    hb = train_baseline(model2, opt, X, Y, batch_size=8, max_steps=20, eval_every=5,
                        X_val=Xv, Y_val=Yv, verbose=False)
    assert keys <= set(hb)
    assert hb["steps"] == 20
    assert hb["n_reverts"] == 0                       # baselines never revert
