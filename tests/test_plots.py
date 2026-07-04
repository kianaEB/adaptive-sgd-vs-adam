"""Tests for the pure data-shaping in src.plots (no matplotlib / TF needed)."""

import csv
import math

import pytest

from src.plots import (
    load_results,
    order_batches,
    pivot_gap,
    reverts_grid,
    series_by_batch,
)


def _rows():
    """Two optimizers across two regimes; the (full, 0.0) cell has two seeds."""
    return [
        {"optimizer": "adaptive_sgd", "batch_size": "full", "noise_std": 0.0,
         "matched_root_error": 0.10, "n_reverts": 2, "seed": 42},
        {"optimizer": "adaptive_sgd", "batch_size": "full", "noise_std": 0.0,
         "matched_root_error": 0.20, "n_reverts": 4, "seed": 7},   # 2nd seed
        {"optimizer": "adam", "batch_size": "full", "noise_std": 0.0,
         "matched_root_error": 0.15, "n_reverts": 0, "seed": 42},
        {"optimizer": "adaptive_sgd", "batch_size": 64, "noise_std": 0.2,
         "matched_root_error": 0.50, "n_reverts": 30, "seed": 42},
        {"optimizer": "adam", "batch_size": 64, "noise_std": 0.2,
         "matched_root_error": 0.20, "n_reverts": 0, "seed": 42},
    ]


def test_order_batches_full_first_then_descending():
    assert order_batches([64, "full", 512, 4096]) == ["full", "4096", "512", "64"]
    assert order_batches(["512", "512", "64"]) == ["512", "64"]  # dedups


def test_pivot_gap_is_a_minus_b_and_averages_seeds():
    grid, noises, batches = pivot_gap(_rows())
    assert noises == [0.0, 0.2]
    assert batches == ["full", "64"]
    # (full, 0.0): adaptive mean = (0.10+0.20)/2 = 0.15 ; adam = 0.15 -> gap 0.0
    assert grid[0][batches.index("full")] == pytest.approx(0.0, abs=1e-9)
    # (64, 0.2): adaptive 0.50 - adam 0.20 = +0.30 (Adam wins)
    assert grid[1][batches.index("64")] == pytest.approx(0.30)


def test_reverts_grid_averages_over_seeds():
    grid, noises, batches = reverts_grid(_rows())
    assert grid[0][batches.index("full")] == 3.0   # mean(2, 4)
    assert grid[1][batches.index("64")] == 30.0


def test_series_by_batch_follows_batch_order():
    batches, ys = series_by_batch(_rows(), "adaptive_sgd", noise=0.0)
    assert batches == ["full", "64"]
    assert ys[0] == pytest.approx(0.15)   # averaged over seeds
    assert math.isnan(ys[1])           # (64, 0.0) absent -> NaN, handled gracefully


def test_load_results_roundtrips(tmp_path):
    rows = _rows()
    p = tmp_path / "r.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    loaded = load_results(p)
    assert len(loaded) == len(rows)
    # aggregation helpers coerce strings, so the reshaping still works post-CSV
    grid, _, batches = pivot_gap(loaded)
    assert grid[0][batches.index("full")] == pytest.approx(0.0, abs=1e-9)
