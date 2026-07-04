"""The controlled study.

Question: at what batch size / label-noise level does Adam's advantage over
adaptive backtracking (AdaptiveSGD) appear, and why?

Design (see SPEC.md sections 6-7):
  * Hold the model, dataset, seed, and compute budget (max_steps) FIXED.
  * Vary three things: label-noise level, batch size, optimizer.
  * Score every run on the SAME clean test set with the permutation-aware
    metric, so ordering never confounds the comparison.
  * Also record AdaptiveSGD's revert count -- the diagnostic for the "why".

Run:
    python experiments/regime_study.py            # full sweep
    python experiments/regime_study.py --smoke    # fast check

Outputs land in results/ (a CSV of every run, plus plots).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.preprocessing import MinMaxScaler  # noqa: E402

from src.baselines import build_keras_optimizer, train_baseline  # noqa: E402
from src.config import Config, set_seed  # noqa: E402
from src.data import add_label_noise, generate_polynomial_data  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.models import build_model  # noqa: E402
from src.optimizers import AdaptiveSGD  # noqa: E402

RESULTS = ROOT / "results"


def run_one(cfg, opt_name, batch_size, noise_std, data):
    """Train one config and return a result row scored on the clean test set."""
    Xn, Yn_clean, Xtn, Y_test, sy = data

    # fresh, identically-seeded model so every optimizer starts from the same point
    set_seed(cfg.seed)
    model = build_model(cfg.hidden_units)

    # noise is injected into TRAINING targets only; test set stays clean
    Yn = add_label_noise(Yn_clean, noise_std, seed=cfg.seed)

    if opt_name == "adaptive_sgd":
        trainer = AdaptiveSGD(
            model=model, initial_lr=cfg.initial_lr, grow=cfg.grow,
            shrink=cfg.shrink, lr_min=cfg.lr_min, lr_max=cfg.lr_max,
            patience=cfg.patience,
        )
        history = trainer.train(
            Xn, Yn, epochs=10_000, batch_size=batch_size,
            max_steps=cfg.max_steps, verbose=False,
        )
    else:
        optimizer = build_keras_optimizer(opt_name, learning_rate=cfg.initial_lr)
        history = train_baseline(
            model, optimizer, Xn, Yn, epochs=10_000, batch_size=batch_size,
            max_steps=cfg.max_steps, verbose=False,
        )

    pred = sy.inverse_transform(model.predict(Xtn, verbose=0))
    metrics = evaluate(Y_test, pred)
    return {
        "optimizer": opt_name,
        "batch_size": "full" if batch_size is None else batch_size,
        "noise_std": noise_std,
        "mse": round(metrics["mse"], 5),
        "matched_root_error": round(metrics["matched_root_error"], 5),
        "n_reverts": history.get("n_reverts", 0),
        "steps": history.get("steps", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="fast, tiny sweep")
    args = parser.parse_args()

    cfg = Config()
    if args.smoke:
        cfg.make_smoke()
    set_seed(cfg.seed)

    # Build the testbed ONCE and reuse across every run.
    X, Y = generate_polynomial_data(cfg.n_train, cfg.sort_roots, seed=cfg.seed)
    X_test, Y_test = generate_polynomial_data(cfg.n_test, cfg.sort_roots, seed=cfg.seed + 1)
    sx, sy = MinMaxScaler((-1, 1)), MinMaxScaler((-1, 1))
    Xn, Yn_clean, Xtn = sx.fit_transform(X), sy.fit_transform(Y), sx.transform(X_test)
    data = (Xn, Yn_clean, Xtn, Y_test, sy)

    rows = []
    for noise in cfg.noise_levels:
        for bs in cfg.batch_sizes:
            for opt in cfg.optimizers:
                row = run_one(cfg, opt, bs, noise, data)
                rows.append(row)
                print(
                    f"noise={noise:<5} batch={row['batch_size']:<5} "
                    f"{opt:<13} -> matched_root_error={row['matched_root_error']:.4f} "
                    f"(reverts={row['n_reverts']})"
                )

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / ("regime_study_smoke.csv" if args.smoke else "regime_study.csv")
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} runs to {out.relative_to(ROOT)}")

    # TODO(M5): plotting. The two figures that tell the story:
    #   1. Crossover curve: matched_root_error vs batch size, one line per
    #      optimizer, at a fixed noise level -- shows where Adam overtakes.
    #   2. Heatmap: (batch size x noise) grid of the AdaptiveSGD-minus-Adam gap,
    #      to reveal the regime boundary. Overlay AdaptiveSGD's revert count to
    #      connect the "why" to the "what".
    # plot_results(rows, RESULTS)  # implement in src/plots.py


if __name__ == "__main__":
    main()
