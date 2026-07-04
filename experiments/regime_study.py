"""The controlled study.

Question: at what batch size / label-noise level does Adam's advantage over
adaptive backtracking (AdaptiveSGD) appear, and why?

Design (see SPEC.md sections 6-7):
  * Hold the model, dataset, seed, and compute budget (max_steps) FIXED.
  * Vary three things: label-noise level, batch size, optimizer.
  * Score every run on the SAME clean test set with the permutation-aware
    metric, so ordering never confounds the comparison.
  * Every optimizer makes its accept/revert and model-selection decisions on the
    SAME fixed step cadence (eval_every), so backtracking granularity does not
    vary with batch size, and the final model is chosen the same fair way.
  * Also record AdaptiveSGD's revert count -- the diagnostic for the "why".

Run:
    python experiments/regime_study.py                 # full sweep
    python experiments/regime_study.py --smoke         # fast check
    python experiments/regime_study.py --seeds 42,7,1  # repeat for variance
    python experiments/regime_study.py --plots-only    # rebuild figures from CSV

Outputs land in results/ (a CSV of every run, plus the two figures).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.preprocessing import MinMaxScaler  # noqa: E402

from src.baselines import build_keras_optimizer, train_baseline  # noqa: E402
from src.config import Config, set_seed  # noqa: E402
from src.data import (  # noqa: E402
    add_label_noise,
    generate_polynomial_data,
    holdout_split,
)
from src.evaluate import evaluate  # noqa: E402
from src.models import build_model  # noqa: E402
from src.optimizers import AdaptiveSGD  # noqa: E402
from src.plots import load_results, plot_results  # noqa: E402

RESULTS = ROOT / "results"


def run_one(cfg, opt_name, batch_size, noise_std, seed, data):
    """Train one config and return a result row scored on the clean test set."""
    Xn_fit, Yn_fit_clean, Xn_val, Yn_val, Xtn, Y_test, sy = data

    # fresh, identically-seeded model so every optimizer starts from the same point
    set_seed(seed)
    model = build_model(cfg.hidden_units)

    # noise is injected into the FIT targets only; val and test stay clean
    Yn_fit = add_label_noise(Yn_fit_clean, noise_std, seed=seed)

    t0 = time.perf_counter()
    if opt_name == "adaptive_sgd":
        trainer = AdaptiveSGD(
            model=model, initial_lr=cfg.initial_lr, grow=cfg.grow,
            shrink=cfg.shrink, lr_min=cfg.lr_min, lr_max=cfg.lr_max,
            patience=cfg.patience,
        )
        history = trainer.train(
            Xn_fit, Yn_fit, epochs=10_000, batch_size=batch_size,
            max_steps=cfg.max_steps, eval_every=cfg.eval_every,
            X_val=Xn_val, Y_val=Yn_val, verbose=False,
        )
    else:
        optimizer = build_keras_optimizer(opt_name, learning_rate=cfg.initial_lr)
        history = train_baseline(
            model, optimizer, Xn_fit, Yn_fit, epochs=10_000, batch_size=batch_size,
            max_steps=cfg.max_steps, eval_every=cfg.eval_every,
            X_val=Xn_val, Y_val=Yn_val, verbose=False,
        )
    seconds = time.perf_counter() - t0

    pred = sy.inverse_transform(model.predict(Xtn, verbose=0))
    metrics = evaluate(Y_test, pred)
    return {
        "optimizer": opt_name,
        "batch_size": "full" if batch_size is None else batch_size,
        "noise_std": noise_std,
        "seed": seed,
        "mse": round(metrics["mse"], 5),
        "matched_root_error": round(metrics["matched_root_error"], 5),
        "n_reverts": history.get("n_reverts", 0),
        "steps": history.get("steps", 0),
        "epochs": round(history.get("epochs", 0.0), 2),
        "final_lr": round(history.get("final_lr", cfg.initial_lr), 8),
        "seconds": round(seconds, 2),
    }


def build_testbed(cfg):
    """Generate + scale the data once and carve the clean held-out val set.

    Scalers are fit on the full training data only (never the test set). The val
    rows are held out of the fit set so model selection scores unseen inputs with
    clean targets -- no label-noise, no test leakage.
    """
    X, Y = generate_polynomial_data(cfg.n_train, cfg.sort_roots, seed=cfg.seed)
    X_test, Y_test = generate_polynomial_data(cfg.n_test, cfg.sort_roots, seed=cfg.seed + 1)
    sx, sy = MinMaxScaler((-1, 1)), MinMaxScaler((-1, 1))
    Xn, Yn_clean, Xtn = sx.fit_transform(X), sy.fit_transform(Y), sx.transform(X_test)

    fit_idx, val_idx = holdout_split(cfg.n_train, cfg.n_val, seed=cfg.seed)
    return (
        Xn[fit_idx], Yn_clean[fit_idx],  # fit inputs / clean fit targets
        Xn[val_idx], Yn_clean[val_idx],  # clean held-out val
        Xtn, Y_test, sy,                 # test (scored in original units)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="fast, tiny sweep")
    parser.add_argument("--seeds", type=str, default=None,
                        help="comma-separated run seeds (default: the config seed)")
    parser.add_argument("--plots-only", action="store_true",
                        help="rebuild figures from the existing CSV, no training")
    args = parser.parse_args()

    cfg = Config()
    if args.smoke:
        cfg.make_smoke()
    prefix = "smoke_" if args.smoke else ""
    out = RESULTS / f"regime_study{'_smoke' if args.smoke else ''}.csv"

    if args.plots_only:
        rows = load_results(out)
        RESULTS.mkdir(exist_ok=True)
        figs = plot_results(rows, RESULTS, prefix=prefix)
        print(f"Rebuilt {len(figs)} figures from {out.relative_to(ROOT)}")
        return

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [cfg.seed]
    set_seed(cfg.seed)
    data = build_testbed(cfg)

    # Write each row as it finishes (flushed) so a long sweep is salvageable if
    # interrupted -- the partial CSV still feeds --plots-only.
    RESULTS.mkdir(exist_ok=True)
    fieldnames = ["optimizer", "batch_size", "noise_std", "seed", "mse",
                  "matched_root_error", "n_reverts", "steps", "epochs",
                  "final_lr", "seconds"]
    rows = []
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()
        for seed in seeds:
            for noise in cfg.noise_levels:
                for bs in cfg.batch_sizes:
                    for opt in cfg.optimizers:
                        row = run_one(cfg, opt, bs, noise, seed, data)
                        rows.append(row)
                        writer.writerow(row)
                        f.flush()
                        print(
                            f"seed={seed} noise={noise:<5} batch={row['batch_size']:<5} "
                            f"{opt:<13} -> matched_root_error={row['matched_root_error']:.4f} "
                            f"(reverts={row['n_reverts']}, {row['seconds']:.1f}s)"
                        )
    print(f"\nWrote {len(rows)} runs to {out.relative_to(ROOT)}")

    figs = plot_results(rows, RESULTS, prefix=prefix)
    for fig in figs:
        print(f"Wrote {fig.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
