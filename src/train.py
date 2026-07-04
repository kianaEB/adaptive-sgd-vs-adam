"""Single-run training entry point (train AdaptiveSGD once on the testbed).

For the actual study -- the sweep over batch size and noise -- see
experiments/regime_study.py. This file is just a convenient single run.

Examples
--------
    python -m src.train --smoke-test
    python -m src.train --batch-size 512 --noise 0.1 --max-steps 4000
"""

from __future__ import annotations

import argparse

from sklearn.preprocessing import MinMaxScaler

from .config import Config, set_seed
from .data import add_label_noise, generate_polynomial_data
from .evaluate import evaluate
from .models import build_model
from .optimizers import AdaptiveSGD


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the model once with AdaptiveSGD.")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="mini-batch size (omit for a small default)")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="std of Gaussian label noise on training targets")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true",
                        help="tiny/fast run to check the pipeline end to end")
    args = parser.parse_args()

    cfg = Config()
    n_train, n_test = cfg.n_train, cfg.n_test
    max_steps = args.max_steps if args.max_steps is not None else cfg.max_steps
    if args.smoke_test:
        n_train, n_test, max_steps = 2000, 500, 200

    set_seed(cfg.seed)

    X, Y = generate_polynomial_data(n_train, sort_roots=cfg.sort_roots, seed=cfg.seed)
    X_test, Y_test = generate_polynomial_data(n_test, sort_roots=cfg.sort_roots,
                                              seed=cfg.seed + 1)
    sx, sy = MinMaxScaler((-1, 1)), MinMaxScaler((-1, 1))
    Xn, Yn, Xtn = sx.fit_transform(X), sy.fit_transform(Y), sx.transform(X_test)
    Yn = add_label_noise(Yn, args.noise, seed=cfg.seed)

    model = build_model(cfg.hidden_units)
    trainer = AdaptiveSGD(
        model=model, initial_lr=cfg.initial_lr, grow=cfg.grow, shrink=cfg.shrink,
        lr_min=cfg.lr_min, lr_max=cfg.lr_max, patience=cfg.patience,
    )
    history = trainer.train(Xn, Yn, epochs=10_000, batch_size=args.batch_size,
                            max_steps=max_steps)

    pred = sy.inverse_transform(model.predict(Xtn, verbose=0))
    metrics = evaluate(Y_test, pred)
    print("\nTest metrics (original units):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.5f}")
    print(f"  reverts during training: {history['n_reverts']}")


if __name__ == "__main__":
    main()
