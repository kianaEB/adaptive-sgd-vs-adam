# CLAUDE.md

Operational guide for working in this repository. Read this before making changes.
For *what* we're building and *why*, see `SPEC.md`. For the original prototype this
grew from, see `reference/AdaptiveSGD_original.ipynb` (frozen reference — do not edit).

## Project in one line
A controlled study of when adaptive learning-rate backtracking (AdaptiveSGD) holds
up versus when per-parameter moment estimates (Adam) win — as a function of batch
size and label noise. Polynomial-root regression is only the testbed.

## Tech stack
- Python 3.11+, TensorFlow 2.x (Keras), NumPy, SciPy, scikit-learn, Matplotlib
- pytest for tests

## Repo structure
- `src/data.py` — testbed: polynomial generation, sorted targets, label-noise knob
- `src/models.py` — the fixed model
- `src/optimizers.py` — AdaptiveSGD training strategy (owns the loop; see SPEC §6)
- `src/baselines.py` — Adam/SGD/RMSprop through a matching interface + step budget
- `src/evaluate.py` — permutation-aware metric (Hungarian matching) + MSE
- `src/config.py` — the sweep grids and compute budget
- `experiments/regime_study.py` — THE STUDY: the batch x noise x optimizer sweep
- `tests/` — unit tests (data + metrics; run without TensorFlow)
- `results/` — generated CSV + plots (git-ignored except final chosen figures)
- `reference/` — frozen original notebook. READ-ONLY.

## Commands
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest -q                                    # tests
python -m src.train --smoke-test             # single-run pipeline check
python experiments/regime_study.py --smoke   # fast sweep
python experiments/regime_study.py           # full sweep -> results/regime_study.csv
```

## Coding conventions
- Small, pure, testable functions. I/O and plotting live in scripts, not core logic.
- Type hints + docstrings (explain the *why*). Thread the seed everywhere.
- No magic numbers — hyperparameters and grids live in `src/config.py`.
- TF imported lazily inside functions so data/metric/tests run without TF installed.

## Definition of done (per change)
1. New logic has a test in `tests/` and `pytest` passes.
2. New experiment output is reproducible and written to `results/`.
3. Public functions typed and documented. No new deprecation warnings.

## Guardrails — do NOT
- Do not edit anything in `reference/`.
- Do not commit generated data, `.venv/`, checkpoints, or large artifacts.
- Do not let AdaptiveSGD's learning rate grow unbounded — keep the [lr_min, lr_max] clamp.
- Do not give optimizers different compute budgets — the comparison must be fair (fixed `max_steps`).
- Do not pre-write the conclusion. Run the sweep; report whatever it shows, even if Adam wins everywhere.
- Do not leak the test set into scaler fitting, noise injection, or tuning.

## How to work with me on this repo
- For non-trivial tasks, read `SPEC.md` + the relevant `src/` files, run the tests
  and the smoke sweep, then propose a short plan and wait before writing code.
- Work one milestone at a time (SPEC §9). Commit after each. Next up: M3 (run the
  sweep) then M4 (plots) then M5 (analysis).
