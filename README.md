# Adaptive Backtracking vs. Adam

> A controlled study: **when does adaptive learning-rate backtracking hold up, and
> when do per-parameter moment estimates (Adam) win** — as a function of batch size
> and label noise?

<!-- TODO: write this README LAST, once results/ has real numbers and the two
     figures. A reader should get the question, the design, and the finding in
     ~3 minutes. Keep the arc below. -->

## The question
A custom optimizer, **AdaptiveSGD**, grows the learning rate when the loss falls
and reverts the weights when it rises. That rule needs a *trustworthy* loss
signal. So: at what batch size and noise level does that assumption break, and
where does Adam's smoothed-gradient approach take over — and why?

## Hypothesis
AdaptiveSGD should be competitive with full batches and clean labels, and lose
ground as batches shrink or noise rises, because a noisy loss estimate triggers
needless reverts. Adam should be robust to exactly that. The **revert count**
should climb precisely where AdaptiveSGD loses.
<!-- The point of the project is to TEST this, not assert it. Report what happens. -->

## What's here
- **AdaptiveSGD** — adaptive-LR backtracking as a training strategy, with LR
  clamping, mini-batches, patience, and a revert counter (`src/optimizers.py`).
- **Matched baselines** — Adam / SGD / RMSprop trained under the same compute
  budget so the comparison is fair (`src/baselines.py`).
- **A permutation-aware metric** — matches predicted to true roots by optimal
  assignment before scoring (`src/evaluate.py`).
- **The study** — a sweep over batch size x noise x optimizer (`experiments/regime_study.py`).

## The finding
<!-- TODO: crossover plot + regime heatmap here, plus the one-paragraph "why"
     grounded in the revert-count diagnostic. State the result honestly even if
     Adam wins everywhere or AdaptiveSGD holds up better than expected. -->

| Regime | Winner | Why |
|--------|--------|-----|
| Full-batch, no noise | _TODO_ | _TODO_ |
| Small batch or high noise | _TODO_ | _TODO_ |

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest -q                              # tests (no GPU needed)
python -m src.train --smoke-test       # single-run pipeline check
python experiments/regime_study.py --smoke   # fast sweep
python experiments/regime_study.py     # full sweep -> results/regime_study.csv
```

## Design notes
See [`SPEC.md`](SPEC.md) for the full design, including the compute-budget
decision (fix gradient *steps*, not epochs, to isolate batch size) and the
threats to validity. The project grew from a Colab notebook preserved in
[`reference/`](reference/AdaptiveSGD_original.ipynb).

## Honest framing
AdaptiveSGD's grow/shrink rule is in the family of Levenberg-Marquardt damping
and RPROP-style adaptive rates. This project implements and benchmarks that idea;
it does not claim a new algorithm. The value is the controlled comparison and the
mechanistic explanation, not a leaderboard win.
