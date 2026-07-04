# SPEC.md — Adaptive Backtracking vs. Adam: a controlled study

## 1. Background & origin
This project began as a single Colab notebook (`reference/AdaptiveSGD_original.ipynb`)
that trained a small network to predict degree-5 polynomial roots with a custom
Levenberg-Marquardt-flavored loop: grow the learning rate when the loss falls,
revert the weights and shrink it when the loss rises. That notebook is preserved
unchanged as the historical starting point.

The project is **not** a claim that this optimizer beats Adam. It is a controlled
study of *when each approach wins*.

## 2. Research question
> **At what batch size and label-noise level does Adam's advantage over adaptive
> backtracking (AdaptiveSGD) appear — and why?**

Polynomial-root regression is just the **testbed**: a convenient task with a
smooth-ish loss and a controllable difficulty knob (label noise). The deliverable
is the study, not a root solver.

## 3. Hypothesis
AdaptiveSGD's accept/revert rule needs a *trustworthy* loss signal. Full-batch,
low-noise training gives it one, so it should be competitive there. As batches
shrink or label noise rises, the measured loss gets noisy: a single unlucky
evaluation triggers a needless revert and a learning-rate cut, so progress
stalls. Adam sidesteps this by smoothing gradients through moving averages, so
its advantage should **grow as batch size shrinks and noise rises**. The
`n_reverts` diagnostic should climb in exactly the regimes where AdaptiveSGD
loses — connecting the "why" to the "what".

This hypothesis is falsifiable and might be wrong (e.g. AdaptiveSGD may lose even
full-batch, or hold up better than expected). Report whatever the sweep shows.

## 4. Goals
- G1. A reusable AdaptiveSGD training strategy (LR clamp, mini-batches, patience,
  best-weight restore, revert counter). — done in `src/optimizers.py`.
- G2. A fair, matched-interface baseline path for Adam / SGD / RMSprop so every
  optimizer sees the same compute budget. — done in `src/baselines.py`.
- G3. A controlled sweep over {batch size} x {noise level} x {optimizer}, scored
  on one clean test set with a permutation-aware metric. — `experiments/regime_study.py`.
- G4. Two figures that tell the story (crossover curve + regime heatmap) and a
  written explanation grounded in the revert-count diagnostic.
- G5. Full reproducibility: seeds, `requirements.txt`, one-command sweep.

## 5. Non-goals
- Not beating classical root-finders (this is an optimizer study, not numerics).
- Not claiming a novel optimizer — AdaptiveSGD is a known family, benchmarked honestly.
- Not hyperparameter-tuning each optimizer to death; hold a fair, fixed budget.

## 6. Experimental design
**Held fixed:** model architecture (6->64->64->10), dataset + seed, and the
compute budget.

**Varied (the independent variables):**
- `batch_size`: full-batch -> tiny (`None, 4096, 512, 64`).
- `noise_std`: Gaussian noise added to *normalized training targets only*
  (`0.0, 0.05, 0.10, 0.20`). Test targets stay clean.
- `optimizer`: `adaptive_sgd, adam, sgd, rmsprop`.

**Compute budget — a real design decision.** Batch size changes how many updates
an epoch contains, so fixing *epochs* would confound batch size with update
count. We instead fix the total number of gradient **updates** (`max_steps`), so
every configuration gets the same number of parameter updates. Log epochs and
wall-clock too, and state this choice explicitly in the writeup.

**Cadence subtlety to decide in M2/M3:** AdaptiveSGD currently makes its
accept/revert decision once per full pass. With tiny batches a "pass" is many
steps, so the control cadence varies with batch size. Consider switching to a
fixed step cadence (evaluate every N steps) so the backtracking granularity is
independent of batch size. Note whichever you choose.

## 7. Metrics
Scored on the clean test set, reported in original units:
- `matched_root_error` — predicted roots matched to true roots by optimal
  assignment (Hungarian) before measuring distance. The fair primary metric.
- `mse` — order-sensitive; reported for context / as a foil.
- `n_reverts` — AdaptiveSGD diagnostic; the mechanistic link for the "why".

## 8. Module layout
`data.py` (testbed + noise knob) -> `models.py` -> {`optimizers.py`,
`baselines.py`} -> `evaluate.py` -> `experiments/regime_study.py`. See CLAUDE.md.

## 9. Roadmap (one milestone at a time; commit after each)
- **M0 — Scaffold.** Structure, requirements, seeds, notebook in `reference/`. (done)
- **M1 — Testbed + metric.** Data, label-noise knob, permutation-aware metric, tests. (done)
- **M2 — Optimizers.** AdaptiveSGD strategy + baseline path, matched interface, step budget. (done)
- **M3 — Run the sweep.** Execute `regime_study.py`, save `results/regime_study.csv`.
- **M4 — Plots.** Crossover curve + regime heatmap into `results/` (add `src/plots.py`).
- **M5 — Analysis.** Read the results; write the "why", grounded in `n_reverts`.
- **M6 — README.** Write the narrative around the real figures and numbers.

## 10. Definition of "portfolio-ready"
- A reader understands the question, the design, and the finding in ~3 minutes.
- A crossover plot shows where Adam overtakes as batch/noise change.
- The revert-count diagnostic mechanistically explains the boundary.
- Runs from a clean clone with one command; tests pass; result reported honestly
  whichever way it falls.

## 11. Threats to validity (address in the writeup)
- Each optimizer uses one learning rate; a fairer study sweeps LR per optimizer.
- One model / one task — findings may not generalize; say so.
- Fixed-steps vs fixed-epochs changes the story; be explicit about the budget.
- Single seed per config hides variance; repeat with several seeds if time allows.
