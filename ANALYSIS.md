# Analysis — where AdaptiveSGD breaks, and why

Grounded in `results/regime_study.csv` (64 runs: 4 batch sizes × 4 noise levels ×
4 optimizers, single seed, a fixed budget of 4000 gradient updates each). The
metric is `matched_root_error` — predicted roots matched to true roots by optimal
(Hungarian) assignment before measuring distance, scored on a clean held-out test
set (see [`src/evaluate.py`](src/evaluate.py)).

## The one-line answer
> **Batch size is the binding constraint; label noise is almost irrelevant within
> the range tested.** AdaptiveSGD is competitive at full-batch and collapses the
> instant you mini-batch — a cliff, not a slope. The collapse is fully explained by
> two diagnostics moving together: the revert count and the learning rate.

## The headline (noise σ = 0; the other three noise levels are within ~1%)

| batch | passes over data | AdaptiveSGD reverts / 400 | AdaptiveSGD final LR | AdaptiveSGD error | Adam error |
|------:|-----------------:|--------------------------:|:--------------------:|------------------:|-----------:|
| full  | 4000 | **16**  | **1.0** (ceiling) | 0.174 | 0.105 |
| 4096  |  252 | 376 | 1e-6 (floor) | 0.438 | 0.110 |
| 512   |   32 | 374 | 1e-6 (floor) | 0.418 | 0.122 |
| 64    |    4 | 395 | 1e-6 (floor) | **2.244** | 0.150 |

Adam is the best optimizer in **all 16 regimes**; RMSprop tracks it (0.15–0.19);
SGD is stuck at 0.854 everywhere (a fixed lr=1e-3 is simply too small for plain
SGD on this task). So AdaptiveSGD never strictly *wins* — the story is about how
far it falls and why.

## The mechanism (the "why" ↔ "what")

**1. The control signal's noise scales with 1/batch.** AdaptiveSGD accepts or
reverts based on the loss of a *fresh mini-batch of size `batch_size`*, evaluated
every 10 steps (a fixed cadence, so every run gets exactly 400 decisions
regardless of batch size — see "Design decisions" below). At full-batch that
"mini-batch" is the entire fit set, so the signal is rock-steady. Shrink the batch
and the same signal starts jittering purely from sampling.

**2. The revert count cliffs, it does not climb.** The hypothesis expected reverts
to rise *gradually* with difficulty. They don't: they jump from **16 → ~375 the
moment the batch stops being full**, then creep up to ~395 at batch 64. Once the
signal is a subsample at all, a single unlucky evaluation trips the accept/revert
rule on ~95% of decisions. This is visible as the sharp full→mini transition in
the right panel of `results/regime_heatmap.png`.

**3. The learning rate is the smoking gun.** `final_lr` pins to opposite rails:

- Full-batch → LR climbs to the **ceiling (lr_max = 1.0)**. Progress is trusted, so
  the grow rule fires almost every step. AdaptiveSGD behaves like a well-tuned,
  aggressive optimizer and lands a competitive 0.174.
- Any mini-batch → LR collapses to the **floor (lr_min = 1e-6)**. Constant reverts
  mean the shrink rule dominates; within a few dozen decisions the LR is 1e-6 —
  a thousandth of SGD's 1e-3 — *and* most of the steps it does take are rolled
  back. The network barely leaves initialization.

This is why batch-64 AdaptiveSGD (2.244) is worse than even stuck SGD (0.854):
SGD at least takes 4000 un-reverted steps, while AdaptiveSGD reverts 395/400 and
starves its own learning rate, so the model it keeps is essentially untrained.

**4. Why noise barely matters here.** Across σ = 0 → 0.20, AdaptiveSGD's full-batch
error moves only 0.174 → 0.177, and every row of the gap heatmap is nearly
identical. Sampling noise from a small batch swamps the label noise we injected:
by the time σ=0.20 adds a little jitter, the batch-64 signal is already so noisy
that the reverts have saturated. Label noise would likely dominate again at much
larger batches or much higher σ — but within this design, **batch size is the
lever.**

## Reading the two figures
- **`results/crossover.png`** — matched error vs batch size, one panel per noise
  level. Adam and RMSprop are flat and low across the whole x-axis; AdaptiveSGD
  (blue) leaves them at full-batch and rockets up as the batch shrinks, even
  crossing *above* naive SGD between 512 and 64. The four panels are near-copies —
  the visual proof that noise is not the driver.
- **`results/regime_heatmap.png`** — left: the AdaptiveSGD−Adam gap, white at
  full-batch (+0.06, a near-tie) deepening to dark red (+2.1) at batch 64; right:
  the revert count, pale (14–16) at full-batch and saturated (373–395) everywhere
  else. The two panels line up cell-for-cell: the gap *is* the revert cliff.

## Threats to validity (read the result with these in mind)
- **One learning rate per optimizer.** SGD's poor showing is partly a too-small LR;
  a fairer study sweeps LR per optimizer. AdaptiveSGD's own `initial_lr`, `grow`,
  `shrink`, and especially `patience=1` (revert on the *first* down-tick) make it
  trigger-happy; a larger patience or a smoothed control loss would soften the
  cliff. The finding is about *this* configuration, honestly reported.
- **The control-signal design is a choice, not a given.** SPEC §6 left the
  backtracking cadence open; we evaluate every 10 steps on a batch-sized loss (see
  below). A full-training-set control loss would remove the batch-size mechanism
  entirely — and with it, most of this result. That decision is the study.
- **Single seed, one model, one task.** Variance is unmeasured (the `--seeds` flag
  exists to add it); degree-5 polynomial-root regression is a convenient testbed,
  not a claim about all regression.

## Design decisions that make the comparison fair
- **Fixed compute budget** — every optimizer gets exactly 4000 gradient *updates*
  (not epochs), so batch size doesn't smuggle in extra updates.
- **Fixed decision cadence** — 400 accept/revert (and model-selection) checks per
  run regardless of batch size, resolving the SPEC §6 confound (otherwise batch 64
  would get only ~4 whole-pass decisions and its revert count could never move).
- **Batch-sized control loss** — so the decision signal's noise is a function of
  batch size, which is the mechanism under test.
- **Clean held-out validation** — 5000 rows carved out of training (test set never
  touched); the *same* val-best machinery selects the final weights for every
  optimizer, so only the update rule differs.
- **Permutation-aware metric** — roots are an unordered set; Hungarian matching
  scores them fairly (a plain MSE would punish the right roots in the wrong order).

## Reproduce
```bash
python experiments/regime_study.py            # full sweep -> results/ (CSV + figures)
python experiments/regime_study.py --plots-only   # rebuild figures from the CSV
```
