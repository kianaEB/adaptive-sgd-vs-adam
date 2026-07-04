"""AdaptiveSGD: an adaptive learning-rate training strategy with backtracking.

The custom method under study. It generalizes the loop from the original notebook.

Design note (worth explaining in the README):
"Revert *all* weights when the global loss rises" is a whole-model decision based
on a global loss value, which does not fit a per-variable
`tf.keras.optimizers.Optimizer.update_step`. So AdaptiveSGD is a *training
strategy* that owns the loop. A lighter variant is a Keras Callback that adjusts
the LR plus `restore_best_weights`.

Prior work (state honestly): the grow/shrink-on-progress rule is in the family of
Levenberg-Marquardt damping adjustment and RPROP-style adaptive rates. This
project implements and *benchmarks* that idea rather than claiming novelty.

The central hypothesis of the study: the accept/revert decision needs a
trustworthy loss signal. With small batches or noisy labels the *measured* loss
becomes noisy, so AdaptiveSGD reverts needlessly and stalls -- exactly where
Adam's smoothed moment estimates are expected to pull ahead. `n_reverts` in the
returned history is the diagnostic that lets you see this happening.

Cadence (SPEC section 6 decision): the accept/revert decision is made on a FIXED
STEP cadence (`eval_every` gradient updates) rather than once per epoch, so the
backtracking granularity is identical across batch sizes -- otherwise a run with
tiny batches gets only a handful of decisions inside the step budget and its
revert count cannot climb. The decision (control) loss is measured on a fresh
mini-batch of size `batch_size`, so its variance scales with batch size and label
noise -- the very mechanism the study probes. Which weights we ultimately KEEP is
a separate, clean choice: the best weights on a held-out clean validation set,
selected by the identical machinery used for every baseline optimizer (see
`src/baselines.py`), so the comparison of final models stays fair.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BacktrackDecision:
    """Outcome of one accept/revert decision (pure data, no TensorFlow)."""

    action: str  # "improved" | "no-improve" | "reverted"
    lr: float  # new, clamped learning rate
    best_loss: float  # new running-best control loss
    fails: int  # new consecutive-non-improvement counter
    snapshot_best: bool  # caller should snapshot current weights as control-best
    restore_best: bool  # caller should restore the control-best weights


def apply_backtracking_decision(
    control_loss: float,
    best_loss: float,
    fails: int,
    lr: float,
    *,
    grow: float,
    shrink: float,
    lr_min: float,
    lr_max: float,
    patience: int,
) -> BacktrackDecision:
    """Pure accept / tolerate / revert policy for AdaptiveSGD.

    Given the latest control-loss reading and the current control state, decide
    whether to (a) accept progress -- grow the LR and snapshot the weights, (b)
    tolerate a non-improvement while within ``patience``, or (c) revert -- restore
    the best weights and shrink the LR. The learning rate is always clamped to
    ``[lr_min, lr_max]`` (a guardrail: AdaptiveSGD's LR must never run away).

    Extracted from the training loop so the decision logic is unit-testable
    without a TensorFlow install; the caller performs the weight side-effects the
    returned flags ask for.
    """
    if control_loss < best_loss:
        return BacktrackDecision(
            action="improved",
            lr=min(lr * grow, lr_max),
            best_loss=control_loss,
            fails=0,
            snapshot_best=True,
            restore_best=False,
        )
    fails += 1
    if fails >= patience:
        return BacktrackDecision(
            action="reverted",
            lr=max(lr * shrink, lr_min),
            best_loss=best_loss,
            fails=0,
            snapshot_best=False,
            restore_best=True,
        )
    return BacktrackDecision(
        action="no-improve",
        lr=lr,
        best_loss=best_loss,
        fails=fails,
        snapshot_best=False,
        restore_best=False,
    )


@dataclass
class AdaptiveSGD:
    model: object
    initial_lr: float = 1e-3
    grow: float = 1.5
    shrink: float = 0.4
    lr_min: float = 1e-6
    lr_max: float = 1.0
    patience: int = 1

    lr: float = field(init=False)

    def __post_init__(self) -> None:
        self.lr = self.initial_lr

    def _loss(self, X, Y) -> float:
        import tensorflow as tf

        preds = self.model(X, training=False)
        return float(tf.reduce_mean(tf.keras.losses.MSE(Y, preds)))

    def train(
        self,
        X_train,
        Y_train,
        epochs: int = 50,
        batch_size: int | None = 256,
        max_steps: int | None = None,
        eval_every: int = 10,
        X_val=None,
        Y_val=None,
        verbose: bool = True,
    ) -> dict:
        """Train ``self.model`` in place. Returns a history dict for plotting.

        Parameters
        ----------
        batch_size : int or None
            None means full-batch. Also the size of the random sample used to
            measure the accept/revert control loss, so that signal's noise scales
            with batch size (full-batch -> the whole fit set -> a stable signal).
        max_steps : int or None
            If set, cap the total number of gradient updates. Fixing this across
            configs isolates the batch-size variable (see SPEC.md section 6).
        eval_every : int
            Make an accept/revert decision every this many gradient updates. A
            fixed step cadence keeps the backtracking granularity independent of
            batch size.
        X_val, Y_val : array or None
            Clean held-out set for choosing the weights we keep. When omitted,
            model selection falls back to the full training set.
        """
        import tensorflow as tf

        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
        n = int(X_train.shape[0])
        bs = n if batch_size is None else batch_size

        has_val = X_val is not None and Y_val is not None
        sel_X = tf.convert_to_tensor(X_val if has_val else X_train, dtype=tf.float32)
        sel_Y = tf.convert_to_tensor(Y_val if has_val else Y_train, dtype=tf.float32)

        def control_loss() -> float:
            # fresh random draw of `bs` rows -> variance grows as batch shrinks
            b = tf.random.shuffle(tf.range(n))[:bs]
            return self._loss(tf.gather(X_train, b), tf.gather(Y_train, b))

        # control state: drives accept/revert on the noisy batch signal
        control_best_loss = float("inf")
        control_best_weights = self.model.get_weights()
        fails = 0

        # selection state: the clean-val best is the model we actually keep
        best_val_loss = float("inf")
        best_val_weights = self.model.get_weights()

        steps = 0
        history = {"loss": [], "lr": [], "n_reverts": 0, "steps": 0}

        idx = tf.random.shuffle(tf.range(n))
        pos = 0
        epochs_done = 0
        while True:
            if max_steps is not None and steps >= max_steps:
                break
            if max_steps is None and epochs_done >= epochs:
                break
            if pos >= n:
                idx = tf.random.shuffle(tf.range(n))
                pos = 0
                epochs_done += 1
                continue

            b = idx[pos : pos + bs]
            pos += bs
            xb, yb = tf.gather(X_train, b), tf.gather(Y_train, b)
            with tf.GradientTape() as tape:
                preds = self.model(xb, training=True)
                loss = tf.reduce_mean(tf.keras.losses.MSE(yb, preds))
            grads = tape.gradient(loss, self.model.trainable_variables)
            for var, g in zip(self.model.trainable_variables, grads):
                if g is not None:
                    var.assign_sub(self.lr * g)
            steps += 1

            last_step = max_steps is not None and steps >= max_steps
            if steps % eval_every != 0 and not last_step:
                continue

            # ---- accept/revert decision on the noisy control signal ----------
            c_loss = control_loss()
            decision = apply_backtracking_decision(
                c_loss, control_best_loss, fails, self.lr,
                grow=self.grow, shrink=self.shrink,
                lr_min=self.lr_min, lr_max=self.lr_max, patience=self.patience,
            )
            self.lr = decision.lr
            control_best_loss = decision.best_loss
            fails = decision.fails
            if decision.snapshot_best:
                control_best_weights = self.model.get_weights()
            elif decision.restore_best:
                self.model.set_weights(control_best_weights)
                history["n_reverts"] += 1

            history["loss"].append(c_loss)
            history["lr"].append(self.lr)

            # ---- clean, fair model selection (identical for baselines) -------
            s_loss = self._loss(sel_X, sel_Y)
            if s_loss < best_val_loss:
                best_val_loss = s_loss
                best_val_weights = self.model.get_weights()

            if verbose:
                print(
                    f"step {steps:>5} | ctrl {c_loss:.5f} | val {s_loss:.5f} "
                    f"| lr {self.lr:.2e} | {decision.action}"
                )

        self.model.set_weights(best_val_weights)  # keep the clean-val best
        history["best_loss"] = control_best_loss
        history["best_val_loss"] = best_val_loss
        history["final_lr"] = self.lr
        history["steps"] = steps
        history["epochs"] = steps * bs / n
        return history
