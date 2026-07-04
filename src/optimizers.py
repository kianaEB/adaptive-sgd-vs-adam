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
trustworthy loss signal. With small batches or noisy labels the per-epoch loss
becomes noisy, so AdaptiveSGD reverts needlessly and stalls -- exactly where
Adam's smoothed moment estimates are expected to pull ahead. `n_reverts` in the
returned history is the diagnostic that lets you see this happening.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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

    def _loss(self, X, Y):
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
        X_val=None,
        Y_val=None,
        verbose: bool = True,
    ) -> dict:
        """Train ``self.model`` in place. Returns a history dict for plotting.

        Parameters
        ----------
        batch_size : int or None
            None means full-batch.
        max_steps : int or None
            If set, cap the total number of gradient updates. Fixing this across
            configs isolates the batch-size variable (see SPEC.md section 6).
        """
        import tensorflow as tf

        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
        n = int(X_train.shape[0])
        bs = n if batch_size is None else batch_size

        eval_X = X_val if X_val is not None else X_train
        eval_Y = Y_val if Y_val is not None else Y_train
        eval_X = tf.convert_to_tensor(eval_X, dtype=tf.float32)
        eval_Y = tf.convert_to_tensor(eval_Y, dtype=tf.float32)

        best_loss = float("inf")
        best_weights = self.model.get_weights()
        fails = 0
        steps = 0
        history = {"loss": [], "lr": [], "n_reverts": 0, "steps": 0}

        for epoch in range(1, epochs + 1):
            idx = tf.random.shuffle(tf.range(n))
            for start in range(0, n, bs):
                b = idx[start : start + bs]
                xb, yb = tf.gather(X_train, b), tf.gather(Y_train, b)
                with tf.GradientTape() as tape:
                    preds = self.model(xb, training=True)
                    loss = tf.reduce_mean(tf.keras.losses.MSE(yb, preds))
                grads = tape.gradient(loss, self.model.trainable_variables)
                for var, g in zip(self.model.trainable_variables, grads):
                    if g is not None:
                        var.assign_sub(self.lr * g)
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    break

            epoch_loss = self._loss(eval_X, eval_Y)
            history["loss"].append(epoch_loss)
            history["lr"].append(self.lr)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_weights = self.model.get_weights()
                fails = 0
                self.lr = min(self.lr * self.grow, self.lr_max)
                status = "improved"
            else:
                fails += 1
                if fails >= self.patience:
                    self.model.set_weights(best_weights)
                    self.lr = max(self.lr * self.shrink, self.lr_min)
                    fails = 0
                    history["n_reverts"] += 1
                    status = "reverted"
                else:
                    status = "no-improve"

            if verbose:
                print(
                    f"Epoch {epoch:>3} | loss {epoch_loss:.5f} "
                    f"| lr {self.lr:.2e} | steps {steps} | {status}"
                )
            if max_steps is not None and steps >= max_steps:
                break

        self.model.set_weights(best_weights)  # leave model at its best weights
        history["best_loss"] = best_loss
        history["steps"] = steps
        return history
