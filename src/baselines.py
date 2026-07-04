"""Baseline optimizers (Adam, SGD, RMSprop) trained through a matching interface.

Mirrors AdaptiveSGD.train so experiments/regime_study.py can call either path and
get a history dict of the same shape. Kept as a manual step loop (rather than
model.fit) so the compute budget -- max_steps -- means the same thing for every
optimizer in the study.

Model selection is deliberately identical to AdaptiveSGD's: on the same fixed
step cadence (`eval_every`), evaluate a clean held-out validation set and keep the
best-scoring weights. That way the only thing separating the optimizers is their
update rule, not how their final model was chosen.
"""

from __future__ import annotations


def build_keras_optimizer(name: str, learning_rate: float = 1e-3):
    """Map a short name to a stock Keras optimizer."""
    import tensorflow as tf

    name = name.lower()
    table = {
        "adam": tf.keras.optimizers.Adam,
        "sgd": tf.keras.optimizers.SGD,
        "rmsprop": tf.keras.optimizers.RMSprop,
    }
    if name not in table:
        raise ValueError(f"Unknown baseline optimizer: {name!r}")
    return table[name](learning_rate=learning_rate)


def train_baseline(
    model,
    optimizer,
    X_train,
    Y_train,
    epochs: int = 50,
    batch_size: int | None = 256,
    max_steps: int | None = None,
    eval_every: int = 10,
    X_val=None,
    Y_val=None,
    verbose: bool = False,
) -> dict:
    """Train ``model`` with a stock Keras optimizer. Returns a history dict.

    Same signature and return shape as AdaptiveSGD.train (n_reverts stays 0, since
    these never revert), so the two are drop-in comparable in the study. The
    validation-based best-weight selection runs on the identical `eval_every`
    cadence AdaptiveSGD uses, keeping model selection fair across optimizers.
    """
    import tensorflow as tf

    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
    n = int(X_train.shape[0])
    bs = n if batch_size is None else batch_size

    has_val = X_val is not None and Y_val is not None
    sel_X = tf.convert_to_tensor(X_val if has_val else X_train, dtype=tf.float32)
    sel_Y = tf.convert_to_tensor(Y_val if has_val else Y_train, dtype=tf.float32)

    def select_loss() -> float:
        preds = model(sel_X, training=False)
        return float(tf.reduce_mean(tf.keras.losses.MSE(sel_Y, preds)))

    best_val_loss = float("inf")
    best_val_weights = model.get_weights()
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
            preds = model(xb, training=True)
            loss = tf.reduce_mean(tf.keras.losses.MSE(yb, preds))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        steps += 1

        last_step = max_steps is not None and steps >= max_steps
        if steps % eval_every != 0 and not last_step:
            continue

        s_loss = select_loss()
        history["loss"].append(s_loss)
        history["lr"].append(float(optimizer.learning_rate.numpy()))
        if s_loss < best_val_loss:
            best_val_loss = s_loss
            best_val_weights = model.get_weights()
        if verbose:
            print(f"step {steps:>5} | val {s_loss:.5f}")

    model.set_weights(best_val_weights)
    history["best_loss"] = best_val_loss
    history["best_val_loss"] = best_val_loss
    history["final_lr"] = float(optimizer.learning_rate.numpy())
    history["steps"] = steps
    history["epochs"] = steps * bs / n
    return history
