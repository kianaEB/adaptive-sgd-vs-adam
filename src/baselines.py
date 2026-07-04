"""Baseline optimizers (Adam, SGD, RMSprop) trained through a matching interface.

Mirrors AdaptiveSGD.train so experiments/regime_study.py can call either path and
get a history dict of the same shape. Kept as a manual step loop (rather than
model.fit) so the compute budget -- max_steps -- means the same thing for every
optimizer in the study.
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
    X_val=None,
    Y_val=None,
    verbose: bool = False,
) -> dict:
    """Train ``model`` with a stock Keras optimizer. Returns a history dict.

    Same signature and return shape as AdaptiveSGD.train (minus n_reverts, which
    is meaningless for these), so the two are drop-in comparable in the study.
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

    def loss_on(X, Y):
        preds = model(X, training=False)
        return float(tf.reduce_mean(tf.keras.losses.MSE(Y, preds)))

    best_loss = float("inf")
    best_weights = model.get_weights()
    steps = 0
    history = {"loss": [], "lr": [], "n_reverts": 0, "steps": 0}

    for epoch in range(1, epochs + 1):
        idx = tf.random.shuffle(tf.range(n))
        for start in range(0, n, bs):
            b = idx[start : start + bs]
            xb, yb = tf.gather(X_train, b), tf.gather(Y_train, b)
            with tf.GradientTape() as tape:
                preds = model(xb, training=True)
                loss = tf.reduce_mean(tf.keras.losses.MSE(yb, preds))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break

        epoch_loss = loss_on(eval_X, eval_Y)
        history["loss"].append(epoch_loss)
        history["lr"].append(float(optimizer.learning_rate.numpy()))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_weights = model.get_weights()
        if verbose:
            print(f"Epoch {epoch:>3} | loss {epoch_loss:.5f} | steps {steps}")
        if max_steps is not None and steps >= max_steps:
            break

    model.set_weights(best_weights)
    history["best_loss"] = best_loss
    history["steps"] = steps
    return history
