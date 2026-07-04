"""Model definitions.

Kept intentionally small — matches the original notebook's architecture so the
optimizer comparison is the focus, not the model. Swap or grow this later.
"""

from __future__ import annotations


def build_model(hidden_units: int = 64):
    """Build the 6 -> h -> h -> 10 regression model from the original notebook.

    Imported lazily so that non-TF code (data, metrics, tests) runs without a
    TensorFlow install.
    """
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Sequential

    # Explicit Input layer (Keras 3 idiom): passing input_dim= to the first Dense
    # is deprecated and warns under the Keras 3 that ships with TF 2.16+.
    model = Sequential(
        [
            Input(shape=(6,)),
            Dense(hidden_units, activation="relu"),
            Dense(hidden_units, activation="relu"),
            Dense(10, activation="linear"),  # 5 roots x (real, imag)
        ]
    )
    return model
