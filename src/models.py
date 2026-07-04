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
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential

    model = Sequential(
        [
            Dense(hidden_units, input_dim=6, activation="relu"),
            Dense(hidden_units, activation="relu"),
            Dense(10, activation="linear"),  # 5 roots x (real, imag)
        ]
    )
    return model
