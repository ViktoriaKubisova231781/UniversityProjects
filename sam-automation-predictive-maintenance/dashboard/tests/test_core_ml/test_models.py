import numpy as np
from core_ml.models import get_autoencoder


def test_autoencoder_shapes():
    """Verify input/output shapes match (Autoencoder property)."""
    # 30 time steps, 1 feature
    input_shape = (30, 1)

    model = get_autoencoder(input_shape=input_shape)

    # Check Keras model properties
    assert model.input_shape == (None, 30, 1)
    assert model.output_shape == (None, 30, 1)

    # Check that it compiles
    assert model.optimizer is not None
    assert model.loss == 'mse'


def test_autoencoder_flexible_input():
    """Verify it handles batch dimension tuple variations."""
    # Sometimes shape is passed as (Batch, Steps, Features)
    input_data = np.zeros((10, 20, 5))  # 20 steps, 5 features

    model = get_autoencoder(input_data=input_data)

    # Output should have 5 features (reconstruction)
    assert model.output_shape == (None, 20, 5)
