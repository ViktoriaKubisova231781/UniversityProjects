import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import MagicMock, patch
from sklearn.preprocessing import StandardScaler
from core_ml.anomaly_detection import AnomalyDetector

# --- HELPERS ---


class FakeTensor:
    """Simulates a TensorFlow tensor with a .numpy() method."""
    def __init__(self, array):
        self._array = array

    def numpy(self):
        return self._array


class FakeKerasModel:
    """
    Simulates a Keras model.
    Dynamically returns an output array of the same shape as the input.
    """
    def __call__(self, x, training=False):
        # x shape: (Batch_Size, Seq_Len, Features)
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Create a real numpy array of zeros with the expected shape (Batch, Seq, 1)
        # This guarantees y_pred matches x_input dimensionally
        result = np.zeros((batch_size, seq_len, 1))

        return FakeTensor(result)


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='s'),
        'temp': np.random.rand(100),
        'pressure': np.random.rand(100)
    })

# --- TESTS ---


@patch("core_ml.anomaly_detection.get_autoencoder")
def test_training_loop(mock_get_model, sample_data):
    """Test that training iterates over columns and saves models."""
    mock_keras_model = MagicMock()
    # Mock predict output (Batch, Seq, Feat)
    mock_keras_model.return_value.numpy.return_value = np.zeros((10, 30, 1))
    mock_get_model.return_value = mock_keras_model

    detector = AnomalyDetector(
        data_frame=sample_data,
        columns=['temp'],
        sequence_length=30,
        training_epochs=1
    )

    assert 'temp' in detector.model_dict
    mock_keras_model.fit.assert_called_once()


def test_inference_live_vs_batch(sample_data):
    """
    Test the difference between Live Mode (scalar)
    and History Mode (array).
    """
    detector = AnomalyDetector()
    detector.columns = ['temp']
    detector.sequence_length = 10
    detector.diff_cols = []
    detector.transform_output = False  # Ensure no inverse transform logic runs

    # 1. Mock Norm: Must return numpy array (not DataFrame)
    # The pipeline code expects to slice a numpy array.
    # If we return a DataFrame, it crashes or misbehaves during slicing.
    mock_norm = MagicMock()
    mock_norm.transform.side_effect = lambda x: x.values if hasattr(x, 'values') else x
    detector.norm = mock_norm

    # 2. Use the Robust FakeKerasModel
    detector.model_dict = {'temp': FakeKerasModel()}

    # --- TEST EXECUTION ---

    # 1. Test LIVE MODE (batch_mode=False)
    live_data = sample_data.head(30)
    errs_live = detector.per_column_inference(live_data, batch_mode=False)

    # Should return a single scalar float
    assert isinstance(errs_live['temp'], float)

    # 2. Test BATCH MODE (batch_mode=True)
    batch_data = sample_data.head(30)
    errs_batch = detector.per_column_inference(batch_data, batch_mode=True)

    # Should return an array of errors
    assert isinstance(errs_batch['temp'], (np.ndarray, list))
    assert len(errs_batch['temp']) > 1


def test_save_load_metadata(tmp_path):
    """Test serialization of detector state."""
    d_out = AnomalyDetector()
    d_out.columns = ['a', 'b']
    d_out.threshold_dict = {'a': (0.5, 1.0)}
    d_out.sequence_length = 15

    d_out.norm = StandardScaler()
    d_out.norm.mean_ = np.array([0.0])
    d_out.norm.scale_ = np.array([1.0])
    d_out.diff_cols = []

    save_dir = str(tmp_path / "model_v1")
    d_out.save_to_disk(save_dir)

    assert os.path.exists(os.path.join(save_dir, 'metadata.pkl'))

    d_in = AnomalyDetector.load_from_disk(save_dir)
    assert d_in.columns == ['a', 'b']
    assert d_in.threshold_dict['a'] == (0.5, 1.0)
