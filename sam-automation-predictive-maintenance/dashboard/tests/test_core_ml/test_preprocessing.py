import pandas as pd
import numpy as np
from unittest.mock import patch
from core_ml.preprocessing import (
    split_time_series,
    to_sequence,
    check_stationarity,
    inverse_transform_single_column,
    preprocessing_pipeline
)
from sklearn.preprocessing import StandardScaler


def test_split_time_series():
    """Test strict temporal splitting (no shuffling)."""
    df = pd.DataFrame({'val': range(100)})
    train, test = split_time_series(df, test_ratio=0.2)

    assert len(train) == 80
    assert len(test) == 20
    assert train.iloc[-1]['val'] == 79
    assert test.iloc[0]['val'] == 80


def test_to_sequence():
    """Test LSTM sliding window generation."""
    series = np.array([0, 1, 2, 3, 4])
    x, y = to_sequence(series, seq_len=3)

    assert x.shape == (2, 3)
    assert y.shape == (2,)
    assert np.array_equal(x[0], [0, 1, 2])
    assert y[0] == 3


@patch("core_ml.preprocessing.adfuller")
def test_check_stationarity(mock_adfuller):
    """Test stationarity detection logic."""
    df = pd.DataFrame({
        'stat': [1, 1, 1, 1, 1, 2, 1, 1, 1, 1],
        'non_stat': range(10)
    })

    # FIX: Provide a dictionary with '1%' key for the critical values (Index 4)
    # adfuller returns: (adf_stat, pvalue, usedlag, nobs, critical_values, icbest)
    mock_adfuller.side_effect = [
        # Call 1 (Stationary): p-value 0.01 (small)
        (-5.0, 0.01, 0, 0, {'1%': -3.5}, 0),
        # Call 2 (Non-Stationary): p-value 0.99 (large)
        (-1.0, 0.99, 0, 0, {'1%': -3.5}, 0)
    ]

    results = check_stationarity(df)
    assert results['stat'] is True
    assert results['non_stat'] is False


def test_inverse_transform():
    """Test scaling reversal."""
    scaler = StandardScaler()
    data = np.array([[0], [10]])
    scaler.fit(data)

    scaled = scaler.transform([[10]])
    assert scaled[0][0] == 1.0

    restored = inverse_transform_single_column(scaler, 1.0, column_index=0)
    assert restored == 10.0


@patch("core_ml.preprocessing.check_stationarity")
@patch("core_ml.preprocessing.apply_kalman_filtering")
def test_pipeline_integration(mock_kalman, mock_stationarity):
    """Test the full pipeline end-to-end."""
    mock_kalman.side_effect = lambda x: x
    mock_stationarity.return_value = {'sensor': False}

    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=50, freq='500ms'),
        'sensor': range(50)
    })

    x, y, diff_cols, norm = preprocessing_pipeline(
        df, sampling_period='500ms', sequence_len=5
    )

    assert x.shape[1] == 5
    assert 'sensor' in diff_cols
