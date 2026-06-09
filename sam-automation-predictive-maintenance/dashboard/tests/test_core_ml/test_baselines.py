import pytest
import pandas as pd
import numpy as np
from core_ml.baselines import apply_LOF, apply_iforest


@pytest.fixture
def synthetic_data():
    """
    Creates a DataFrame with 99 'normal' points and 1 massive outlier.
    """
    # 99 data points clustered around 10
    normal = np.random.normal(loc=10, scale=0.5, size=99)
    # 1 outlier at 1000
    data = np.append(normal, 1000)

    df = pd.DataFrame(data, columns=['sensor_val'])
    return df


def test_lof_detects_outlier(synthetic_data):
    """Test Local Outlier Factor identifies the obvious anomaly."""
    preds, scores = apply_LOF(synthetic_data, contamination=0.05)

    # In Sklearn: -1 is outlier, 1 is inlier
    assert preds[-1] == -1


def test_iforest_detects_outlier(synthetic_data):
    """Test Isolation Forest identifies the obvious anomaly."""
    preds, probs = apply_iforest(synthetic_data, contamination=0.05)

    # The last element (1000) should be flagged as an outlier (-1)
    assert preds[-1] == -1

    # FIX: Isolation Forest scores are LOWER for anomalies.
    # So the outlier (last index) should have a LOWER score than normal data (first index).
    assert probs[-1] < probs[0]


def test_baselines_handle_multi_dim():
    """Ensure functions don't crash on multi-column data."""
    df = pd.DataFrame({
        'a': range(20),
        'b': range(20)
    })

    preds, scores = apply_LOF(df)
    assert len(preds) == 20

    preds_i, probs_i = apply_iforest(df)
    assert len(preds_i) == 20
