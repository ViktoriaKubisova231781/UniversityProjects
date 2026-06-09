import pandas as pd
from core_ml.filters import apply_kalman_filtering


def test_kalman_shape():
    """Ensure filter returns series of same shape."""
    data = pd.Series([1.0, 2.0, 10.0, 4.0, 5.0])  # 10.0 is a spike

    result = apply_kalman_filtering(data)

    assert len(result) == len(data)

    # FIX: Use numpy indexing.
    # Handle both (N, 1) and (N,) shapes just in case
    if result.ndim > 1:
        val = result[2][0]
    else:
        val = result[2]

    # Kalman should smooth the spike at index 2 (value 10.0)
    # The result should be significantly less than 10.0
    assert val < 10.0
