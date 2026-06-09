import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from services.state_manager import SystemStateManager


@pytest.fixture
def active_manager():
    """
    Creates a Manager that is already 'connected' and has a 'model' loaded.
    This skips the boilerplate setup so we can test the Loop.
    """
    with patch("services.state_manager.UnifiedDataCollector") as mock_collector:
        manager = SystemStateManager()

        # 1. Simulate Active Connection
        manager.collector = mock_collector.return_value
        manager.is_connected = True
        manager.inference_active = True

        # 2. Simulate Loaded Model
        mock_model = MagicMock()
        mock_model.sequence_length = 10
        mock_model.columns = ['temp', 'vibration']
        # Define Thresholds: Warning at 0.5, Critical at 1.0
        mock_model.threshold_dict = {'temp': (0.5, 1.0)}

        manager.model = mock_model

        return manager


def test_inference_loop_detects_critical_anomaly(active_manager):
    """
    SCENARIO:
    The robot sensor sends data. The model calculates a high reconstruction error (1.5).
    The System Manager should immediately generate a CRITICAL alert.
    """
    # --- ARRANGE ---
    # 1. Create fake sensor data (Enough rows to satisfy the 'sequence_length * 2' check)
    # We need at least 20 rows since sequence_length is 10
    fake_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=50, freq='s'),
        'temp': np.random.rand(50),
        'vibration': np.random.rand(50)
    })

    # Mock the collector to return this data
    active_manager.collector.get_unified_dataframe.return_value = fake_data

    # 2. Mock the Model's Inference Result
    # Simulate that the model found a massive error in 'temp' (1.5 is > 1.0 threshold)
    # format: (errors_dict, predictions_dict)
    active_manager.model.per_column_inference.return_value = (
        {'temp': 1.5},
        {'temp': [0.1] * 10}
    )

    # --- ACT ---
    # Run the main system loop once
    active_manager.update_system_state()

    # --- ASSERT ---
    # 1. Did it actually try to run inference?
    active_manager.model.per_column_inference.assert_called_once()

    # 2. Did it generate an alert?
    assert len(active_manager.alerts) == 1
    alert = active_manager.alerts[0]

    # 3. Is the alert accurate?
    assert alert['severity'] == 'critical'
    assert alert['metric'] == 'temp'
    assert "1.500°C" in alert['value']  # Checks if it formatted the unit correctly
    print("\n✅ SUCCESS: System correctly escalated a 1.5 error to a Critical Alert.")
