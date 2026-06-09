import pytest
import pandas as pd
from unittest.mock import patch
from collectors.unified_collector import UnifiedDataCollector


@pytest.fixture
def unified_collector():
    """Mock the sub-collectors so we don't need network."""
    with patch("collectors.unified_collector.RobotDataCollector"), \
         patch("collectors.unified_collector.UDPSensorCollector"):

        collector = UnifiedDataCollector("ip", "u", "p")
        return collector


def test_merge_logic_alignment(unified_collector):
    """
    Test that robot and sensor data merge correctly by timestamp.
    This simulates the core logic of _merge_worker.
    """
    # 1. Create Fake Robot Data (Base time T)
    robot_df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01 10:00:00', '2024-01-01 10:00:01']),
        'motor_temp': [50, 51]
    })

    # 2. Create Fake Sensor Data (Slightly offset time T + 0.1s)
    # The merge tolerance is 1s, so these should match.
    sensor_df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01 10:00:00.100', '2024-01-01 10:00:01.100']),
        'pressure': [1000, 1005]
    })

    # Mock the getters
    unified_collector.robot_collector.get_dataframe.return_value = robot_df
    unified_collector.sensor_collector.get_dataframe.return_value = sensor_df

    # 3. Run one iteration of logic manually (copy-paste logic from _merge_worker essentially)
    # We use pd.merge_asof same as the actual code
    merged = pd.merge_asof(
        robot_df,
        sensor_df,
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('1s')
    )

    # 4. Assertions
    assert len(merged) == 2
    # Row 1: Robot at 10:00:00 should match Sensor at 10:00:00.100
    assert merged.iloc[0]['motor_temp'] == 50
    assert merged.iloc[0]['pressure'] == 1000

    # Verify strict tolerance (if sensor was too far away)
    sensor_late = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01 10:00:05']),  # 5 seconds later
        'pressure': [9999]
    })

    merged_late = pd.merge_asof(
        robot_df,
        sensor_late,
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('1s')
    )
    # Should find NO match for the pressure, resulting in NaN
    assert pd.isna(merged_late.iloc[0]['pressure'])


def test_save_data_structure(unified_collector, tmp_path):
    """Test that it creates folders and parquet files."""
    # Add dummy data to buffer
    unified_collector.unified_buffer.append({'timestamp': 1, 'data': 1})

    # Save to a temporary directory provided by pytest
    saved_dir = unified_collector.save_data(output_dir=str(tmp_path))

    import os
    assert os.path.exists(saved_dir)
    # Check for files
    files = os.listdir(saved_dir)
    assert any("unified" in f for f in files)
