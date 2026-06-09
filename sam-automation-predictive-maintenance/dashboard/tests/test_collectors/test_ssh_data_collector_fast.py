import pytest
from unittest.mock import patch
from collectors.ssh_data_collector_fast import RobotDataCollector


@pytest.fixture
def ssh_collector():
    """Mocks the SSH client and channel creation."""
    with patch("paramiko.SSHClient") as mock_client:
        print(f"Mock SSHClient: {mock_client}")
        # The constructor calls connect() and invoke_shell() twice
        collector = RobotDataCollector("192.168.1.1", "user", "pass")
        return collector


def test_parse_line_temperatures(ssh_collector):
    """Test parsing the specific string format from the robot."""
    line = "temperatures: [45.5, 46.0, 47.1, 0, 0, 0, 0, 0]"
    metric, values = ssh_collector._parse_line(line)

    assert metric == "temperatures"
    assert len(values) == 8
    assert values[0] == 45.5


def test_parse_line_malformed(ssh_collector):
    """Ensure parser doesn't crash on bad lines."""
    line = "Random system log message..."
    metric, values = ssh_collector._parse_line(line)
    assert metric is None
    assert values is None


def test_data_collection_logic(ssh_collector):
    """Test extracting a single sample from the latest stream data."""
    # 1. Setup internal state (simulating that threads have updated these)
    ssh_collector.latest_hw_data = {
        'temperatures': [40, 41, 42, 43, 44, 45, 0, 0],
        'voltages': [12.1, 12.0, 12.1, 12.2, 12.0, 12.1, 0, 0]
    }
    ssh_collector.latest_joint_data = {
        'position': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    }

    # 2. Collect
    sample = ssh_collector.collect_sample()

    # 3. Assertions
    assert sample['j1'] == 0.1
    assert sample['motor1_temp'] == 40
    assert sample['motor1_voltage'] == 12.1
    # Check calculated fields
    assert sample['avg_motor_temp'] == 42.5  # Mean of 40..45
