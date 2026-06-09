import pytest
import socket
from unittest.mock import patch
from collectors.udp_sensor_collector_raw import UDPSensorCollector


@pytest.fixture
def udp_collector():
    """Creates a collector with a mocked socket."""
    with patch("socket.socket") as mock_socket_cls:
        print(f"Mock socket.socket: {mock_socket_cls}")
        collector = UDPSensorCollector(udp_port=5005)
        return collector


def test_parse_sensor_data_valid(udp_collector):
    """Test parsing standard Arduino string format."""
    # Input format: Unquoted keys, nested dict
    raw_data = "{pressure: 3000, vibrations: {1.5: 0.02, 3.0: 0.05}}"

    result = udp_collector._parse_sensor_data(raw_data)

    assert result is not None
    assert result['pressure'] == 3000

    # FIX: The parser regex turns numbers into strings ("1.5"),
    # so we must access them as strings here.
    assert result['vibrations']['1.5'] == 0.02
    assert result['vibrations']['3.0'] == 0.05


def test_parse_sensor_data_malformed(udp_collector):
    """Test resilience against bad data."""
    bad_data = "pressure: 3000, broken_json..."
    result = udp_collector._parse_sensor_data(bad_data)
    assert result is None


def test_collect_sample_logic(udp_collector):
    """Test converting raw parsed data into a flat dictionary row."""
    # 1. Inject parsed state manually
    # Note: The collector stores these as they were parsed (keys as strings or floats depending on parse logic)
    # But let's assume valid floats for the internal logic test if that's what the rest of your app expects.
    # However, based on the parse test above, they are likely strings.
    # The UDPSensorCollector logic iterates .items(), so it handles both.

    udp_collector.latest_pressure = 1000
    udp_collector.latest_vibrations = {'10.0': 0.5, '20.0': 0.8}

    # 2. Collect sample
    sample = udp_collector.collect_sample()

    # 3. Verify flattening
    assert sample['pressure'] == 1000
    # The math functions (np.mean) work on the values, so keys being strings is fine
    assert sample['vib_mean'] == 0.65  # (0.5 + 0.8) / 2

    # 4. Verify dynamic columns exist
    # The collector creates keys like f'vib_{freq}'
    assert sample['vib_10.0'] == 0.5
    assert sample['vib_20.0'] == 0.8


def test_socket_integration_loop(udp_collector):
    """Simulate receiving a packet and processing it."""
    fake_packet = b"{pressure: 5000, vibrations: {50.0: 0.1}}"

    # FIX: Define a side effect that returns data once, then STOPS the loop
    def intelligent_side_effect(*args, **kwargs):
        # On first call, return data
        if not getattr(intelligent_side_effect, 'called', False):
            intelligent_side_effect.called = True
            return (fake_packet, ('127.0.0.1', 1234))
        else:
            # On second call, stop the loop AND raise timeout
            udp_collector.collecting = False
            raise socket.timeout

    udp_collector.sock.recvfrom.side_effect = intelligent_side_effect

    # Run the worker
    udp_collector.collecting = True

    # This calls the worker directly (no thread) to keep it synchronous
    udp_collector._udp_listener_worker()

    # Assert data was updated
    assert udp_collector.latest_pressure == 5000
    # Access as string because of the regex parser
    assert '50.0' in udp_collector.latest_vibrations
