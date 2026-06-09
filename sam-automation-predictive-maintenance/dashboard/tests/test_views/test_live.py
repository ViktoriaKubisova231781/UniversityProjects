import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from views.live import render_live_view


@pytest.fixture
def mock_streamlit():
    with patch("views.live.st") as mock_st:
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_st.tabs.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()]

        # FIX: Default buttons to False so we don't accidentally trigger 'Load Model'
        mock_st.button.return_value = False

        yield mock_st


@pytest.fixture
def mock_manager():
    with patch("views.live.get_manager") as mock_get:
        mgr = MagicMock()
        mock_get.return_value = mgr

        # Default State: Not connected
        mgr.is_connected = False
        mgr.collector = MagicMock()
        mgr.collector.collecting = False

        # FIX: Prevent unpacking error if load_model is called
        mgr.load_model.return_value = (True, "Mock Loaded")

        yield mgr


def test_live_view_disconnected_state(mock_streamlit, mock_manager):
    """Ensure it asks user to connect if disconnected."""
    mock_manager.is_connected = False

    render_live_view()

    mock_streamlit.info.assert_called_with("👈 Please connect to the robot in the sidebar to begin.")
    mock_streamlit.metric.assert_not_called()


def test_live_view_connected_controls(mock_streamlit, mock_manager):
    """Test Start/Stop buttons."""
    mock_manager.is_connected = True

    # Case A: Not collecting -> Show Start
    mock_manager.collector.collecting = False
    # Only click "Start"
    mock_streamlit.button.side_effect = lambda label, **kwargs: label == "▶️ Start"

    render_live_view()

    mock_manager.collector.start_collection.assert_called()

    # Case B: Collecting -> Show Stop
    mock_manager.collector.collecting = True
    mock_manager.collector.start_collection.reset_mock()
    # Only click "Stop"
    mock_streamlit.button.side_effect = lambda label, **kwargs: label == "⏹️ Stop"

    render_live_view()

    mock_manager.collector.stop_collection.assert_called()


def test_live_alert_banner(mock_streamlit, mock_manager):
    """Test that critical alerts trigger the big red error banner."""
    mock_manager.is_connected = True

    mock_manager.get_active_alerts.return_value = [
        {'severity': 'critical', 'message': 'High Temp',
         'timestamp': datetime.now(), 'value': '100C', 'acknowledged': False}
    ]

    render_live_view()

    mock_streamlit.error.assert_any_call("🚨 1 CRITICAL ANOMALIES DETECTED")


def test_model_loader_sidebar(mock_streamlit, mock_manager):
    """Test loading a model from the sidebar."""
    mock_manager.get_available_models.return_value = ["model_v1"]

    mock_streamlit.selectbox.return_value = "model_v1"

    # Enable the "Load" button specifically
    def button_logic(label, **kwargs):
        return "Load" in label
    mock_streamlit.button.side_effect = button_logic

    mock_manager.load_model.return_value = (True, "Loaded OK")

    render_live_view()

    mock_manager.load_model.assert_called_with("model_v1")
    mock_streamlit.success.assert_called_with("Loaded OK")
