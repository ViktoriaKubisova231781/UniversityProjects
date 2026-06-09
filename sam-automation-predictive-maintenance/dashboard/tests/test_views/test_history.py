import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from views.history import render_history_view


@pytest.fixture
def mock_streamlit():
    """Patches all streamlit functions used in history.py"""
    with patch("views.history.st") as mock_st:
        mock_st.radio.return_value = "All"
        mock_st.button.return_value = False
        yield mock_st


@pytest.fixture
def mock_manager():
    """Patches the singleton manager"""
    with patch("views.history.get_manager") as mock_get:
        mgr = MagicMock()
        mock_get.return_value = mgr
        mgr.load_model.return_value = (True, "Mock Loaded")
        yield mgr


def test_history_no_files_found(mock_streamlit, mock_manager):
    """Test behavior when directory scan returns nothing."""
    with patch("os.walk", return_value=[]):
        render_history_view()

    mock_streamlit.warning.assert_any_call("No .parquet files found.")


def test_history_file_selection_loads_data(mock_streamlit, mock_manager):
    """Test that selecting a file loads it into a dataframe."""
    fake_walk = [('/root', [], ['data.parquet'])]
    fake_df = pd.DataFrame({'timestamp': [1, 2, 3], 'temp': [10, 20, 30]})

    with patch("os.walk", return_value=fake_walk), \
         patch("os.path.getsize", return_value=1024), \
         patch("os.path.getmtime", return_value=10000), \
         patch("pandas.read_parquet", return_value=fake_df):

        mock_streamlit.selectbox.return_value = {
            "path": "/root/data.parquet",
            "name": "data.parquet",
            "display": "data.parquet (1 KB)"
        }

        render_history_view()

        args, _ = mock_streamlit.info.call_args
        assert "Loaded: data.parquet" in args[0]

        mock_streamlit.multiselect.assert_called()
        mock_streamlit.plotly_chart.assert_called()


def test_run_inference_on_file(mock_streamlit, mock_manager):
    """Test the 'Run Inference' button logic."""
    fake_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='s'),
        'temp': range(10)
    })

    mock_manager.model = MagicMock()
    mock_manager.model.columns = ['temp']
    # Thresholds: Warning @ 5, Critical @ 8
    mock_manager.model.threshold_dict = {'temp': (5, 8)}

    # Errors: [0...0, 6, 9]
    # 6 is > 5 (Warning)
    # 9 is > 8 (Critical) -> Only 1 Critical Anomaly
    errors = {'temp': pd.Series([0] * 8 + [6, 9])}
    mock_manager.model.per_column_inference.return_value = (errors, None)

    with patch("os.walk", return_value=[('/root', [], ['test.parquet'])]), \
         patch("os.path.getsize", return_value=1024), \
         patch("os.path.getmtime", return_value=10000), \
         patch("pandas.read_parquet", return_value=fake_df):

        mock_streamlit.selectbox.return_value = {"path": "test.parquet", "name": "test.parquet"}

        def button_side_effect(label, **kwargs):
            return "Run Inference" in label
        mock_streamlit.button.side_effect = button_side_effect

        render_history_view()

        mock_manager.model.per_column_inference.assert_called()

        # FIX: Expect 1 anomaly (Value 9 > 8), not 2
        mock_streamlit.error.assert_any_call("Found 1 critical anomalies!")
