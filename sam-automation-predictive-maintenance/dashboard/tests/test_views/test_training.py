import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from views.training import render_training_view


@pytest.fixture
def mock_streamlit():
    with patch("views.training.st") as mock_st:
        # FIX: Make columns return a list of the exact length requested
        def columns_side_effect(spec):
            if isinstance(spec, int):
                count = spec
            else:
                count = len(spec)
            return [MagicMock() for _ in range(count)]

        mock_st.columns.side_effect = columns_side_effect

        mock_st.radio.return_value = "All"
        mock_st.button.return_value = False
        yield mock_st


@pytest.fixture
def mock_manager():
    with patch("views.training.get_manager") as mock_get:
        mgr = MagicMock()
        mock_get.return_value = mgr

        mgr.trainer = MagicMock()
        mgr.trainer.is_training = False
        mgr.trainer.result_model = None
        mgr.trainer.error = None
        mgr.save_model.return_value = (True, "Saved")

        yield mgr


def test_training_data_source_live_buffer(mock_streamlit, mock_manager):
    """Test loading data from the live collector buffer."""
    mock_streamlit.radio.return_value = "📊 Live Buffer"

    fake_df = pd.DataFrame({'temp': [1, 2, 3]})
    mock_manager.collector.get_unified_dataframe.return_value = fake_df

    render_training_view()

    args, _ = mock_streamlit.success.call_args
    assert "Loaded 3 samples" in args[0]


def test_start_training_button(mock_streamlit, mock_manager):
    """Test that clicking Start triggers the async trainer."""
    mock_streamlit.radio.return_value = "📊 Live Buffer"
    fake_df = pd.DataFrame({'temp': [1, 2, 3], 'timestamp': [1, 2, 3]})
    mock_manager.collector.get_unified_dataframe.return_value = fake_df

    mock_streamlit.multiselect.return_value = ['temp']

    mock_streamlit.button.side_effect = lambda label, **kwargs: "Start Training" in label

    render_training_view()

    mock_manager.trainer.start_training.assert_called()


def test_training_completion_and_saving(mock_streamlit, mock_manager):
    """Test the flow after training finishes successfully."""
    # 1. SETUP: Ensure the view has data so it doesn't return early
    mock_streamlit.radio.return_value = "📊 Live Buffer"
    mock_manager.collector.get_unified_dataframe.return_value = pd.DataFrame({'temp': [1, 2, 3]})

    # 2. Simulate Training Complete state
    mock_manager.trainer.is_training = False
    mock_manager.trainer.result_model = MagicMock()
    mock_manager.trainer.result_model.threshold_dict = {'temp': (0.5, 1.0)}

    # 3. Simulate User Input for Save
    mock_streamlit.text_input.return_value = "my_new_model"
    mock_streamlit.button.side_effect = lambda label, **kwargs: label == "Save to Disk"

    mock_manager.save_model.return_value = (True, "Saved!")

    render_training_view()

    # 4. Assertions
    mock_streamlit.success.assert_any_call("✅ Training Complete!")
    mock_manager.save_model.assert_called_with("my_new_model")
