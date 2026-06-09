import pytest
import json
import os
from unittest.mock import MagicMock, patch
from services.async_trainer import AsyncTrainer


@pytest.fixture
def trainer(tmp_path):
    """Create trainer with temporary paths."""
    # Redirect internal paths to a temp directory so we don't clutter the project
    with patch("services.async_trainer.os.getcwd", return_value=str(tmp_path)):
        t = AsyncTrainer()
        return t


@patch("services.async_trainer.multiprocessing.Process")
def test_start_training(mock_process, trainer):
    """Test that start_training initializes the process correctly."""
    df = MagicMock()

    success = trainer.start_training(
        df=df,
        selected_features=['temp'],
        epochs=1,
    )

    assert success is True
    assert trainer.is_training is True
    assert trainer.status_message == "Starting..."
    mock_process.assert_called_once()
    mock_process.return_value.start.assert_called_once()


def test_check_status_running(trainer, tmp_path):
    """Test reading 'running' status from JSON file."""
    trainer.is_training = True

    # Create a fake status file in the temp directory
    status_file = os.path.join(trainer._base_temp, "status.json")
    os.makedirs(trainer._base_temp, exist_ok=True)

    with open(status_file, 'w') as f:
        json.dump({'state': 'running', 'message': 'Epoch 5/10', 'progress': 0.5}, f)

    # Act
    trainer.check_status()

    # Assert
    assert trainer.status_message == "Epoch 5/10"
    # Progress logic: 0.1 + (0.5 * 0.8) = 0.5
    assert trainer.progress == 0.5


@patch("services.async_trainer.AnomalyDetector")
def test_check_status_done(mock_detector, trainer, tmp_path):
    """Test reading 'done' status and loading the model."""
    trainer.is_training = True
    trainer._process = MagicMock()  # Mock the process so we can call join()

    # Create fake completion status
    status_file = os.path.join(trainer._base_temp, "status.json")
    os.makedirs(trainer._base_temp, exist_ok=True)

    fake_model_path = os.path.join(trainer._base_temp, "model_output")
    with open(status_file, 'w') as f:
        json.dump({'state': 'done', 'path': fake_model_path}, f)

    # Act
    trainer.check_status()

    # Assert
    assert trainer.is_training is False
    assert trainer.status_message == "Complete!"
    assert trainer.progress == 1.0
    mock_detector.load_from_disk.assert_called_with(fake_model_path)
