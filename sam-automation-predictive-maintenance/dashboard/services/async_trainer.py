import multiprocessing
import pandas as pd
import traceback
import os
import json
from core_ml.anomaly_detection import AnomalyDetector


def training_worker(status_file, df, features, epochs, temp_dir):
    """
    Multiprocessing worker function that executes the model training pipeline in a separate process.

    It communicates progress, status, and errors back to the main process by writing to a JSON file.

    Args:
        status_file (str): Path to the JSON file used for Inter-Process Communication (IPC).
        df (pd.DataFrame): The training dataset.
        features (list): List of column names to train on.
        epochs (int): Maximum number of training epochs.
        seq_len (int): Length of the input sequences (lookback window).
        patience (int): Early stopping patience epochs.
        temp_dir (str): Directory to save the trained model artifacts.
    """
    try:
        print(f"DEBUG: Child process started. PID: {os.getpid()}")

        with open(status_file, 'w') as f:
            json.dump({'state': 'init', 'message': 'Preparing data...'}, f)

        df = df.reset_index(drop=True)
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='S')

        cols_to_use = ['timestamp'] + [f for f in features if f != 'timestamp']

        detector = AnomalyDetector()
        detector.data_frame = df
        detector.columns = cols_to_use
        detector.training_epochs = epochs
        detector.test_split_ratio = 0.2
        detector.transform_output = True

        print("DEBUG: Starting _per_col_training...")
        detector._per_col_training(status_file=status_file)

        with open(status_file, 'w') as f:
            json.dump({'state': 'saving', 'message': 'Saving to disk...'}, f)

        detector.save_to_disk(temp_dir)

        with open(status_file, 'w') as f:
            json.dump({'state': 'done', 'path': temp_dir}, f)
        print("DEBUG: Child process finished successfully.")

    except Exception as e:
        print(f"DEBUG: Child process ERROR: {e}")
        traceback.print_exc()
        try:
            with open(status_file, 'w') as f:
                json.dump({'state': 'error', 'message': str(e)}, f)
        except Exception as fe:
            print(f"DEBUG: Failed to write error status: {fe}")


class AsyncTrainer:
    """
    Manages background model training to prevent blocking the main UI thread.

    Handles process spawning, status polling, and model rehydration upon completion.
    """
    def __init__(self):
        """
        Initializes the trainer state and prepares temporary directories for IPC.
        """
        self.is_training = False
        self.progress = 0.0
        self.status_message = ""
        self.result_model = None
        self.error = None
        self._process = None

        self._base_temp = os.path.join(os.getcwd(), "temp_training_data")
        if not os.path.exists(self._base_temp):
            os.makedirs(self._base_temp)

        self._temp_dir = os.path.join(self._base_temp, "model_output")
        self._status_file = os.path.join(self._base_temp, "status.json")

    def start_training(self, df: pd.DataFrame, selected_features: list,
                       epochs: int):
        """
        Spawns a new background process to train the anomaly detection model.

        Args:
            df (pd.DataFrame): The training data.
            selected_features (list): Columns to include in the model.
            epochs (int): Training epochs.
            seq_len (int): Sequence length.
            patience (int): Early stopping patience.

        Returns:
            bool: True if training started successfully, False if training is already in progress.
        """
        if self.is_training:
            return False

        self.is_training = True
        self.progress = 0.0
        self.status_message = "Starting..."
        self.error = None
        self.result_model = None

        if os.path.exists(self._status_file):
            os.remove(self._status_file)

        self._process = multiprocessing.Process(
            target=training_worker,
            args=(self._status_file, df, selected_features, epochs, self._temp_dir)
        )
        self._process.start()
        return True

    def check_status(self):
        """
        Polls the status file to update progress or retrieve the trained model.

        If training is 'done', this method loads the model from disk into `self.result_model`
        and cleans up the background process.
        """
        if not self.is_training:
            return

        if os.path.exists(self._status_file):
            try:
                with open(self._status_file, 'r') as f:
                    status = json.load(f)

                state = status.get('state', '')

                if state == 'done':
                    self.status_message = "Loading model..."
                    try:
                        self.result_model = AnomalyDetector.load_from_disk(status['path'])
                        self.progress = 1.0
                        self.status_message = "Complete!"
                    except Exception as e:
                        print(f"DEBUG: Model Load Error: {e}")
                        self.error = f"Load failed: {e}"
                    finally:
                        self.is_training = False
                        if self._process:
                            self._process.join()

                elif state == 'error':
                    self.error = status.get('message', 'Unknown error')
                    self.is_training = False
                    if self._process:
                        self._process.join()

                else:
                    self.status_message = status.get('message', 'Working...')
                    if 'progress' in status:
                        self.progress = 0.1 + (status['progress'] * 0.8)
                    else:
                        self.progress = 0.5

            except json.JSONDecodeError:
                pass  # File being written to, skip
            except Exception as e:
                print(f"DEBUG: check_status failed: {e}")
