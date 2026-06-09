import pandas as pd
import numpy as np
import tensorflow as tf
import os
import shutil
import json
import pickle
import time
from tensorflow.keras.models import Model, load_model  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, Callback  # type: ignore

# RESPECTING YOUR IMPORT STRUCTURE
from core_ml.preprocessing import (
    preprocessing_pipeline,
    split_time_series,
    inverse_transform_single_column,
)
from core_ml.models import get_autoencoder
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
log = logging.getLogger(__name__)


class FileProgressCallback(Callback):
    """
    Custom Keras callback to write training progress to a JSON status file.
    Allows external processes (like a UI dashboard) to monitor training state in real-time.
    """
    def __init__(self, status_file, col_name, total_epochs):
        """
        Initializes the callback.

        Args:
            status_file (str): Path to the JSON file where status updates will be written.
            col_name (str): Name of the column currently being trained (for display).
            total_epochs (int): Total number of epochs for progress calculation.
        """
        super().__init__()
        self.status_file = status_file
        self.col_name = col_name
        self.total_epochs = total_epochs
        self.last_update = 0

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of an epoch. Updates the status file if more than 1 second
        has passed since the last update.

        Args:
            epoch (int): Index of epoch.
            logs (dict): Metric results for this training epoch.
        """
        now = time.time()
        if now - self.last_update > 1.0:
            logs = logs or {}
            status = {
                'state': 'training',
                'message': f"Training {self.col_name}: Epoch {epoch+1}/{self.total_epochs}",
                'loss': float(logs.get('loss', 0.0)),
                'progress': (epoch + 1) / self.total_epochs
            }
            try:
                with open(self.status_file, 'w') as f:
                    json.dump(status, f)
            except Exception as e:
                log.error(f"Error writing status file: {e}")
            self.last_update = now


class AnomalyDetector:
    """
    Manages the end-to-end anomaly detection pipeline using per-column Autoencoders.
    Handles data preprocessing, model training, threshold calculation, inference,
    and model serialization.
    """
    def __init__(
        self,
        data_frame: pd.DataFrame = None,
        filepath: str = None,
        model: Model = None,
        columns: list = None,
        test_split_ratio: float = 0.2,
        sequence_length: int = 60,
        training_epochs: int = 100,
        es_patience: int = 6,
        custom_tresholds: tuple = (3, 5),
        transform_output: bool = True,
        train_on_init: bool = True,
        verbose: int = 1,
        learning_rate: float = 0.001,
        encoder_seq: list = [64, 32],
    ):
        """
        Initializes the AnomalyDetector with configuration for data loading and model architecture.

        Args:
            data_frame (pd.DataFrame, optional): Input dataframe.
            filepath (str, optional): Path to a parquet file to load data from.
            model (Model, optional): Pre-existing Keras model (usually created internally).
            columns (list, optional): List of column names to train on.
            test_split_ratio (float): Ratio of data to use for testing/threshold calibration.
            sequence_length (int): Length of the time-series window (lookback).
            training_epochs (int): Maximum training epochs.
            es_patience (int): Early stopping patience epochs.
            custom_tresholds (tuple): Z-score multipliers for (yellow, red) alert thresholds.
            transform_output (bool): Whether to inverse transform predictions to original scale.
            train_on_init (bool): If True, starts training immediately upon initialization.
            verbose (int): Verbosity level.
            learning_rate (float): Learning rate for the Adam optimizer.
            encoder_seq (list): Layer sizes for the encoder portion of the autoencoder.
        """
        self.data_frame = data_frame
        self.filepath = filepath
        self.model = model
        self.columns = columns
        self.test_split_ratio = test_split_ratio
        self.sequence_length = sequence_length
        self.training_epochs = training_epochs
        self.es_patience = es_patience
        self.custom_tresholds = custom_tresholds
        self.transform_output = transform_output
        self.train_on_init = train_on_init
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.encoder_seq = encoder_seq

        self.model_dict = {}
        self.threshold_dict = {}

        if (self.data_frame is not None or self.filepath is not None) and self.train_on_init:
            self._per_col_training()

    def _load_data_file(self):
        """
        Loads data from the configured filepath (parquet) or returns the in-memory dataframe.

        Returns:
            pd.DataFrame: The loaded data.
        """
        if self.filepath is not None:
            return pd.read_parquet(self.filepath, engine="fastparquet")
        return self.data_frame

    def _preprocess_training_data(self):
        """
        Runs the full preprocessing pipeline: column filtering, train/test split,
        differencing, and normalization.

        Returns:
            tuple: (x_train, x_test, y_train, y_test) arrays ready for model fitting.
        """
        self.data_frame = self._load_data_file()
        if self.columns is not None:
            cols_to_keep = list(set(self.columns) | {"timestamp"})
            cols_to_keep = [c for c in cols_to_keep if c in self.data_frame.columns]
            self.data_frame = self.data_frame[cols_to_keep]

        train_df, test_df = split_time_series(self.data_frame, test_ratio=self.test_split_ratio)
        x_train, y_train, self.diff_cols, self.norm = preprocessing_pipeline(train_df, sequence_len=self.sequence_length)
        x_test, y_test, _, _ = preprocessing_pipeline(test_df, sequence_len=self.sequence_length, diff_cols=self.diff_cols, norm=self.norm)
        return x_train, x_test, y_train, y_test

    def _per_col_training(self, status_file=None):
        """
        Trains a separate Autoencoder model for each specified column.
        Calculates reconstruction error thresholds for anomaly detection.

        Args:
            status_file (str, optional): Path to write progress updates for the FileProgressCallback.

        Returns:
            dict: Dictionary mapping column names to trained Keras models.
        """
        tf.keras.backend.clear_session()
        x_train, x_test, y_train, y_test = self._preprocess_training_data()
        cols = [c for c in self.data_frame.columns if c != "timestamp"]

        for i, col in enumerate(cols):
            if status_file:
                with open(status_file, 'w') as f:
                    json.dump({'state': 'training', 'message': f"Initializing model for {col}...", 'progress': 0}, f)

            x_train_col = np.expand_dims(x_train[:, :, i], axis=-1).astype('float32')
            y_train_col = np.expand_dims(y_train[:, i], axis=-1).astype('float32')
            x_test_col = np.expand_dims(x_test[:, :, i], axis=-1).astype('float32')
            y_test_col = np.expand_dims(y_test[:, i], axis=-1).astype('float32')

            model = get_autoencoder(input_shape=(None, self.sequence_length, 1), learning_rate=self.learning_rate, encoder_seq=self.encoder_seq)
            es = EarlyStopping(monitor="val_loss", patience=self.es_patience, restore_best_weights=True)

            callbacks = [es]
            if status_file:
                callbacks.append(FileProgressCallback(status_file, col, self.training_epochs))

            model.fit(
                x_train_col, y_train_col, epochs=self.training_epochs, batch_size=32,
                validation_data=(x_test_col, y_test_col), callbacks=callbacks, verbose=0
            )

            thresh_y, thresh_r = self._get_feature_threshold(model, x_test_col, transform=self.transform_output, col_index=i)
            self.threshold_dict[col] = (float(thresh_y), float(thresh_r))
            self.model_dict[col] = model

        return self.model_dict

    # CRITICAL UPDATE: added batch_mode parameter
    def per_column_inference(self, df, return_preds=False, batch_mode=False):
        """
        Runs inference on new data to detect anomalies.

        Args:
            df (pd.DataFrame): New data to analyze.
            return_preds (bool): If True, returns predictions alongside errors.
            batch_mode (bool):
                - If True: Returns full arrays of errors (used for historical analysis/graphs).
                - If False: Returns single scalar values for the most recent timestamp (used for live monitoring).

        Returns:
            dict or tuple: A dictionary of errors per column, or (errors, predictions) if return_preds is True.
        """
        x_inf, _, _, _ = preprocessing_pipeline(df, sequence_len=self.sequence_length, diff_cols=self.diff_cols, norm=self.norm)
        error_dict = {}
        pred_dict = {}
        self.last_input_dict = {}

        valid_cols = [c for c in df.columns if c in self.model_dict]
        for col in valid_cols:
            try:
                feature_cols = []
                if self.columns:
                    feature_cols = [c for c in self.columns if c != 'timestamp']
                elif self.data_frame is not None:
                    feature_cols = [c for c in self.data_frame.columns if c != 'timestamp']

                if col in feature_cols:
                    col_idx = feature_cols.index(col)
                else:
                    continue
            except ValueError:
                continue

            model = self.model_dict[col]
            x_inf_col = np.expand_dims(x_inf[:, :, col_idx], axis=-1).astype('float32')

            y_pred_col = model(x_inf_col, training=False).numpy()

            x_input_last = x_inf[:, -1, col_idx]
            y_pred_last = np.mean(y_pred_col, axis=1).squeeze()

            if self.transform_output:
                x_input_last = inverse_transform_single_column(self.norm, x_input_last, column_index=col_idx)
                y_pred_last = inverse_transform_single_column(self.norm, y_pred_last, column_index=col_idx)

            abs_error = np.abs(x_input_last - y_pred_last)

            # --- FIX FOR "FLOAT HAS NO LEN" ---
            if not batch_mode:
                # Live Mode: Reduce to single scalar
                if np.ndim(abs_error) > 0:
                    abs_error = abs_error[-1] if len(abs_error) > 1 else abs_error[0]
                if np.ndim(y_pred_last) > 0:
                    y_pred_last = y_pred_last[-1] if len(y_pred_last) > 1 else y_pred_last[0]
                if np.ndim(x_input_last) > 0:
                    x_input_last = x_input_last[-1] if len(x_input_last) > 1 else x_input_last[0]

                error_dict[col] = float(abs_error)
                pred_dict[col] = float(y_pred_last)
                self.last_input_dict[col] = float(x_input_last)
            else:
                # Batch Mode: Keep the full array
                error_dict[col] = abs_error
                pred_dict[col] = y_pred_last
                self.last_input_dict[col] = x_input_last

        if return_preds:
            return error_dict, pred_dict
        return error_dict

    def _get_zscore_thresholds(self, errors, z1=3, z2=5):
        """
        Calculates threshold values based on the mean and standard deviation of errors.

        Args:
            errors (np.array): Array of reconstruction errors.
            z1 (int): Z-score multiplier for the first (warning) threshold.
            z2 (int): Z-score multiplier for the second (critical) threshold.

        Returns:
            tuple: (threshold_1, threshold_2)
        """
        mean, std = np.mean(errors), np.std(errors)
        return (mean + z1 * std), (mean + z2 * std)

    def _get_feature_threshold(self, model, training_data, yellow=3, red=5, transform=False, col_index=None):
        """
        Computes anomaly thresholds for a specific model based on test data reconstruction error.

        Args:
            model (Model): The trained autoencoder.
            training_data (np.array): Data to run inference on for calibration.
            yellow (int): Z-score for warning threshold.
            red (int): Z-score for critical threshold.
            transform (bool): Whether to inverse transform values before error calculation.
            col_index (int): Index of the column (for inverse transformation).

        Returns:
            tuple: (yellow_threshold, red_threshold)
        """
        training_data = training_data.astype('float32')
        y_pred = model(training_data, training=False).numpy()
        y_pred_mean = np.mean(y_pred, axis=1)
        training_data_last = training_data[:, -1, 0]

        if transform:
            y_pred_mean = inverse_transform_single_column(self.norm, y_pred_mean, column_index=col_index)
            training_data_last = inverse_transform_single_column(self.norm, training_data_last, column_index=col_index)

        error = np.abs(training_data_last - y_pred_mean)
        return self._get_zscore_thresholds(error, yellow, red)

    def save_to_disk(self, directory):
        """
        Serializes the entire detector state to disk.
        Saves metadata, preprocessing scalers, and a Keras model file for each column.

        Args:
            directory (str): Path to the output directory. Clears directory if it exists.
        """
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

        metadata = {
            'columns': self.columns,
            'sequence_length': self.sequence_length,
            'threshold_dict': self.threshold_dict,
            'transform_output': self.transform_output
        }
        with open(os.path.join(directory, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)

        with open(os.path.join(directory, 'preprocessing.pkl'), 'wb') as f:
            pickle.dump({'norm': self.norm, 'diff_cols': self.diff_cols}, f)

        for col, model in self.model_dict.items():
            safe_col = "".join([c if c.isalnum() else "_" for c in col])
            model.save(os.path.join(directory, f"model_{safe_col}.keras"))

    @classmethod
    def load_from_disk(cls, directory, dummy_df=None):
        """
        Factory method to restore an AnomalyDetector instance from disk.

        Args:
            directory (str): Path to the directory containing saved models and metadata.
            dummy_df (pd.DataFrame, optional): Optional dataframe to initialize the instance with.

        Returns:
            AnomalyDetector: A fully initialized and loaded instance.
        """
        instance = cls()
        with open(os.path.join(directory, 'metadata.pkl'), 'rb') as f:
            meta = pickle.load(f)

        instance.columns = meta['columns']
        instance.sequence_length = meta['sequence_length']
        instance.threshold_dict = meta['threshold_dict']
        instance.transform_output = meta['transform_output']

        instance.data_frame = pd.DataFrame(columns=instance.columns)

        with open(os.path.join(directory, 'preprocessing.pkl'), 'rb') as f:
            prep = pickle.load(f)
            instance.norm = prep['norm']
            instance.diff_cols = prep['diff_cols']

        instance.model_dict = {}
        for col in instance.columns:
            if col == 'timestamp':
                continue
            safe_col = "".join([c if c.isalnum() else "_" for c in col])
            model_path = os.path.join(directory, f"model_{safe_col}.keras")
            if os.path.exists(model_path):
                instance.model_dict[col] = load_model(model_path, compile=False)
                instance.model_dict[col].compile(optimizer='adam', loss='mse', run_eagerly=True)

        if dummy_df is not None:
            instance.data_frame = dummy_df

        return instance
