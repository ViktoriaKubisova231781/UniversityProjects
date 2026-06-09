import streamlit as st
import pandas as pd
import os
from datetime import datetime
from collections import deque
from collectors.unified_collector import UnifiedDataCollector
from services.async_trainer import AsyncTrainer
from core_ml.anomaly_detection import AnomalyDetector


class SystemStateManager:
    """
    Central state management singleton for the dashboard.

    This class orchestrates the interaction between:
    1. Hardware (UnifiedDataCollector)
    2. Machine Learning (AsyncTrainer and AnomalyDetector)
    3. UI State (Alerts, Buffers, Connections)
    """
    def __init__(self):
        """
        Initializes system components, buffers, and directory structures.
        """
        # Hardware
        self.collector: UnifiedDataCollector = None
        self.is_connected = False

        # AI/ML
        self.model = None
        self.trainer = AsyncTrainer()
        self.inference_active = False

        # Paths
        self.models_dir = os.path.join(os.getcwd(), 'models')
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        # Buffers
        self.recent_predictions = deque(maxlen=200)
        self.alerts = deque(maxlen=50)

    # --- HELPER: UNITS OF MEASUREMENT ---
    def get_unit(self, column_name: str) -> str:
        """
        Returns the appropriate physical unit string for a given feature name.

        Args:
            column_name (str): The name of the data column.

        Returns:
            str: The unit (e.g., "°C", "V", "bar") or empty string if unknown.
        """
        col = column_name.lower()
        if 'temp' in col:
            return "°C"
        if 'volt' in col:
            return "V"
        if 'pressure' in col:
            return "bar"
        if 'vib' in col:
            return "g"  # Assuming acceleration
        if 'j' in col and any(char.isdigit() for char in col):
            return "rad"
        return ""

    def connect_robot(self, ip, user, password, port):
        """
        Establishes connection to the data collector.

        Args:
            ip (str): Robot IP.
            user (str): SSH Username.
            password (str): SSH Password.
            port (int): UDP Port for sensors.

        Returns:
            bool: True if connection successful, False otherwise.
        """
        if self.is_connected and self.collector:
            return True
        try:
            self.collector = UnifiedDataCollector(ip, user, password, port)
            self.is_connected = True
            return True
        except Exception as e:
            st.error(f"Connection Error: {e}")
            return False

    def disconnect_robot(self):
        """
        Safely closes hardware connections and stops active inference.
        """
        if self.collector:
            self.collector.close()
            self.collector = None
        self.is_connected = False
        self.inference_active = False

    # --- MODEL MANAGEMENT ---
    def get_available_models(self):
        """
        Lists all saved models available in the models directory.

        Returns:
            list[str]: List of model directory names.
        """
        if not os.path.exists(self.models_dir):
            return []
        return [d for d in os.listdir(self.models_dir) if os.path.isdir(os.path.join(self.models_dir, d))]

    def save_model(self, model_name):
        """
        Persists the currently loaded model to disk.

        Args:
            model_name (str): Name/Folder for the saved model.

        Returns:
            tuple: (Success boolean, Status message)
        """
        if not self.model:
            return False, "No active model."
        try:
            self.model.save_to_disk(os.path.join(self.models_dir, model_name))
            return True, f"Saved {model_name}"
        except Exception as e:
            return False, str(e)

    def load_model(self, model_name):
        """
        Loads a specific model from disk into active memory.

        Args:
            model_name (str): Name of the model to load.

        Returns:
            tuple: (Success boolean, Status message)
        """
        try:
            target = os.path.join(self.models_dir, model_name)
            if not os.path.exists(target):
                return False, "Not found."
            self.model = AnomalyDetector.load_from_disk(target)
            return True, f"Loaded {model_name}"
        except Exception as e:
            return False, str(e)

    # --- SYSTEM LOOP ---
    def update_system_state(self):
        """
        The main 'game loop' for the dashboard state.

        1. Updates training status if training is active.
        2. Swaps in new models upon training completion.
        3. Runs an inference step if connected and inference is enabled.
        """
        if not self.is_connected or not self.collector:
            if self.trainer.is_training:
                self.trainer.check_status()
            return

        if self.trainer.is_training:
            self.trainer.check_status()

        if self.trainer.result_model:
            self.model = self.trainer.result_model
            self.trainer.result_model = None
            st.toast("✅ New model loaded successfully!")

        if self.trainer.is_training:
            return

        if self.inference_active and self.model:
            self._run_inference_step()

    def _run_inference_step(self):
        """
        Internal method to fetch latest data, run model inference, and generate alerts.

        Updates `self.recent_predictions` and `self.alerts` based on results.
        """
        current_model = self.model
        if current_model is None:
            return

        try:
            df = self.collector.get_unified_dataframe()
            if df.empty:
                return

            raw_data = df.tail(1000).copy()
            if len(raw_data) < current_model.sequence_length * 2:
                return

            if not current_model.columns:
                return

            required_cols = list(set(current_model.columns) | {'timestamp'})
            if not all(c in raw_data.columns for c in required_cols if c != 'timestamp'):
                return

            inference_subset = raw_data[[c for c in required_cols if c in raw_data.columns]].copy()

            errs, preds = current_model.per_column_inference(inference_subset, return_preds=True)

            if not errs:
                return

            self.recent_predictions.append({
                'timestamp': pd.Timestamp.now(),
                'errors': errs,
                'predictions': preds,
                'actuals': current_model.last_input_dict
            })

            # Threshold Checks with UNITS
            for col, err in errs.items():
                if col in current_model.threshold_dict:
                    warn, crit = current_model.threshold_dict[col]
                    # Get Unit
                    unit = self.get_unit(col)

                    if err > crit:
                        self.add_alert('critical', f"{col} anomaly", col, f"{err:.3f}{unit}")
                    elif err > warn:
                        self.add_alert('warning', f"{col} warning", col, f"{err:.3f}{unit}")

        except Exception as e:
            print(f"DEBUG: Inference error: {e}")

    def add_alert(self, severity, message, metric, value):
        """
        Adds a new alert to the system with debounce logic (prevents spamming the same alert).

        Args:
            severity (str): 'critical' or 'warning'.
            message (str): Display message.
            metric (str): The column/metric causing the alert.
            value (str): The value triggering the alert (formatted with units).
        """
        if self.alerts:
            last = self.alerts[-1]
            if last['message'] == message and (datetime.now() - last['timestamp']).seconds < 5:
                return

        self.alerts.append({
            'timestamp': datetime.now(),
            'severity': severity,
            'message': message,
            'metric': metric,
            'value': value,
            'acknowledged': False
        })

    def get_active_alerts(self):
        """
        Returns all unacknowledged alerts.
        """
        return [a for a in self.alerts if not a['acknowledged']]


@st.cache_resource
def get_manager() -> SystemStateManager:
    """
    Streamlit cache resource to ensure only one instance of the SystemStateManager exists.
    """
    return SystemStateManager()
