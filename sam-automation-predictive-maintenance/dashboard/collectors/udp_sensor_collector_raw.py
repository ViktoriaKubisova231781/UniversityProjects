import socket
import threading
import time
import ast
import os
from collections import deque
from datetime import datetime
import pandas as pd
import numpy as np


class UDPSensorCollector:
    """
    Listens for UDP packets containing sensor data (pressure and vibration spectra),
    parses raw string data, and buffers structured samples.
    """
    def __init__(self, udp_ip="", udp_port=5005):
        """
        Initializes the UDP socket and binds to the specified port.

        Args:
            udp_ip (str): IP to bind to (default "" for all interfaces).
            udp_port (int): Port to listen on (default 5005).
        """
        self.udp_ip = udp_ip
        self.udp_port = udp_port

        # Setup UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.udp_ip, self.udp_port))
        self.sock.settimeout(1.0)

        # Data storage
        self.data_buffer = deque(maxlen=50000)
        self.collecting = False
        self.samples_collected = 0

        # Latest sensor data
        self.latest_pressure = 0
        self.latest_vibrations = {}
        self.data_lock = threading.Lock()

        # Track all unique frequencies we've seen
        self.all_frequencies = set()

        print(f"✅ UDP Listener started on port {udp_port}")

    def _parse_sensor_data(self, data_str):
        """
        Parses the raw incoming string (often quasi-JSON from Arduino) into a valid Python dict.
        Handles specific string replacements to fix unquoted keys in nested dictionaries.

        Args:
            data_str (str): Raw string received over UDP.

        Returns:
            dict: Parsed data dictionary or None if parsing fails.
        """
        try:
            # Example: {pressure: 32310, vibrations: {1.6:0.211427, 3.1:0.047727, ...}}

            # Replace to make it valid Python
            data_str = data_str.replace('pressure:', '"pressure":')
            data_str = data_str.replace('vibrations:', '"vibrations":')

            # Fix vibrations dict - quote the frequency keys
            import re
            vib_match = re.search(r'"vibrations":\s*\{([^}]+)\}', data_str)
            if vib_match:
                vib_content = vib_match.group(1)
                # Quote all numeric keys: 1.6: -> "1.6":
                vib_content = re.sub(r'(\d+\.?\d*):', r'"\1":', vib_content)
                data_str = data_str.replace(vib_match.group(1), vib_content)

            # Parse
            data = ast.literal_eval(data_str)

            return data

        except Exception as e:
            print(f"⚠️ Parse error: {e}")
            return None

    def _udp_listener_worker(self):
        """
        Background thread that continuously polls the socket for new packets.
        Updates self.latest_pressure and self.latest_vibrations upon receipt.
        """
        print("🌊 UDP listener started")

        while self.collecting:
            try:
                data, addr = self.sock.recvfrom(4096)
                data_str = data.decode('utf-8', errors='ignore')

                parsed = self._parse_sensor_data(data_str)

                if parsed:
                    with self.data_lock:
                        self.latest_pressure = parsed.get('pressure', 0)
                        self.latest_vibrations = parsed.get('vibrations', {})

                        # Track all frequencies
                        self.all_frequencies.update(self.latest_vibrations.keys())

            except socket.timeout:
                pass
            except Exception as e:
                print(f"⚠️ UDP error: {e}")
                time.sleep(0.1)

        print("🛑 UDP listener stopped")

    def collect_sample(self):
        """
        Captures the current state of sensor data, flattening vibration frequencies into
        dynamic dictionary keys (e.g., 'vib_1.6').

        Returns:
            dict: A sample dictionary with timestamp, pressure, aggregate vibration stats,
                  and individual frequency amplitudes.
        """
        try:
            with self.data_lock:
                pressure = self.latest_pressure
                vibrations = dict(self.latest_vibrations)

            # Calculate aggregate stats
            if vibrations:
                vib_values = list(vibrations.values())
                vib_mean = np.mean(vib_values)
                vib_max = max(vib_values)
                vib_min = min(vib_values)
                dominant_freq = max(vibrations, key=vibrations.get)
            else:
                vib_mean = vib_max = vib_min = 0
                dominant_freq = 0

            # Build sample with ALL raw frequencies as separate columns
            sample = {
                'timestamp': datetime.now(),
                'pressure': pressure,
                'vib_mean': vib_mean,
                'vib_max': vib_max,
                'vib_min': vib_min,
                'dominant_freq': float(dominant_freq),
            }

            # Add each frequency as a column: vib_1.6, vib_3.1, etc.
            for freq, amp in vibrations.items():
                sample[f'vib_{freq}'] = amp

            return sample

        except Exception as e:
            print(f"⚠️ Sample error: {e}")
            return None

    def start_collection(self):
        """
        Starts the UDP listener thread and the sample collection loop (approx 10Hz).
        """
        self.collecting = True

        # Start UDP listener
        listener_thread = threading.Thread(target=self._udp_listener_worker, daemon=True)
        listener_thread.start()

        # Start sample collection
        def collection_loop():
            print("🤖 Sensor data collection started")

            while self.collecting:
                sample = self.collect_sample()

                if sample and sample['pressure'] > 0:
                    self.data_buffer.append(sample)
                    self.samples_collected += 1

                    if self.samples_collected == 1:
                        print("\n📊 First sensor sample:")
                        print(f"   Pressure: {sample['pressure']} Pa")
                        print(f"   Frequencies: {len([k for k in sample.keys() if k.startswith('vib_')] ) - 4}")

                time.sleep(0.1)  # 10 Hz

            print("🛑 Sensor collection stopped")

        coll_thread = threading.Thread(target=collection_loop, daemon=True)
        coll_thread.start()

        print(f"✅ Sensor collection started on port {self.udp_port}")

    def stop_collection(self):
        """
        Signals the listener and collector threads to stop.
        """
        self.collecting = False
        time.sleep(1)

    def get_dataframe(self):
        """
        Converts the buffer to a DataFrame. Columns will include all unique vibration
        frequencies seen during the session.

        Returns:
            pd.DataFrame: The sensor data history.
        """
        return pd.DataFrame(list(self.data_buffer))

    def get_latest_vibration_spectrum(self):
        """
        Retrieves the most recent vibration spectrum for real-time visualization.

        Returns:
            dict: Mapping of frequency (float) to amplitude (float).
        """
        with self.data_lock:
            return dict(self.latest_vibrations)

    def save_data(self, filename='sensor_data.parquet', output_dir=None):
        """
        Saves the buffered sensor data to a Parquet file.

        Args:
            filename (str): Name of the file.
            output_dir (str, optional): Target directory.

        Returns:
            str: Path to the saved file.
        """

        df = self.get_dataframe()
        if df.empty:
            print("⚠️ No sensor data to save")
            return None

        # Determine save path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, filename)
        else:
            filepath = filename

        df.to_parquet(filepath)

        vib_cols = [c for c in df.columns if c.startswith('vib_') and c not in ['vib_mean', 'vib_max', 'vib_min']]
        print(f"💾 Saved {len(df)} sensor samples with {len(vib_cols)} frequencies to {os.path.abspath(filepath)}")

        return filepath

    def close(self):
        """
        Stops collection and closes the UDP socket.
        """
        self.collecting = False
        time.sleep(1)
        self.sock.close()
        print("🔌 UDP socket closed")
