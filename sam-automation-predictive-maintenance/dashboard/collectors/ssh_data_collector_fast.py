import paramiko
import pandas as pd
import numpy as np
from datetime import datetime
import time
import ast
from collections import deque
import threading


class RobotDataCollector:
    """
    Manages SSH connection to a robot to collect hardware and joint data via
    continuous ROS topic streams.
    """
    def __init__(self, host, username, password):
        """
        Initializes the SSH client and opens dedicated channels for hardware and joint streams.

        Args:
            host (str): The IP address or hostname of the robot.
            username (str): SSH username.
            password (str): SSH password.
        """
        self.host = host
        self.username = username
        self.password = password

        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(hostname=self.host,
                            username=self.username,
                            password=self.password)

        # Create TWO channels - one for hardware, one for joints
        self.hw_channel = self.client.invoke_shell()
        self.hw_channel.settimeout(0.0)

        self.joint_channel = self.client.invoke_shell()
        self.joint_channel.settimeout(0.0)

        self.data_buffer = deque(maxlen=50000)
        self.collecting = False
        self.samples_collected = 0

        # Latest data from streams
        self.latest_hw_data = {'temperatures': [0] * 8, 'voltages': [0] * 8}
        self.latest_joint_data = {'position': [0] * 6}
        self.hw_lock = threading.Lock()
        self.joint_lock = threading.Lock()

        print(f"✅ SSH Connected to {host}")
        time.sleep(1)

        # Clear initial output
        if self.hw_channel.recv_ready():
            self.hw_channel.recv(4096)
        if self.joint_channel.recv_ready():
            self.joint_channel.recv(4096)

    def _parse_line(self, line):
        """
        Parses a single line of raw text from the SSH output stream.

        Args:
            line (str): A raw string line (e.g., "temperatures: [52.0, 51.0...]").

        Returns:
            tuple: A tuple containing (metric_name, value_list). Returns (None, None) on failure.
        """
        try:
            metric, values = line.split(': ', 1)
            values = ast.literal_eval(values)
            return metric, values
        except Exception as e:
            print(f"⚠️ Parse error: {e}")
            return None, None

    def _hw_stream_worker(self):
        """
        Background thread worker that continuously reads from the hardware status channel.
        Executes 'rostopic echo' and updates self.latest_hw_data in real-time.
        """
        print("🌊 Hardware stream started")

        # Start persistent rostopic echo (no head limit = continuous)
        self.hw_channel.send("rostopic echo /niryo_robot_hardware_interface/hardware_status\n")
        time.sleep(0.5)

        buffer = ""

        while self.collecting:
            if self.hw_channel.recv_ready():
                chunk = self.hw_channel.recv(4096).decode("utf-8", errors="ignore")
                buffer += chunk

                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()

                    if line.startswith('temperatures:') or line.startswith('voltages:'):
                        metric, values = self._parse_line(line)
                        if metric and values:
                            with self.hw_lock:
                                self.latest_hw_data[metric] = values

            time.sleep(0.01)  # Very fast polling

        print("🛑 Hardware stream stopped")

    def _joint_stream_worker(self):
        """
        Background thread worker that continuously reads from the joint states channel.
        Executes 'rostopic echo' and updates self.latest_joint_data in real-time.
        """
        print("🌊 Joint stream started")

        # Start persistent rostopic echo
        self.joint_channel.send("rostopic echo /joint_states\n")
        time.sleep(0.5)

        buffer = ""

        while self.collecting:
            if self.joint_channel.recv_ready():
                chunk = self.joint_channel.recv(4096).decode("utf-8", errors="ignore")
                buffer += chunk

                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()

                    if line.startswith('position:'):
                        metric, values = self._parse_line(line)
                        if metric and values:
                            with self.joint_lock:
                                self.latest_joint_data[metric] = values

            time.sleep(0.01)  # Very fast polling

        print("🛑 Joint stream stopped")

    def collect_sample(self):
        """
        Aggregates the most recent data from hardware and joint streams into a single dictionary.

        Returns:
            dict: A dictionary containing timestamp, joint positions (j1-j6),
                  temperatures, voltages, and error states. Returns None if collection fails.
        """
        try:
            # Get latest data from streams
            with self.hw_lock:
                temps = self.latest_hw_data.get('temperatures', [0] * 8)
                volts = self.latest_hw_data.get('voltages', [0] * 8)

            with self.joint_lock:
                joints = self.latest_joint_data.get('position', [0] * 6)
            motor_temps = temps[:6] if len(temps) >= 6 else [0] * 6
            motor_volts = volts[:6] if len(volts) >= 6 else [0] * 6
            joint_positions = joints[:6] if len(joints) >= 6 else [0] * 6

            sample = {
                'timestamp': datetime.now(),
                'j1': joint_positions[0],
                'j2': joint_positions[1],
                'j3': joint_positions[2],
                'j4': joint_positions[3],
                'j5': joint_positions[4],
                'j6': joint_positions[5],
                'rpi_temp': temps[0] if temps else 0,
                'motor1_temp': motor_temps[0],
                'motor2_temp': motor_temps[1],
                'motor3_temp': motor_temps[2],
                'motor4_temp': motor_temps[3],
                'motor5_temp': motor_temps[4],
                'motor6_temp': motor_temps[5],
                'avg_motor_temp': np.mean(motor_temps) if motor_temps else 0,
                'max_motor_temp': np.max(motor_temps) if motor_temps else 0,
                'motor1_voltage': motor_volts[0],
                'motor2_voltage': motor_volts[1],
                'motor3_voltage': motor_volts[2],
                'motor4_voltage': motor_volts[3],
                'motor5_voltage': motor_volts[4],
                'motor6_voltage': motor_volts[5],
                'avg_motor_voltage': np.mean(motor_volts) if motor_volts else 0,
                'error_state': 0
            }

            return sample

        except Exception as e:
            print(f"⚠️ Collection error: {e}")
            return None

    def start_collection(self, interval=0.1):
        """
        Starts the background stream threads and the main data collection loop.

        Args:
            interval (float): The target time in seconds between data samples (default 0.1s).
        """
        self.collecting = True

        # Start stream workers
        hw_thread = threading.Thread(target=self._hw_stream_worker, daemon=True)
        joint_thread = threading.Thread(target=self._joint_stream_worker, daemon=True)

        hw_thread.start()
        joint_thread.start()

        time.sleep(2)  # Let streams initialize

        # Start collection worker
        def collection_loop():
            print("🤖 Data collection started")
            last_sample_time = 0

            while self.collecting:
                sample = self.collect_sample()

                if sample:
                    # Check if we got real data (not zeros)
                    if sample['avg_motor_temp'] > 0 or sample['avg_motor_voltage'] > 0:
                        self.data_buffer.append(sample)
                        self.samples_collected += 1

                        # Print stats every 100 samples
                        if self.samples_collected % 100 == 0:
                            elapsed = time.time() - last_sample_time if last_sample_time else interval * 100
                            actual_hz = 100 / elapsed if elapsed > 0 else 0
                            print(f"📊 {self.samples_collected} samples | Actual rate: {actual_hz:.1f} Hz")
                            last_sample_time = time.time()

                        if self.samples_collected == 1:
                            print("\n📊 First sample:")
                            print(f"   Temp: {sample['avg_motor_temp']:.1f}°C")
                            print(f"   Voltage: {sample['avg_motor_voltage']:.2f}V")
                            print(f"   Joint 1: {sample['j1']:.3f} rad\n")

                time.sleep(interval)

            print("🛑 Data collection stopped")

        coll_thread = threading.Thread(target=collection_loop, daemon=True)
        coll_thread.start()

        print(f"✅ Started collecting at {1/interval:.1f} Hz target via SSH streams")

    def stop_collection(self):
        """
        Signals the collection loop and stream threads to terminate.
        """
        self.collecting = False
        time.sleep(2)

    def get_dataframe(self):
        """
        Converts the internal deque buffer into a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing all collected samples.
        """
        return pd.DataFrame(list(self.data_buffer))

    def save_data(self, filename='robot_data.parquet', output_dir=None):
        """
        Saves the current data buffer to a Parquet file.

        Args:
            filename (str): The name of the output file.
            output_dir (str, optional): The directory path. If None, uses current working directory.

        Returns:
            str: The absolute path to the saved file.
        """
        import os

        df = self.get_dataframe()
        if df.empty:
            print("⚠️ No data to save")
            return None

        # Determine save path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, filename)
        else:
            filepath = filename  # Save to current working directory

        df.to_parquet(filepath)
        print(f"💾 Saved {len(df)} samples to {os.path.abspath(filepath)}")

        return filepath

    def close(self):
        """
        Stops collection and closes the SSH client connection.
        """
        self.collecting = False
        time.sleep(1)
        self.client.close()
        print("🔌 SSH connection closed")
