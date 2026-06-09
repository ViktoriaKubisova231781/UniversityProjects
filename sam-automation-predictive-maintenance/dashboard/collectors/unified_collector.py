from collectors.ssh_data_collector_fast import RobotDataCollector
from collectors.udp_sensor_collector_raw import UDPSensorCollector
import pandas as pd
import time
from collections import deque
import threading


class UnifiedDataCollector:
    """
    Orchestrator that manages both Robot and Sensor collectors.
    It runs a dedicated thread to merge data streams based on timestamps.
    """
    def __init__(self, robot_ip, ssh_username, ssh_password, udp_port=5005):
        """
        Initializes both the RobotDataCollector and UDPSensorCollector.

        Args:
            robot_ip (str): Robot IP address.
            ssh_username (str): SSH login username.
            ssh_password (str): SSH login password.
            udp_port (int): Local UDP port for sensor data.
        """
        print("🚀 Initializing unified collector...\n")

        print("1️⃣ Connecting to robot via SSH...")
        self.robot_collector = RobotDataCollector(robot_ip, ssh_username, ssh_password)

        print("\n2️⃣ Starting sensor listener...")
        self.sensor_collector = UDPSensorCollector(udp_port=udp_port)

        # Track which samples we've already merged
        self.last_merged_robot_idx = 0
        self.last_merged_sensor_idx = 0

        # Unified data storage
        self.unified_buffer = deque(maxlen=50000)
        self.collecting = False
        self.samples_collected = 0

        print("\n✅ Unified collector ready!")

    def _merge_worker(self):
        """
        Background thread that performs an as-of merge between robot and sensor dataframes.
        It uses a 1-second tolerance to align the nearest sensor sample to each new robot sample.
        """
        print("🔄 Merge thread started")

        while self.collecting:
            try:
                robot_df = self.robot_collector.get_dataframe()
                sensor_df = self.sensor_collector.get_dataframe()

                # Only merge if we have new data from BOTH sources
                if len(robot_df) > self.last_merged_robot_idx and len(sensor_df) > self.last_merged_sensor_idx:
                    # Get only NEW robot samples
                    new_robot = robot_df.iloc[self.last_merged_robot_idx:].copy()

                    # Get all sensor samples (for matching)
                    all_sensor = sensor_df.copy()

                    if not new_robot.empty and not all_sensor.empty:
                        # Merge new robot samples with nearest sensor data
                        merged = pd.merge_asof(
                            new_robot.sort_values('timestamp'),
                            all_sensor.sort_values('timestamp'),
                            on='timestamp',
                            direction='nearest',
                            tolerance=pd.Timedelta('1s')  # Tighter tolerance
                        )

                        # Add merged rows to unified buffer
                        for _, row in merged.iterrows():
                            self.unified_buffer.append(row.to_dict())
                            self.samples_collected += 1

                        # Update tracking
                        self.last_merged_robot_idx = len(robot_df)

                        if self.samples_collected % 100 == 0:
                            print(f"📊 Unified: {self.samples_collected} samples "
                                  f"(Robot: {len(robot_df)}, Sensor: {len(sensor_df)})")

                time.sleep(1)  # Merge every second

            except Exception as e:
                print(f"⚠️ Merge error: {e}")
                time.sleep(1)

        print("🛑 Merge thread stopped")

    def start_collection(self, robot_interval=0.2):
        """
        Starts all child collectors and the merge worker thread.

        Args:
            robot_interval (float): Sampling interval for the robot collector.
        """
        self.collecting = True

        # Start robot collection
        self.robot_collector.start_collection(interval=robot_interval)

        # Start sensor collection
        self.sensor_collector.start_collection()

        # Start merge worker
        merge_thread = threading.Thread(target=self._merge_worker, daemon=True)
        merge_thread.start()

        print("\n✅ Unified collection started")

    def stop_collection(self):
        """
        Stops all collectors (robot, sensor, and merge worker).
        """
        self.collecting = False
        self.robot_collector.stop_collection()
        self.sensor_collector.stop_collection()
        time.sleep(2)
        print("🛑 Unified collection stopped")

    def get_robot_dataframe(self):
        """
        Returns the raw data collected from the robot (unmerged).
        """
        return self.robot_collector.get_dataframe()

    def get_sensor_dataframe(self):
        """
        Returns the raw data collected from the UDP sensors (unmerged).
        """
        return self.sensor_collector.get_dataframe()

    def get_unified_dataframe(self):
        """
        Returns the fully merged dataset containing synchronized robot and sensor data.
        """
        return pd.DataFrame(list(self.unified_buffer))

    def save_data(self, filename_prefix='data', output_dir=None, script_path=None):
        """
        Saves three files (Unified, Robot-only, Sensor-only) into a timestamped subdirectory.

        Args:
            filename_prefix (str): Prefix for all generated files.
            output_dir (str, optional): Custom base directory.
            script_path (str, optional): Path to the calling script to determine relative 'data' folder.

        Returns:
            str: The directory path where files were saved.
        """
        import os

        timestamp = time.strftime('%Y%m%d_%H%M%S')

        # Determine base directory
        if output_dir is None:
            if script_path:
                # Save relative to where the script is located
                script_dir = os.path.dirname(os.path.abspath(script_path))
                base_dir = os.path.join(script_dir, 'data')
            else:
                # Fallback to current working directory
                base_dir = os.path.join(os.getcwd(), 'data')
        else:
            base_dir = output_dir

        # Create timestamped subfolder
        session_dir = os.path.join(base_dir, timestamp)
        os.makedirs(session_dir, exist_ok=True)

        print(f"📁 Saving to: {os.path.abspath(session_dir)}\n")

        # Save UNIFIED file
        unified_df = self.get_unified_dataframe()
        if not unified_df.empty:
            unified_filename = f'{filename_prefix}_unified_{timestamp}.parquet'
            unified_filepath = os.path.join(session_dir, unified_filename)

            unified_df.to_parquet(unified_filepath)

            vib_cols = [c for c in unified_df.columns if c.startswith('vib_') and c not in ['vib_mean', 'vib_max', 'vib_min']]

            print(f"💾 UNIFIED: {unified_filename}")
            print(f"   • Samples: {len(unified_df)}")
            print(f"   • Total columns: {len(unified_df.columns)}")
            print("   • Robot metrics: temps, voltages, joints")
            print(f"   • Sensor: pressure + {len(vib_cols)} vibration frequencies")
        else:
            print("⚠️ No unified data to save")

        # Save robot backup
        robot_df = self.get_robot_dataframe()
        if not robot_df.empty:
            robot_filename = f'{filename_prefix}_robot_{timestamp}.parquet'
            robot_filepath = os.path.join(session_dir, robot_filename)
            robot_df.to_parquet(robot_filepath)
            print(f"\n💾 Robot backup: {robot_filename} ({len(robot_df)} samples)")

        # Save sensor backup
        sensor_df = self.get_sensor_dataframe()
        if not sensor_df.empty:
            sensor_filename = f'{filename_prefix}_sensor_{timestamp}.parquet'
            sensor_filepath = os.path.join(session_dir, sensor_filename)
            sensor_df.to_parquet(sensor_filepath)

            vib_cols = [c for c in sensor_df.columns if c.startswith('vib_') and c not in ['vib_mean', 'vib_max', 'vib_min']]
            print(f"💾 Sensor backup: {sensor_filename} ({len(sensor_df)} samples, {len(vib_cols)} frequencies)")

        print(f"\n✅ All files saved to: {os.path.abspath(session_dir)}")

        return session_dir

    def close(self):
        """
        Stops all processes and closes all connections (SSH and UDP).
        """
        self.stop_collection()
        self.robot_collector.close()
        self.sensor_collector.close()
        print("🔌 All connections closed")
