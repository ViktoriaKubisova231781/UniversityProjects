# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Robot Defaults
ROBOT_IP = os.getenv("ROBOT_IP", "169.254.200.200")
SSH_USER = os.getenv("SSH_USER", "niryo")
SSH_PASS = os.getenv("SSH_PASS", "robotics")  # Ideally set this in .env
UDP_PORT = int(os.getenv("UDP_PORT", 5005))

# System Settings
MAX_BUFFER_SIZE = 5000  # Number of points to keep in RAM for live plotting
REFRESH_RATE = 3.0      # UI update frequency in seconds
