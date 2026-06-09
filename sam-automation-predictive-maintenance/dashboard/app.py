import streamlit as st
import multiprocessing
import os
from dotenv import load_dotenv  # pip install python-dotenv

# 1. Load Environment Variables directly
load_dotenv()

# --- CRITICAL FIX FOR TENSORFLOW MULTIPROCESSING ---
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import time
from services.state_manager import get_manager
from views import live, training, history

st.set_page_config(page_title="Niryo Commander", page_icon="🤖", layout="wide")

manager = get_manager()

with st.sidebar:
    st.title("🤖 Niryo System")
    st.caption(f"Status: {'🟢 Online' if manager.is_connected else '🔴 Offline'}")

    mode = st.radio("Navigation", ["Live Monitor", "Model Training", "Data History"])
    st.divider()

    # CONNECTION SETTINGS
    # We now pull defaults directly from os.getenv(), using your .env values
    if not manager.is_connected:
        with st.expander("🔌 Connection Settings", expanded=True):
            # Default to "169.254.200.200" if ROBOT_IP is missing from .env
            default_ip = os.getenv("ROBOT_IP", "169.254.200.200")
            default_user = os.getenv("SSH_USER", "niryo")
            default_pass = os.getenv("SSH_PASS", "robotics")
            default_port = int(os.getenv("UDP_PORT", 5005))

            ip = st.text_input("IP Address", value=default_ip)
            user = st.text_input("SSH User", value=default_user)
            pwd = st.text_input("SSH Pass", value=default_pass, type="password")

            if st.button("Connect Now", use_container_width=True):
                # Pass the raw values to the manager
                manager.connect_robot(ip, user, pwd, default_port)
                st.rerun()
    else:
        if st.button("Disconnect", use_container_width=True):
            manager.disconnect_robot()
            st.rerun()

# --- ROUTER ---
if mode == "Live Monitor":
    live.render_live_view()

    # Auto-refresh loop for Live View
    if manager.is_connected and manager.collector and manager.collector.collecting:
        time.sleep(float(os.getenv("REFRESH_RATE", 3.0)))
        st.rerun()

elif mode == "Model Training":
    training.render_training_view()

elif mode == "Data History":
    history.render_history_view()
