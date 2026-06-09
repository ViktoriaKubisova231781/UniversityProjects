# views/training.py
import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime
from services.state_manager import get_manager


def render_training_view():
    """
    Renders the Model Training Studio interface.

    Workflow:
    1.  **Source Selection**: Users choose between using the active in-memory buffer
        (from the current live session) or loading a historical `.parquet` file.
    2.  **Configuration**: Users set the number of training epochs and select
        specific features (columns) to train on.
    3.  **Execution**: Initiates the background training process via `AsyncTrainer`.
    4.  **Tuning**: Once training completes, users can fine-tune the anomaly detection
        thresholds (Warning/Critical levels) before saving.
    5.  **Persistence**: Saves the model to disk or activates it immediately in memory.
    """
    manager = get_manager()

    # Check for updates from the background process
    manager.update_system_state()

    st.markdown("## 🎓 Model Training Studio")

    # 1. Data Selection
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("1. Source Data")
        data_source = st.radio("Input Source", ["📂 Browse Files", "📊 Live Buffer"], horizontal=True)

    df = None

    if data_source == "📊 Live Buffer":
        if manager.collector and not manager.collector.get_unified_dataframe().empty:
            df = manager.collector.get_unified_dataframe()
            st.success(f"✅ Loaded {len(df)} samples from active session.")
        else:
            st.warning("⚠️ Live buffer is empty. Connect to robot or load a file.")

    else:
        # File Scan Logic
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_script_dir)
        search_root = os.path.dirname(project_root)

        parquet_files = []
        for root, dirs, files in os.walk(search_root):
            for file in files:
                if file.endswith(".parquet"):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, search_root)
                    try:
                        size_kb = os.path.getsize(full_path) / 1024
                        mod_time = datetime.fromtimestamp(os.path.getmtime(full_path))
                        parquet_files.append({
                            "path": full_path, "display": f"{rel_path} ({size_kb:.0f} KB)",
                            "name": file, "timestamp": mod_time
                        })
                    except Exception as e:
                        print(f"Error accessing file {full_path}: {e}")

        if parquet_files:
            parquet_files.sort(key=lambda x: x["timestamp"], reverse=True)
            filter_type = st.radio("Filter", ["All", "Unified", "Robot", "Sensor"], horizontal=True)

            filtered = parquet_files
            if filter_type != "All":
                filtered = [f for f in parquet_files if filter_type.lower() in f["name"]]

            if not filtered:
                st.warning("No files match filter.")
            else:
                sel = st.selectbox("Select File", filtered, format_func=lambda x: f"{x['display']}")
                if sel:
                    try:
                        df = pd.read_parquet(sel["path"])
                        st.info(f"📄 Loaded: {sel['name']} ({len(df)} rows)")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.warning("No .parquet files found.")

    if df is None:
        return

    st.divider()

    # 2. Configuration
    st.subheader("2. Model Configuration")
    c1, c2 = st.columns(2)
    with c1:
        epochs = st.number_input("Epochs", 5, 200, 20)

    with c2:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        valid = [c for c in num_cols if c not in ['timestamp', 'index', 'Unnamed: 0']]
        defaults = [c for c in valid if any(x in c for x in ['motor', 'vib_mean'])]
        features = st.multiselect("Features", valid, default=defaults[:6])

    if not features:
        st.error("Select features.")
        return

    st.divider()

    # 3. Execution
    if manager.trainer.is_training:
        st.info(f"⏳ {manager.trainer.status_message}")
        st.progress(manager.trainer.progress)
        time.sleep(1)
        st.rerun()

    else:
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            manager.trainer.start_training(df, features, epochs)
            st.rerun()

        if manager.trainer.result_model:
            st.success("✅ Training Complete!")

            # --- FEATURE ADDED: THRESHOLD MODIFICATION ---
            with st.expander("🎛️ Tune Thresholds", expanded=True):
                st.info("Adjust the sensitivity for each feature before saving.")

                # We iterate over a copy of the keys to avoid runtime modification issues
                for feat in list(manager.trainer.result_model.threshold_dict.keys()):
                    current_warn, current_crit = manager.trainer.result_model.threshold_dict[feat]

                    c_name, c_warn, c_crit = st.columns([2, 1, 1])
                    c_name.markdown(f"**{feat}**")

                    # Number Inputs for Warning (Yellow) and Critical (Red)
                    new_warn = c_warn.number_input(
                        f"⚠️ Warning ({feat})",
                        value=float(current_warn),
                        format="%.4f",
                        key=f"w_{feat}"
                    )
                    new_crit = c_crit.number_input(
                        f"🚨 Critical ({feat})",
                        value=float(current_crit),
                        format="%.4f",
                        key=f"c_{feat}"
                    )

                    # Update the model in memory immediately
                    manager.trainer.result_model.threshold_dict[feat] = (new_warn, new_crit)
            # ---------------------------------------------

            # Save Section
            st.divider()
            st.subheader("💾 Save Model")
            c_name, c_btn = st.columns([3, 1])
            with c_name:
                save_name = st.text_input("Model Name", value=f"model_{datetime.now().strftime('%Y%m%d_%H%M')}")
            with c_btn:
                st.write("")
                st.write("")
                if st.button("Save to Disk", type="primary", use_container_width=True):
                    manager.model = manager.trainer.result_model
                    success, msg = manager.save_model(save_name)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

            # Activation Section
            st.divider()
            c_act, c_disc = st.columns(2)
            with c_act:
                if st.button("🚀 Activate Immediately", use_container_width=True):
                    manager.model = manager.trainer.result_model
                    manager.trainer.result_model = None
                    st.toast("Model Activated!")
            with c_disc:
                if st.button("🗑️ Discard", use_container_width=True):
                    manager.trainer.result_model = None
                    st.rerun()

        elif manager.trainer.error:
            st.error(f"❌ Training Failed: {manager.trainer.error}")
