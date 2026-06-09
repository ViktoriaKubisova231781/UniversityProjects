import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
from services.state_manager import get_manager
from datetime import datetime


def render_history_view():
    """
    Renders the History & Backtesting dashboard view.

    This view performs the following functions:
    1.  **File Scanning**: Recursively scans the project directory for `.parquet` data files.
    2.  **Data Visualization**: Loads selected historical files and plots raw telemetry data.
    3.  **Backtesting**: Allows the user to run the currently loaded anomaly detection model
        against the historical file to simulate how the model would have reacted.
    4.  **Anomaly Visualization**: Plots reconstruction errors against configured thresholds,
        highlighting specific timestamps where anomalies occurred.
    """
    manager = get_manager()
    st.markdown("## 📂 Data History & Backtesting")

    df = None

    # --- FILE SCAN LOGIC ---
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

                    if 'timestamp' not in df.columns:
                        if isinstance(df.index, pd.DatetimeIndex):
                            df['timestamp'] = df.index
                        else:
                            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='0.4S')

                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.warning("No .parquet files found.")

    if df is None:
        return

    # 2. Visualization
    st.subheader("📊 Data Viewer")
    numeric = df.select_dtypes(include=['number']).columns.tolist()

    if not numeric:
        st.warning("No numeric data found in this file.")
    else:
        defaults = [c for c in numeric if 'temp' in c or 'vib' in c][:3]
        if not defaults:
            defaults = numeric[:2]
        view_cols = st.multiselect("Plot Columns", numeric, default=defaults)

        if view_cols:
            fig = go.Figure()
            for col in view_cols:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df[col],
                    name=col,
                    hovertemplate='%{y:.2f}'
                ))

            fig.update_layout(
                height=400,
                hovermode="x unified",
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left")
            )
            st.plotly_chart(fig, use_container_width=True)

    # 3. BACKTESTING / INFERENCE
    st.divider()
    st.subheader("🧠 Model Backtesting")

    if not manager.model:
        st.warning("⚠️ No model loaded. Please load a model in the Live View sidebar first.")
    else:
        if st.button("▶️ Run Inference on this File"):
            with st.spinner("Running inference..."):
                req_cols = list(set(manager.model.columns) | {'timestamp'})

                missing = [c for c in req_cols if c not in df.columns and c != 'timestamp']
                if missing:
                    st.error(f"Cannot run model. Missing columns in file: {missing}")
                else:
                    inference_df = df[req_cols].copy()

                    try:
                        errs, preds = manager.model.per_column_inference(
                            inference_df,
                            return_preds=True,
                            batch_mode=True
                        )

                        st.success("Inference Complete!")

                        if not errs:
                            st.warning("No anomalies calculated (data might be too short).")
                        else:
                            tabs = st.tabs(list(errs.keys()))
                            for i, feat in enumerate(errs.keys()):
                                with tabs[i]:
                                    thresh_y, thresh_r = manager.model.threshold_dict.get(feat, (0, 0))
                                    error_series = errs[feat]

                                    # --- CRITICAL FIX: RE-ALIGN TIMESTAMPS ---
                                    # Instead of slicing the end of the original timestamps (which fails if resampling changed the count),
                                    # we generate a linear time range from Start to End of the file to match the error count.
                                    start_ts = df['timestamp'].iloc[0]
                                    end_ts = df['timestamp'].iloc[-1]

                                    # Create evenly spaced timestamps covering the file duration
                                    x_axis = pd.date_range(start=start_ts, end=end_ts, periods=len(error_series))
                                    # -----------------------------------------

                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(x=x_axis, y=error_series, name="Error", line=dict(color='blue')))
                                    fig.add_hline(y=thresh_r, line_color='red', line_dash='dash', name='Critical')
                                    fig.add_hline(y=thresh_y, line_color='orange', line_dash='dot', name='Warning')

                                    anomalies = [e for e in error_series if e > thresh_r]
                                    if anomalies:
                                        st.error(f"Found {len(anomalies)} critical anomalies!")
                                    else:
                                        st.success("No anomalies detected.")

                                    st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Inference failed: {str(e)}")
