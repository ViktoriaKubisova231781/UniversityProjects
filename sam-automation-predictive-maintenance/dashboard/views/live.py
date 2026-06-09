import streamlit as st
import plotly.graph_objects as go
from services.state_manager import get_manager


def render_live_view():
    """
    Renders the main Live Monitor dashboard.

    Features:
    - **Sidebar**: Model loading and management.
    - **Alert System**: Displays and manages (acknowledges/dismisses) active critical and warning alerts.
    - **Controls**: Start/Stop data collection, frequency adjustment, and toggle inference.
    - **Metrics**: Real-time display of key indicators (buffer size, motor temps, anomaly risk).
    - **Visualizations**: Tabbed charts for Temperature, Voltage, Joints, Sensors, and Real-time Inference scoring.
    """
    manager = get_manager()
    manager.update_system_state()

    # --- SIDEBAR MODEL LOADER ---
    with st.sidebar:
        st.divider()
        st.subheader("📁 Model Management")
        models = manager.get_available_models()
        if models:
            sel_model = st.selectbox("Load Saved Model", models, index=None, placeholder="Select model...")
            if sel_model:
                if st.button(f"Load '{sel_model}'", use_container_width=True):
                    success, msg = manager.load_model(sel_model)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
        else:
            st.caption("No saved models found.")

    # --- ALERT BANNER ---
    active_alerts = manager.get_active_alerts()
    if active_alerts:
        crit = sum(1 for a in active_alerts if a['severity'] == 'critical')
        if crit > 0:
            st.error(f"🚨 {crit} CRITICAL ANOMALIES DETECTED")

        with st.expander(f"🔔 Active Alerts ({len(active_alerts)})", expanded=True):
            h1, h2, h3, h4 = st.columns([1, 3, 1, 1])
            h1.caption("Time")
            h2.caption("Message")
            h3.caption("Value")
            h4.caption("Action")

            for i, alert in enumerate(reversed(active_alerts)):
                ts_key = str(alert['timestamp'].timestamp())
                btn_key = f"ack_{ts_key}_{i}"
                c1, c2, c3, c4 = st.columns([1, 3, 1, 1])
                c1.write(f"`{alert['timestamp'].strftime('%H:%M:%S')}`")

                icon = "🔴" if alert['severity'] == 'critical' else "🟡"
                c2.markdown(f"**{icon} {alert['message']}**")

                # Value is already formatted with unit in state_manager
                c3.write(f"{alert['value']}")

                if c4.button("✅", key=btn_key):
                    alert['acknowledged'] = True
                    st.rerun()
            if st.button("🗑️ Dismiss All"):
                for a in active_alerts:
                    a['acknowledged'] = True
                st.rerun()

    st.markdown("### 🤖 Live Monitor")

    if not manager.is_connected:
        st.info("👈 Please connect to the robot in the sidebar to begin.")
        return

    # --- CONTROLS ---
    with st.container():
        c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1.5])
        with c1:
            hz = st.slider("Hz", 1.0, 10.0, 2.5, 0.5, disabled=manager.collector.collecting)
        with c2:
            if manager.collector.collecting:
                if st.button("⏹️ Stop", type="primary", use_container_width=True):
                    manager.collector.stop_collection()
                    st.rerun()
            else:
                if st.button("▶️ Start", type="primary", use_container_width=True):
                    manager.collector.start_collection(robot_interval=1.0 / hz)
                    st.rerun()
        with c3:
            disabled = manager.model is None
            label = "🔮 Inference" if not disabled else "🔮 (No Model)"
            manager.inference_active = st.toggle(label, value=manager.inference_active, disabled=disabled)
        with c4:
            if st.button("💾 Save Data", use_container_width=True):
                path = manager.collector.save_data()
                if path:
                    st.toast(f"Saved to {path}")

    # --- METRICS ---
    df = manager.collector.get_unified_dataframe()
    if df.empty:
        st.warning("Waiting for data stream...")
        return

    last = df.iloc[-1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Buffer Size", len(df))

    # Motor Temp with Unit
    temp_val = last.get('avg_motor_temp', 0)
    temp_unit = manager.get_unit('avg_motor_temp')
    m2.metric("Motor Temp", f"{temp_val:.1f}{temp_unit}")

    if manager.inference_active and manager.recent_predictions:
        last_pred = manager.recent_predictions[-1]
        max_err = 0
        for col, err in last_pred['errors'].items():
            if col in manager.model.threshold_dict:
                _, red = manager.model.threshold_dict[col]
                if red > 0:
                    max_err = max(max_err, err / red)
        lbl = "Normal" if max_err < 1.0 else "ANOMALY"
        color = "normal" if max_err < 1.0 else "inverse"
        m4.metric("Status", lbl, f"{max_err:.0%} Risk", delta_color=color)
    else:
        m4.metric("Status", "Standby")

    # --- VISUALIZATIONS ---
    tabs = st.tabs(["📈 Temps", "⚡ Volts", "🔧 Joints", "📡 Sensors", "🧠 Inference"])

    with tabs[0]:
        st.line_chart(df[[c for c in df.columns if 'temp' in c]].tail(200))

    with tabs[1]:
        volts = [c for c in df.columns if 'voltage' in c]
        if volts:
            st.line_chart(df[volts].tail(200))
        else:
            st.info("No voltage data")

    with tabs[2]:
        joints = [f'j{i}' for i in range(1, 7)]
        available_joints = [j for j in joints if j in df.columns]
        if available_joints:
            cols = st.columns(3)
            for i, joint in enumerate(available_joints):
                with cols[i % 3]:
                    st.caption(f"**Joint {joint.upper()} ({manager.get_unit(joint)})**")
                    st.line_chart(df[joint].tail(200), height=200)
        else:
            st.info("No joint data")

    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            st.caption(f"Vibration Spectrum ({manager.get_unit('vib')})")
            vib_cols = [c for c in df.columns if 'vib_' in c and 'mean' not in c and 'max' not in c]
            if vib_cols:
                st.bar_chart(df[vib_cols].iloc[-1])
        with c2:
            st.caption(f"Pressure ({manager.get_unit('pressure')})")
            if 'pressure' in df.columns:
                st.line_chart(df['pressure'].tail(200))

    with tabs[4]:
        if not manager.inference_active or not manager.recent_predictions:
            st.info("Enable inference to see analysis.")
        else:
            recent = list(manager.recent_predictions)[-50:]
            valid_feats = list(recent[0]['errors'].keys())
            if valid_feats:
                sel_feat = st.selectbox("Select Feature", valid_feats)
                unit = manager.get_unit(sel_feat)  # Get unit for selected feature

                times = [p['timestamp'] for p in recent]
                errs = [p['errors'][sel_feat] for p in recent]

                thresh_y, thresh_r = 0, 0
                if sel_feat in manager.model.threshold_dict:
                    thresh_y, thresh_r = manager.model.threshold_dict[sel_feat]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=times, y=errs, name=f"Error ({unit})", line=dict(color='blue')))
                if thresh_r > 0:
                    fig.add_hline(y=thresh_r, line_color='red', line_dash='dash', annotation_text='Critical')
                if thresh_y > 0:
                    fig.add_hline(y=thresh_y, line_color='orange', line_dash='dot', annotation_text='Warning')
                fig.update_layout(height=350, margin=dict(t=30, b=0, l=0, r=0),
                                  title=f"{sel_feat} - Anomaly Score ({unit})")
                st.plotly_chart(fig, use_container_width=True)
