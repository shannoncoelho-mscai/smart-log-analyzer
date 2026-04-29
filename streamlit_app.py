"""
Smart Log Analyzer - Streamlit Dashboard
"""

import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 🔥 FIXED (important)
API_URL = "http://127.0.0.1:5000"

st.set_page_config(
    page_title="Smart Log Analyzer",
    page_icon="🔍",
    layout="wide",
)

# ─── Header ─────────────────────────────────────────
st.title("🔍 Smart Log Analyzer")
st.markdown("**Predictive Maintenance using Machine Learning & MLOps**")
st.markdown("---")

# ─── Sidebar ────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Parameters")

    air_temp  = st.slider("Air Temperature (K)", 295.0, 305.0, 298.1, 0.1)
    proc_temp = st.slider("Process Temperature (K)", 305.0, 315.0, 308.6, 0.1)
    rpm       = st.slider("Rotational Speed (RPM)", 1168, 2886, 1551)
    torque    = st.slider("Torque (Nm)", 3.8, 76.6, 42.8, 0.1)
    tool_wear = st.slider("Tool Wear (min)", 0, 253, 0)

    predict_btn = st.button("🔮 Predict Failure", use_container_width=True)

# ─── Derived Features ───────────────────────────────
temp_delta  = proc_temp - air_temp
power_proxy = torque * (rpm * 2 * 3.14159 / 60)
wear_stress = tool_wear * torque

col1, col2, col3 = st.columns(3)
col1.metric("🌡️ Temp Delta", f"{temp_delta:.2f} K")
col2.metric("⚡ Power Proxy", f"{power_proxy:.1f}")
col3.metric("🔧 Wear Stress", f"{wear_stress:.1f}")

# ─── Prediction ─────────────────────────────────────
if predict_btn:
    payload = {
        "air_temperature": air_temp,
        "process_temperature": proc_temp,
        "rotational_speed": rpm,
        "torque": torque,
        "tool_wear": tool_wear,
    }

    try:
        res = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        result = res.json()

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        c1, c2, c3 = st.columns(3)

        label = result["label"]
        icon  = "✅" if label == "NORMAL" else "🚨"

        c1.metric("Status", f"{icon} {label}")
        c2.metric("Failure %", f"{result['failure_probability']:.2f}")
        c3.metric("Risk", result["risk_level"])

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["failure_probability"],
            title={'text': "Failure Probability"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

    except:
        st.error("Cannot connect to API")

# ─── Radar ─────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Parameter Radar")

norm_vals = [
    (air_temp - 295)/10,
    (proc_temp - 305)/10,
    (rpm-1168)/(2886-1168),
    (torque-3.8)/(76.6-3.8),
    tool_wear/253,
]

fig = go.Figure(go.Scatterpolar(
    r=norm_vals + [norm_vals[0]],
    theta=["Air","Proc","RPM","Torque","Wear","Air"],
    fill='toself'
))
st.plotly_chart(fig)

# ─── 🔥 NEW: Model Info (MLflow output) ─────────────
st.markdown("---")
st.subheader("📊 Model Info")

try:
    with open("models/model_meta.json") as f:
        meta = json.load(f)

    st.write("**Model:**", meta["model_name"])
    st.write("**F1 Score:**", round(meta["metrics"]["f1"], 4))

except:
    st.warning("Model metadata not found")

# ─── 🔥 NEW: Monitoring (PSI + Alerts) ──────────────
st.markdown("---")
st.subheader("📡 Monitoring Status")

try:
    with open("models/monitoring_log.json") as f:
        logs = json.load(f)

    latest = logs[-1]

    st.write("**Accuracy:**", latest["accuracy"])
    st.write("**F1 Score:**", latest["f1_score"])

    if latest["alerts"]:
        st.error("⚠ Alerts detected")
        for a in latest["alerts"]:
            st.write("-", a)
    else:
        st.success("No issues detected")

except:
    st.warning("Run monitoring first")

# ─── Footer ────────────────────────────────────────
st.markdown("---")
st.caption("Smart Log Analyzer · MSC AI · 2026")