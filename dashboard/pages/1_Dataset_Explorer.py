"""
Dashboard Page 1 — Dataset Explorer
Allows users to visually inspect the telemetry data and injected faults.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.utils import inject_css, load_telemetry, FAULT_COLORS, PLOTLY_TEMPLATE

st.set_page_config(page_title="Dataset Explorer", page_icon=None, layout="wide", initial_sidebar_state="collapsed")
inject_css()

st.markdown("""
<h1 style='font-size:1.8rem; font-weight:800; color:#F1F5F9; margin-bottom:4px;'>
   Dataset Explorer
</h1>
<p style='color:#64748B; margin:0 0 24px 0;'>
  Inspect raw telemetry from the simulated network. See how different anomalies manifest in the metrics.
</p>
""", unsafe_allow_html=True)

df = load_telemetry()

# Controls
c1, c2 = st.columns(2)
dev = c1.selectbox("Select Device", sorted(df['device_id'].unique()))
intfs = sorted(df[df['device_id']==dev]['interface_id'].unique())
intf = c2.selectbox("Select Interface", intfs)

sub = df[(df['device_id']==dev) & (df['interface_id']==intf)].copy()
sub = sub.sort_values('timestamp')

st.markdown("<br>", unsafe_allow_html=True)

def plot_metric(data, y_col, title, color):
    fig = go.Figure()
    
    # Plot normal traffic
    fig.add_trace(go.Scatter(
        x=data['timestamp'], y=data[y_col],
        mode='lines', line=dict(color=color, width=1.5),
        name=y_col
    ))
    
    # Highlight anomalies
    anomalies = data[data['anomaly_type'] != 'none']
    if len(anomalies) > 0:
        for fault_type in anomalies['anomaly_type'].unique():
            fault_data = anomalies[anomalies['anomaly_type'] == fault_type]
            fig.add_trace(go.Scatter(
                x=fault_data['timestamp'], y=fault_data[y_col],
                mode='markers', marker=dict(color=FAULT_COLORS.get(fault_type, '#EF4444'), size=5),
                name=f"Anomaly: {fault_type}"
            ))

    fig.update_layout(PLOTLY_TEMPLATE['layout'])
    fig.update_layout(
        title=title,
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(0,0,0,0)')
    )
    return fig

st.plotly_chart(plot_metric(sub, 'ifInOctets', 'Ingress Traffic (Bytes/sec)', '#6366F1'), use_container_width=True)
st.plotly_chart(plot_metric(sub, 'latency_ms', 'Latency (ms)', '#F97316'), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(plot_metric(sub, 'ifInErrors', 'Ingress Errors', '#EAB308'), use_container_width=True)
with col2:
    st.plotly_chart(plot_metric(sub, 'ifInDiscards', 'Ingress Discards', '#EC4899'), use_container_width=True)

st.markdown("#### Raw Data Preview")
st.dataframe(sub.tail(100), use_container_width=True, height=300)
