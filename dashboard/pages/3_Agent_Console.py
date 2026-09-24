"""
Dashboard Page 3 — Agent Console
No API key in UI — reads GOOGLE_API_KEY from environment.
"""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

from dashboard.utils import inject_css, load_telemetry, load_rca_results

st.set_page_config(page_title="Agent Console", page_icon=None, layout="wide", initial_sidebar_state="collapsed")
inject_css()

st.markdown("""
<h1 style='font-size:1.8rem; font-weight:800; color:#F1F5F9; margin-bottom:4px;'>
   Tier 3 — Agent Console
</h1>
<p style='color:#64748B; margin:0 0 24px 0;'>
  LangGraph ReAct Agent investigates alerts using 6 diagnostic tools and produces evidence-grounded RCA reports.
</p>
""", unsafe_allow_html=True)

api_key = os.environ.get("GOOGLE_API_KEY", "")
if not api_key:
    st.error("**GOOGLE_API_KEY not set.** Launch with: `GOOGLE_API_KEY=your_key streamlit run dashboard/app.py`")

df = load_telemetry()
rca_results = load_rca_results()

tab_live, tab_saved = st.tabs([" Live Investigation", " Saved Reports"])

with tab_live:
    st.markdown("**Select an alert to investigate:**")
    col_dev, col_intf, col_fault, col_mse = st.columns(4)

    devices = sorted(df['device_id'].unique())
    dev = col_dev.selectbox("Device", devices)
    intfs = sorted(df[df['device_id']==dev]['interface_id'].unique())
    intf = col_intf.selectbox("Interface", intfs)
    fault_opt = col_fault.selectbox("Anomaly Type", ['link_failure','congestion','mtu_mismatch','interface_flap','routing_loop','unknown'])
    mse_val = col_mse.number_input("LSTM MSE Score", value=0.141, step=0.001, format="%.3f")

    btn = st.button(" Run Investigation", type="primary", disabled=(not api_key))

    if btn and api_key:
        with st.spinner(f"Agent investigating {dev}/{intf} …"):
            try:
                import torch, joblib, numpy as np
                from agent.tools.diagnostic_tools import ALL_TOOLS, inject_network_state
                from agent.react_agent import run_rca_investigation
                from agent.prompts import RCA_SYSTEM_PROMPT
                from agent.run_agent import get_llm, build_network_state
                from network.topology import NetworkTopology

                if 'ns_ready' not in st.session_state:
                    rf = joblib.load('data/models/random_forest.pkl')
                    lstm = torch.load('data/models/lstm_autoencoder.pth', map_location='cpu', weights_only=False)
                    lstm.device = torch.device('cpu')
                    lstm.model.to('cpu')
                    _, sc = lstm.predict(df)
                    topo = NetworkTopology()
                    topo.build_hierarchical_wan()
                    build_network_state(df, [], topo, sc, df)
                    st.session_state['ns_ready'] = True

                alert = {
                    'device_id': dev, 'interface_id': intf, 'anomaly_type': fault_opt,
                    'mse_score': f"{mse_val:.6f}", 'tier': 2, 'timestamp': str(df['timestamp'].max()),
                }
                result = run_rca_investigation(
                    alert=alert, tools=ALL_TOOLS, llm=get_llm(),
                    system_prompt=RCA_SYSTEM_PROMPT, verbose=False,
                )
                st.session_state['last_result'] = result
                st.session_state['last_alert'] = alert

            except Exception as e:
                st.error(f"Agent error: {e}")

    if 'last_result' in st.session_state:
        res = st.session_state['last_result']
        alert = st.session_state['last_alert']
        st.success(f"Investigation complete — {res['steps']} tool call(s) | Status: `{res['status']}`")
        st.divider()

        st.markdown("#####  Diagnostic Tool Trace")
        for i, step in enumerate(res['trajectory'], 1):
            args_str = json.dumps(step['arguments'])
            obs = step['observation']
            if isinstance(obs, str):
                try: obs = json.loads(obs)
                except Exception: pass
            with st.expander(f"Step {i} · `{step['tool']}({args_str})`"):
                if isinstance(obs, dict): st.json(obs)
                else: st.code(str(obs), language='json')

        st.divider()
        st.markdown("#####  RCA Report")
        st.markdown(res['final_report'])

with tab_saved:
    if not rca_results:
        st.info("No saved reports yet.")
    else:
        st.markdown(f"**{len(rca_results)} saved report(s)**")
        for i, r in enumerate(rca_results, 1):
            alert = r.get('alert', {})
            with st.expander(f"Alert {i} — {alert.get('device_id','?')}/{alert.get('interface_id','?')} | {alert.get('anomaly_type','?')} | {r.get('steps','?')} tools"):
                st.markdown("**RCA Report:**")
                st.markdown(r.get('final_report', '*No report.*'))
