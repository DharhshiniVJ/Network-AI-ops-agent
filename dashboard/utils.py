"""
Shared utilities and CSS for the AIOps dashboard.
"""
import json
import pandas as pd
import numpy as np
import torch
import joblib
import streamlit as st
from pathlib import Path

# ── Professional CSS ──────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Metric cards */
[data-testid="metric-container"] {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label {
    color: #94A3B8 !important;
    font-size: 0.78rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #F1F5F9 !important;
    font-size: 1.8rem !important;
    font-weight: 700;
}
[data-testid="metric-container"] [data-testid="metric-delta"] {
    font-size: 0.75rem !important;
}

/* Section headers */
h2, h3 { color: #E2E8F0 !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0F172A;
    border-right: 1px solid #1E293B;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
}

/* Tabs */
[data-testid="stTab"] {
    font-weight: 600;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 8px;
}

/* Buttons */
[data-testid="baseButton-primary"] {
    background: #6366F1 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* Divider */
hr { border-color: #1E293B !important; }

/* Alert boxes */
[data-testid="stAlert"] {
    border-radius: 8px;
}

/* Status badge */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-green  { background: #064E3B; color: #34D399; }
.badge-red    { background: #7F1D1D; color: #FCA5A5; }
.badge-yellow { background: #78350F; color: #FCD34D; }
.badge-blue   { background: #1E3A5F; color: #93C5FD; }
.badge-purple { background: #3B0764; color: #D8B4FE; }
</style>
"""

# ── Color Palette ─────────────────────────────────────────────────────────────

FAULT_COLORS = {
    'link_failure'   : '#EF4444',
    'congestion'     : '#F97316',
    'mtu_mismatch'   : '#EAB308',
    'interface_flap' : '#8B5CF6',
    'routing_loop'   : '#EC4899',
    'none'           : '#475569',
    'False Positive' : '#10B981',
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.5)',
        font=dict(family='Inter, sans-serif', color='#CBD5E1'),
        title=dict(font=dict(color='#F1F5F9', size=15), x=0),
        xaxis=dict(gridcolor='#1E293B', linecolor='#334155', tickfont=dict(color='#94A3B8')),
        yaxis=dict(gridcolor='#1E293B', linecolor='#334155', tickfont=dict(color='#94A3B8')),
        coloraxis_colorbar=dict(tickfont=dict(color='#94A3B8')),
    )
)

# ── Cached Loaders ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models():
    rf = joblib.load('data/models/random_forest.pkl')
    lstm = torch.load('data/models/lstm_autoencoder.pth',
                      map_location='cpu', weights_only=False)
    lstm.device = torch.device('cpu')
    lstm.model.to('cpu')
    return rf, lstm


@st.cache_resource(show_spinner=False)
def load_telemetry():
    df = pd.read_parquet('data/snmp_telemetry.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['device_id', 'interface_id', 'timestamp']).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def run_detection():
    df = load_telemetry()
    rf, lstm = load_models()
    rf_preds, _         = rf.predict(df)
    lstm_preds, scores  = lstm.predict(df)
    return rf_preds, lstm_preds, scores


@st.cache_data(show_spinner=False)
def load_syslogs():
    syslogs = []
    for path in (sorted(Path("data").glob("syslog*.jsonl")) +
                 sorted(Path("data").glob("syslog*.txt"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    syslogs.append(
                        f"[{obj.get('timestamp','')}] {obj.get('host','')} "
                        f"{obj.get('facility','')} {obj.get('message','')}"
                    )
                except Exception:
                    syslogs.append(line)
        break
    return syslogs


def load_benchmark_results():
    p = Path('results/agent_benchmark.json')
    return json.loads(p.read_text()) if p.exists() else None


def load_rca_results():
    p = Path('results/rca_reports.json')
    return json.loads(p.read_text()) if p.exists() else None


def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def sidebar_branding():
    with st.sidebar:
        st.markdown("##  AIOps RCA")
        st.markdown("**Agentic Network Fault Detection**")
        st.caption("3-Tier ML + LLM Architecture")
        st.divider()
        st.markdown("**Navigation**")
        st.markdown(" Overview")
        st.markdown(" Detection Results")
        st.markdown(" Agent Console")
        st.markdown(" Benchmark")
        st.divider()
        st.caption("Gemini 2.5 Flash · LangGraph · PyTorch")
