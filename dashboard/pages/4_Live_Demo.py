"""
Live Demo Dashboard
===================
Three-panel live demo page:
  Panel 1 — Funnel counter (rows → T1 flags → T2 alerts)
  Panel 2 — Live alert feed with fault badges
  Panel 3 — Agent reasoning stream + RCA report

Reads from demo/state.json written by demo/live_pipeline.py.
Auto-refreshes every 2 seconds.
"""

import json, time
from pathlib import Path

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Live Demo", layout="wide")

ROOT       = Path(__file__).parent.parent.parent
STATE_FILE = ROOT / "demo" / "state.json"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0F172A; color: #F1F5F9; }
[data-testid="stSidebar"]           { background: #0F172A; }
.block-container                    { padding-top: 1.5rem; padding-bottom: 1rem; }
h1, h2, h3                          { color: #F1F5F9; }
.funnel-box {
    background: #1E293B; border: 1px solid #334155; border-radius: 12px;
    padding: 20px; text-align: center;
}
.funnel-number { font-size: 2.4rem; font-weight: 900; }
.funnel-label  { font-size: 0.72rem; text-transform: uppercase;
                  letter-spacing: 0.1em; color: #64748B; margin-top: 4px; }
.alert-card {
    background: #1E293B; border: 1px solid #334155;
    border-radius: 10px; padding: 14px 18px; margin-bottom: 8px;
}
.step-card {
    border-radius: 8px; padding: 10px 14px; margin-bottom: 6px;
    font-size: 0.83rem; font-family: monospace; line-height: 1.6;
}
.step-think   { background: #1e293b; border-left: 3px solid #6366F1; }
.step-act     { background: #1e2a1e; border-left: 3px solid #10B981; }
.step-observe { background: #1a1e2e; border-left: 3px solid #3B82F6; }
.step-error   { background: #2a1e1e; border-left: 3px solid #EF4444; }
.rca-box {
    background: linear-gradient(135deg, #0F2027, #1E293B);
    border: 1px solid #6366F1; border-radius: 12px; padding: 24px;
}
</style>
""", unsafe_allow_html=True)

# ── Load state ────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "status": "idle", "rows_seen": 0, "tier1_flags": 0, "tier2_flags": 0,
        "alerts": [], "agent_steps": [], "rca_report": None, "injected_fault": None,
    }

state = load_state()

# ── Header ────────────────────────────────────────────────────────────────────
status = state.get("status", "idle")
status_color = {"idle": "#475569", "running": "#10B981",
                "investigating": "#F97316", "done": "#6366F1"}.get(status, "#475569")

injected = state.get("injected_fault")
if injected:
    st.markdown(f"""
    <div style='background:{injected["color"]}22; border:1px solid {injected["color"]};
         border-radius:10px; padding:12px 20px; margin-bottom:16px;
         display:flex; align-items:center; gap:14px;'>
      <div style='font-size:1.3rem;'>⚡</div>
      <div>
        <div style='font-size:0.78rem; font-weight:700; color:{injected["color"]};
                    text-transform:uppercase; letter-spacing:0.1em;'>
          Active Fault Injection
        </div>
        <div style='font-size:0.9rem; color:#F1F5F9; font-weight:600; margin-top:2px;'>
          {injected["name"].replace("_"," ").title()}
        </div>
        <div style='font-size:0.78rem; color:#94A3B8;'>{injected["description"]}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown("## AIOps Live Demo")
with col_status:
    st.markdown(f"""
    <div style='text-align:right; padding-top:8px;'>
      <span style='background:{status_color}22; color:{status_color};
                   border:1px solid {status_color}; border-radius:6px;
                   padding:4px 12px; font-size:0.78rem; font-weight:700;
                   text-transform:uppercase;'>
        {status}
      </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — The Funnel
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("#### The Three-Tier Funnel")

rows_seen   = state.get("rows_seen", 0)
tier1_flags = state.get("tier1_flags", 0)
tier2_flags = state.get("tier2_flags", 0)
alerts      = state.get("alerts", [])

f1, arrow1, f2, arrow2, f3, arrow3, f4 = st.columns([2, 0.4, 2, 0.4, 2, 0.4, 2])

with f1:
    st.markdown(f"""
    <div class='funnel-box'>
      <div class='funnel-number' style='color:#94A3B8;'>{rows_seen:,}</div>
      <div class='funnel-label'>SNMP rows ingested</div>
    </div>""", unsafe_allow_html=True)
with arrow1:
    st.markdown("<div style='text-align:center; font-size:1.8rem; padding-top:18px; color:#334155;'>→</div>",
                unsafe_allow_html=True)
with f2:
    st.markdown(f"""
    <div class='funnel-box'>
      <div class='funnel-number' style='color:#F97316;'>{tier1_flags:,}</div>
      <div class='funnel-label'>Tier 1 (RF) flags</div>
    </div>""", unsafe_allow_html=True)
with arrow2:
    st.markdown("<div style='text-align:center; font-size:1.8rem; padding-top:18px; color:#334155;'>→</div>",
                unsafe_allow_html=True)
with f3:
    st.markdown(f"""
    <div class='funnel-box'>
      <div class='funnel-number' style='color:#6366F1;'>{tier2_flags:,}</div>
      <div class='funnel-label'>Tier 2 (LSTM) confirmed</div>
    </div>""", unsafe_allow_html=True)
with arrow3:
    st.markdown("<div style='text-align:center; font-size:1.8rem; padding-top:18px; color:#334155;'>→</div>",
                unsafe_allow_html=True)
with f4:
    rca_done = 1 if state.get("rca_report") else 0
    st.markdown(f"""
    <div class='funnel-box'>
      <div class='funnel-number' style='color:#10B981;'>{rca_done}</div>
      <div class='funnel-label'>Tier 3 (Agent) RCA reports</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Alert Feed + Agent Panel (side by side)
# ══════════════════════════════════════════════════════════════════════════════
col_feed, col_agent = st.columns([1, 1.6])

# ── Left: Live Alert Feed ─────────────────────────────────────────────────────
with col_feed:
    st.markdown("#### Live Anomaly Feed")
    if not alerts:
        st.markdown("""
        <div style='background:#1E293B; border:1px solid #334155; border-radius:10px;
                    padding:24px; text-align:center; color:#475569; font-size:0.9rem;'>
          No alerts yet — pipeline is monitoring...
        </div>""", unsafe_allow_html=True)
    else:
        for alert in alerts[:8]:
            fault = alert.get("fault", "unknown")
            color = {
                "link_failure":   "#ef4444",
                "congestion":     "#f97316",
                "interface_flap": "#8b5cf6",
                "mtu_mismatch":   "#06b6d4",
                "packet_loss":    "#eab308",
                "none":           "#475569",
            }.get(fault, "#6366F1")

            mse_val = alert.get("mse", 0)
            ts_str  = str(alert.get("ts", ""))[:19]

            st.markdown(f"""
            <div class='alert-card'>
              <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                  <span style='font-family:monospace; font-weight:700; color:#F1F5F9;
                               font-size:0.9rem;'>{alert.get("device","?")}</span>
                  <span style='color:#475569; margin:0 6px;'>/</span>
                  <span style='font-family:monospace; font-size:0.8rem; color:#94A3B8;'>
                    {alert.get("interface","?")}
                  </span>
                </div>
                <span style='background:{color}22; color:{color}; border:1px solid {color};
                             border-radius:5px; padding:2px 8px; font-size:0.7rem;
                             font-weight:700; text-transform:uppercase;'>
                  {fault.replace("_"," ")}
                </span>
              </div>
              <div style='margin-top:6px; font-size:0.75rem; color:#475569;'>
                MSE: <span style='color:#94A3B8;'>{mse_val:.4f}</span>
                &nbsp;&nbsp; {ts_str}
              </div>
            </div>""", unsafe_allow_html=True)

# ── Right: Agent Reasoning Stream ────────────────────────────────────────────
with col_agent:
    st.markdown("#### Agent Investigation")

    steps = state.get("agent_steps", [])
    rca   = state.get("rca_report")

    if not steps and not rca:
        st.markdown("""
        <div style='background:#1E293B; border:1px solid #334155; border-radius:10px;
                    padding:24px; text-align:center; color:#475569; font-size:0.9rem;'>
          Waiting for Tier 2 to confirm an alert...
        </div>""", unsafe_allow_html=True)
    else:
        # Reasoning steps
        for step in steps:
            stype = step.get("type", "think")
            label = {"think": "THINK", "act": "ACT",
                     "observe": "OBSERVE", "error": "ERROR"}.get(stype, stype.upper())
            label_color = {
                "THINK": "#6366F1", "ACT": "#10B981",
                "OBSERVE": "#3B82F6", "ERROR": "#EF4444"
            }.get(label, "#94A3B8")
            st.markdown(f"""
            <div class='step-card step-{stype}'>
              <span style='color:{label_color}; font-weight:700; margin-right:10px;'>
                {label}
              </span>{step.get("text","").replace("<","&lt;").replace(">","&gt;")}
            </div>""", unsafe_allow_html=True)

        # Final RCA report
        if rca:
            verdict = rca.get("verdict", "UNKNOWN") if isinstance(rca, dict) else "COMPLETE"
            verdict_color = {
                "LINK_FAILURE": "#ef4444", "CONGESTION": "#f97316",
                "MTU_MISMATCH": "#06b6d4", "INTERFACE_FLAP": "#8b5cf6",
                "PACKET_LOSS": "#eab308", "FALSE_POSITIVE": "#10b981",
                "ERROR": "#ef4444",
            }.get(verdict, "#6366F1")

            summary = rca.get("summary", str(rca)) if isinstance(rca, dict) else str(rca)

            st.markdown(f"""
            <div class='rca-box' style='margin-top:12px;'>
              <div style='font-size:0.72rem; font-weight:700; color:#6366F1;
                          text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;'>
                Root Cause Analysis — Final Report
              </div>
              <div style='display:flex; align-items:center; gap:12px; margin-bottom:12px;'>
                <span style='background:{verdict_color}22; color:{verdict_color};
                             border:1px solid {verdict_color}; border-radius:6px;
                             padding:4px 14px; font-size:0.82rem; font-weight:800;
                             letter-spacing:0.06em;'>
                  {verdict.replace("_"," ")}
                </span>
              </div>
              <div style='font-size:0.83rem; color:#94A3B8; line-height:1.7;'>
                {summary[:800]}
              </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Control Panel in sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Demo Controls")
    st.markdown("<div style='color:#475569; font-size:0.8rem; margin-bottom:12px;'>Run the pipeline first, then inject a fault.</div>",
                unsafe_allow_html=True)

    st.markdown("**Start Pipeline**")
    st.code("python3 demo/live_pipeline.py --replay", language="bash")

    st.markdown("**Inject Fault (simulation)**")
    fault_choice = st.selectbox("Choose fault", [
        "link_failure", "congestion", "interface_flap", "mtu_mismatch", "packet_loss"
    ])
    if st.button("Inject Fault", use_container_width=True):
        import subprocess
        subprocess.Popen(
            ["python3", "demo/inject_fault.py", fault_choice, "--sim"],
            cwd=str(ROOT)
        )
        st.success(f"Injected: {fault_choice}")

    if st.button("Recover / Clear", use_container_width=True):
        import subprocess
        subprocess.Popen(
            ["python3", "demo/inject_fault.py", "recover", "--sim"],
            cwd=str(ROOT)
        )
        st.success("Network recovered.")

    st.markdown("---")
    st.markdown("**Ubuntu Live Mode**")
    st.code("sudo python3 demo/inject_fault.py link_failure", language="bash")
    st.markdown("<div style='color:#475569; font-size:0.76rem;'>No --sim flag = real Linux commands</div>",
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<div style='color:#334155; font-size:0.72rem;'>Last update: {state.get('last_update','—')[:19]}</div>",
                unsafe_allow_html=True)

# ── Auto-refresh every 2 seconds ──────────────────────────────────────────────
time.sleep(2)
st.rerun()
