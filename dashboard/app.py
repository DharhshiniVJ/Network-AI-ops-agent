"""
AIOps Demo Dashboard
Three panels: Funnel | Alert Feed | Agent Reasoning
Reads demo/state.json written by demo/live_pipeline.py
Auto-refreshes every 2 seconds.
"""

import json, time
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="AIOps Demo", layout="wide", initial_sidebar_state="collapsed")

ROOT       = Path(__file__).parent.parent
STATE_FILE = ROOT / "demo" / "state.json"

# ── Minimal dark CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#0F172A; color:#F1F5F9; }
[data-testid="stSidebar"]           { background:#0F172A; }
.block-container { padding:1.5rem 2rem 1rem 2rem; }
div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }

.panel {
    background:#1E293B; border:1px solid #334155;
    border-radius:10px; padding:18px; height:100%;
}
.panel-title {
    font-size:0.72rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.1em; color:#64748B; margin-bottom:14px;
}
.funnel-num   { font-size:2.6rem; font-weight:900; line-height:1; }
.funnel-label { font-size:0.7rem; color:#475569; text-transform:uppercase;
                letter-spacing:0.08em; margin-top:4px; }
.arrow        { font-size:1.4rem; color:#334155; text-align:center; padding-top:28px; }

.alert-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:10px 12px; border-radius:7px; background:#0F172A;
    margin-bottom:6px; font-family:monospace;
}
.badge {
    font-size:0.68rem; font-weight:700; padding:2px 8px;
    border-radius:4px; text-transform:uppercase; letter-spacing:0.05em;
}
.step {
    padding:8px 12px; border-radius:6px; margin-bottom:5px;
    font-family:monospace; font-size:0.8rem; line-height:1.5;
}
.step-think   { background:#1e293b; border-left:3px solid #6366F1; }
.step-act     { background:#0f2a1a; border-left:3px solid #10B981; }
.step-observe { background:#0f1a2a; border-left:3px solid #3B82F6; }
.step-error   { background:#2a0f0f; border-left:3px solid #EF4444; }

.rca-box {
    background:#0F172A; border:1px solid #6366F1;
    border-radius:8px; padding:16px; margin-top:10px;
}
.verdict-badge {
    display:inline-block; font-size:0.78rem; font-weight:800;
    padding:4px 14px; border-radius:5px; letter-spacing:0.06em;
    margin-bottom:10px;
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
    return {"status":"idle","rows_seen":0,"tier1_flags":0,"tier2_flags":0,
            "alerts":[],"agent_steps":[],"rca_report":None,"injected_fault":None}

state = load_state()

FAULT_COLORS = {
    "link_failure":   "#ef4444",
    "congestion":     "#f97316",
    "interface_flap": "#8b5cf6",
    "mtu_mismatch":   "#06b6d4",
    "packet_loss":    "#eab308",
}

# ── Header ────────────────────────────────────────────────────────────────────
status = state.get("status","idle")
status_color = {"idle":"#475569","running":"#10B981",
                "investigating":"#F97316","done":"#6366F1"}.get(status,"#475569")

h_left, h_right = st.columns([5,1])
with h_left:
    st.markdown("## Agentic AIOps — Network Root Cause Analysis")
with h_right:
    inj = state.get("injected_fault")
    label = inj["name"].replace("_"," ").title() if inj else status.upper()
    color = inj["color"] if inj else status_color
    st.markdown(f"""<div style='text-align:right;padding-top:10px;'>
      <span style='background:{color}22;color:{color};border:1px solid {color};
                   border-radius:6px;padding:4px 14px;font-size:0.78rem;font-weight:700;'>
        {label}
      </span></div>""", unsafe_allow_html=True)

st.markdown("<div style='border-top:1px solid #1E293B;margin:6px 0 14px 0;'></div>",
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# THREE PANELS
# ══════════════════════════════════════════════════════════════════════════════
p1, p2, p3 = st.columns([1, 1, 1.6])

# ── Panel 1: Funnel ───────────────────────────────────────────────────────────
with p1:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Three-Tier Funnel</div>", unsafe_allow_html=True)

    rows   = state.get("rows_seen",   0)
    t1     = state.get("tier1_flags", 0)
    t2     = state.get("tier2_flags", 0)
    rca_n  = 1 if state.get("rca_report") else 0

    for num, label, color in [
        (f"{rows:,}",  "SNMP rows ingested",          "#94A3B8"),
        (f"{t1:,}",    "Tier 1 · RF flags",           "#F97316"),
        (f"{t2:,}",    "Tier 2 · LSTM confirmed",     "#6366F1"),
        (str(rca_n),   "Tier 3 · Agent RCA reports",  "#10B981"),
    ]:
        st.markdown(f"""
        <div style='padding:12px 0; border-bottom:1px solid #334155;'>
          <div class='funnel-num' style='color:{color};'>{num}</div>
          <div class='funnel-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── Panel 2: Live Alert Feed ──────────────────────────────────────────────────
with p2:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Live Anomaly Feed</div>", unsafe_allow_html=True)

    alerts = state.get("alerts", [])
    if not alerts:
        st.markdown("<div style='color:#334155;font-size:0.85rem;padding:20px 0;'>"
                    "Monitoring... no alerts yet.</div>", unsafe_allow_html=True)
    else:
        for a in alerts[:10]:
            fault  = a.get("fault","unknown")
            color  = FAULT_COLORS.get(fault, "#6366F1")
            device = a.get("device","?")
            intf   = a.get("interface","?")
            mse    = a.get("mse", 0)
            ts     = str(a.get("ts",""))[:16]
            st.markdown(f"""
            <div class='alert-row'>
              <div>
                <span style='color:#F1F5F9;font-weight:700;'>{device}</span>
                <span style='color:#334155;margin:0 4px;'>/</span>
                <span style='color:#64748B;font-size:0.8rem;'>{intf}</span><br>
                <span style='color:#334155;font-size:0.72rem;'>MSE {mse:.4f} · {ts}</span>
              </div>
              <span class='badge' style='background:{color}22;color:{color};border:1px solid {color};'>
                {fault.replace("_"," ")}
              </span>
            </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── Panel 3: Agent Reasoning + RCA ───────────────────────────────────────────
with p3:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Agent Investigation</div>", unsafe_allow_html=True)

    steps = state.get("agent_steps", [])
    rca   = state.get("rca_report")

    if not steps and not rca:
        st.markdown("<div style='color:#334155;font-size:0.85rem;padding:20px 0;'>"
                    "Waiting for Tier 2 to confirm an alert...</div>", unsafe_allow_html=True)
    else:
        steps_html = ""
        for step in steps[-12:]:   # show last 12 steps max
            stype = step.get("type","think")
            label = {"think":"THINK","act":"ACT","observe":"OBSERVE","error":"ERROR"}.get(stype, stype.upper())
            lcolor = {"THINK":"#6366F1","ACT":"#10B981","OBSERVE":"#3B82F6","ERROR":"#EF4444"}.get(label,"#94A3B8")
            text  = str(step.get("text","")).replace("<","&lt;").replace(">","&gt;")[:200]
            steps_html += f"""<div class='step step-{stype}'>
              <span style='color:{lcolor};font-weight:700;margin-right:8px;'>{label}</span>{text}
            </div>"""
        st.markdown(steps_html, unsafe_allow_html=True)

        if rca:
            verdict = rca.get("verdict","UNKNOWN") if isinstance(rca,dict) else "COMPLETE"
            summary = rca.get("summary", str(rca)) if isinstance(rca,dict) else str(rca)
            vc = {"LINK_FAILURE":"#ef4444","CONGESTION":"#f97316","MTU_MISMATCH":"#06b6d4",
                  "INTERFACE_FLAP":"#8b5cf6","PACKET_LOSS":"#eab308",
                  "FALSE_POSITIVE":"#10b981","ERROR":"#ef4444"}.get(verdict,"#6366F1")
            st.markdown(f"""
            <div class='rca-box'>
              <span class='verdict-badge'
                    style='background:{vc}22;color:{vc};border:1px solid {vc};'>
                {verdict.replace("_"," ")}
              </span>
              <div style='font-size:0.82rem;color:#94A3B8;line-height:1.7;'>
                {summary[:500]}
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── How to run (shown when idle) ──────────────────────────────────────────────
if status == "idle":
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Start pipeline**")
        st.code("python3 demo/live_pipeline.py --replay", language="bash")
    with c2:
        st.markdown("**2. Inject a fault**")
        st.code("python3 demo/inject_fault.py link_failure --sim", language="bash")
    with c3:
        st.markdown("**3. Recover**")
        st.code("python3 demo/inject_fault.py recover --sim", language="bash")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
time.sleep(2)
st.rerun()
