"""
AIOps Demo Dashboard — Incident Response View
Three acts: Healthy → Detected → Diagnosed
"""

import json, time, subprocess
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="AIOps Demo", layout="wide", initial_sidebar_state="collapsed")

ROOT       = Path(__file__).parent.parent
STATE_FILE = ROOT / "demo" / "state.json"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#0A0F1E; color:#F1F5F9; }
[data-testid="stSidebar"]           { background:#0A0F1E; }
.block-container { padding:1.2rem 2rem 1rem 2rem; }
.stButton>button {
    background:#1E293B; color:#F1F5F9; border:1px solid #334155;
    border-radius:8px; padding:8px 20px; font-size:0.85rem; font-weight:600;
    width:100%;
}
.stButton>button:hover { background:#6366F1; border-color:#6366F1; }

.card {
    background:#1E293B; border:1px solid #334155;
    border-radius:12px; padding:20px 22px; height:100%;
}
.card-title {
    font-size:0.68rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.12em; color:#475569; margin-bottom:16px;
}

/* Funnel */
.funnel-row {
    display:flex; align-items:center; gap:0; margin-bottom:2px;
}
.funnel-item {
    flex:1; padding:14px 12px; border-radius:8px; text-align:center;
}
.funnel-num   { font-size:2rem; font-weight:900; line-height:1; }
.funnel-label { font-size:0.62rem; color:#64748B; text-transform:uppercase;
                letter-spacing:0.08em; margin-top:3px; }
.funnel-arrow { color:#334155; font-size:1.2rem; padding:0 4px; flex:0; }

/* Detection pipeline cards */
.detect-card {
    border-radius:10px; padding:16px 18px; margin-bottom:10px;
    border:1px solid #334155;
}
.detect-waiting  { background:#0F172A; opacity:0.5; }
.detect-active   { background:#1E293B; border-color:#6366F1; }
.detect-done     { background:#0F2A1A; border-color:#10B981; }
.detect-label {
    font-size:0.68rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.1em; margin-bottom:6px;
}
.detect-result { font-size:1.05rem; font-weight:800; margin:4px 0; }
.detect-sub    { font-size:0.75rem; color:#64748B; }

/* RCA report */
.rca-card {
    background:linear-gradient(135deg, #0D1B2A, #1E293B);
    border:1px solid #6366F1; border-radius:12px; padding:22px;
}
.verdict-pill {
    display:inline-block; font-size:0.82rem; font-weight:800;
    padding:5px 18px; border-radius:20px; letter-spacing:0.06em;
    margin-bottom:14px;
}
.rca-section { margin-top:12px; }
.rca-section-title {
    font-size:0.65rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.1em; color:#6366F1; margin-bottom:4px;
}
.rca-text { font-size:0.82rem; color:#CBD5E1; line-height:1.7; }

/* Alert row */
.alert-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:9px 12px; border-radius:7px; background:#0F172A;
    margin-bottom:5px; font-family:monospace;
}
.badge {
    font-size:0.65rem; font-weight:700; padding:2px 8px;
    border-radius:4px; text-transform:uppercase; letter-spacing:0.05em;
}

/* Status banner */
.status-bar {
    border-radius:10px; padding:10px 18px; margin-bottom:16px;
    display:flex; align-items:center; justify-content:space-between;
}
</style>
""", unsafe_allow_html=True)

FAULT_COLORS = {
    "link_failure":   "#ef4444", "link failure":   "#ef4444",
    "congestion":     "#f97316",
    "interface_flap": "#8b5cf6", "interface flap": "#8b5cf6",
    "mtu_mismatch":   "#06b6d4", "mtu mismatch":   "#06b6d4",
    "packet_loss":    "#eab308", "packet loss":    "#eab308",
    "routing_loop":   "#ec4899", "routing loop":   "#ec4899",
    "false_positive": "#10b981", "false positive": "#10b981",
    "unknown":        "#6366F1", "none":           "#334155",
}

REMEDIATION = {
    "LINK_FAILURE":   "Re-enable the physical link or activate the standby redundant path. Check fiber/cable on the affected port.",
    "CONGESTION":     "Apply QoS policy to prioritise critical traffic. Increase link capacity or redistribute load via ECMP.",
    "MTU_MISMATCH":   "Standardise MTU to 1500 bytes across all interfaces on this path. Check jumbo-frame settings.",
    "INTERFACE_FLAP": "Inspect physical layer — SFP module, cable integrity, or NIC driver. Enable dampening to suppress brief outages.",
    "PACKET_LOSS":    "Check for CRC errors and duplex mismatch. Run BER test on the physical medium.",
    "ROUTING_LOOP":   "Verify OSPF/BGP route advertisements. Add route filters to prevent loop injection.",
    "FALSE_POSITIVE": "No action required. Alert suppressed — all metrics within normal bounds.",
}

# ── Load state ────────────────────────────────────────────────────────────────
def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"status":"idle","rows_seen":0,"tier1_flags":0,"tier2_flags":0,
            "alerts":[],"agent_steps":[],"rca_report":None,
            "rf_verdict":None,"lstm_verdict":None,"injected_fault":None}

state = load_state()
status      = state.get("status","idle")
rf_v        = state.get("rf_verdict")
lstm_v      = state.get("lstm_verdict")
rca         = state.get("rca_report")
inj         = state.get("injected_fault")
alerts      = state.get("alerts", [])

# ── Status Banner ─────────────────────────────────────────────────────────────
if rca:
    verdict = rca.get("verdict","UNKNOWN") if isinstance(rca,dict) else "UNKNOWN"
    vc = FAULT_COLORS.get(verdict.lower(), "#6366F1")
    banner_bg = f"{vc}18"; banner_border = vc
    banner_text = f"INCIDENT DETECTED — {verdict.replace('_',' ')}"
elif status == "investigating":
    banner_bg="#F9731618"; banner_border="#F97316"
    banner_text = "INVESTIGATING ANOMALY..."
elif rf_v:
    banner_bg="#6366F118"; banner_border="#6366F1"
    banner_text = f"ANOMALY DETECTED — {rf_v['fault'].upper()}"
elif status == "running":
    banner_bg="#10B98118"; banner_border="#10B981"
    banner_text = "NETWORK HEALTHY — MONITORING"
else:
    banner_bg="#33415518"; banner_border="#334155"
    banner_text = "IDLE — Start the pipeline to begin"

st.markdown(f"""
<div class='status-bar' style='background:{banner_bg};border:1px solid {banner_border};'>
  <div style='font-size:0.9rem;font-weight:800;color:{banner_border};letter-spacing:0.06em;'>
    {banner_text}
  </div>
  <div style='font-size:0.72rem;color:#475569;'>
    {str(state.get("last_update",""))[:19]}
  </div>
</div>
""", unsafe_allow_html=True)

# ── FUNNEL ROW ────────────────────────────────────────────────────────────────
st.markdown("<div class='card-title' style='margin-bottom:10px;'>THREE-TIER DETECTION FUNNEL</div>",
            unsafe_allow_html=True)

rows  = state.get("rows_seen",0)
t1    = state.get("tier1_flags",0)
t2    = state.get("tier2_flags",0)
rca_n = 1 if rca else 0

funnel_html = f"""
<div style='display:flex;align-items:center;gap:6px;margin-bottom:20px;'>
  <div style='flex:1;background:#1E293B;border:1px solid #334155;border-radius:10px;
              padding:14px;text-align:center;'>
    <div class='funnel-num' style='color:#94A3B8;'>{rows:,}</div>
    <div class='funnel-label'>SNMP rows ingested</div>
  </div>
  <div class='funnel-arrow'>→</div>
  <div style='flex:1;background:#1E293B;border:1px solid #F97316;border-radius:10px;
              padding:14px;text-align:center;'>
    <div class='funnel-num' style='color:#F97316;'>{t1:,}</div>
    <div class='funnel-label'>Tier 1 · RF Flags</div>
  </div>
  <div class='funnel-arrow'>→</div>
  <div style='flex:1;background:#1E293B;border:1px solid #6366F1;border-radius:10px;
              padding:14px;text-align:center;'>
    <div class='funnel-num' style='color:#6366F1;'>{t2:,}</div>
    <div class='funnel-label'>Tier 2 · LSTM Confirmed</div>
  </div>
  <div class='funnel-arrow'>→</div>
  <div style='flex:1;background:#1E293B;border:1px solid {"#10B981" if rca_n else "#334155"};
              border-radius:10px;padding:14px;text-align:center;'>
    <div class='funnel-num' style='color:{"#10B981" if rca_n else "#475569"};'>{rca_n}</div>
    <div class='funnel-label'>Tier 3 · Agent RCA</div>
  </div>
</div>
"""
st.markdown(funnel_html, unsafe_allow_html=True)

# ── THREE COLUMNS ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1.5])

# ── Col 1: Detection Pipeline ─────────────────────────────────────────────────
with col1:
    st.markdown("<div class='card-title'>DETECTION PIPELINE</div>", unsafe_allow_html=True)

    # Tier 1 — RF
    if rf_v:
        fc = FAULT_COLORS.get(rf_v["fault"].lower(), "#F97316")
        st.markdown(f"""
        <div class='detect-card detect-done'>
          <div class='detect-label' style='color:#F97316;'>Tier 1 · Random Forest</div>
          <div class='detect-result' style='color:{fc};'>{rf_v["fault"]}</div>
          <div class='detect-sub'>
            Confidence: <b style='color:#F1F5F9;'>{rf_v["confidence"]}%</b>
            &nbsp;·&nbsp; {rf_v["device"]} / {rf_v["interface"]}
          </div>
        </div>""", unsafe_allow_html=True)
    elif status == "running":
        st.markdown("""
        <div class='detect-card detect-active'>
          <div class='detect-label' style='color:#F97316;'>Tier 1 · Random Forest</div>
          <div class='detect-result' style='color:#475569;font-size:0.85rem;'>Scanning...</div>
          <div class='detect-sub'>Watching for known fault signatures</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='detect-card detect-waiting'>
          <div class='detect-label' style='color:#475569;'>Tier 1 · Random Forest</div>
          <div class='detect-result' style='color:#334155;font-size:0.85rem;'>Waiting</div>
        </div>""", unsafe_allow_html=True)

    # Tier 2 — LSTM
    if lstm_v:
        ratio = lstm_v.get("ratio", 0)
        st.markdown(f"""
        <div class='detect-card detect-done'>
          <div class='detect-label' style='color:#6366F1;'>Tier 2 · LSTM Autoencoder</div>
          <div class='detect-result' style='color:#6366F1;'>ANOMALY CONFIRMED</div>
          <div class='detect-sub'>
            MSE: <b style='color:#F1F5F9;'>{lstm_v["mse"]:.6f}</b>
            &nbsp;·&nbsp; Threshold: {lstm_v["threshold"]:.6f}
            &nbsp;·&nbsp; <b style='color:#EF4444;'>{ratio}× above normal</b>
          </div>
        </div>""", unsafe_allow_html=True)
    elif rf_v:
        st.markdown("""
        <div class='detect-card detect-active'>
          <div class='detect-label' style='color:#6366F1;'>Tier 2 · LSTM Autoencoder</div>
          <div class='detect-result' style='color:#6366F1;font-size:0.85rem;'>Confirming...</div>
          <div class='detect-sub'>Computing reconstruction error</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='detect-card detect-waiting'>
          <div class='detect-label' style='color:#475569;'>Tier 2 · LSTM Autoencoder</div>
          <div class='detect-result' style='color:#334155;font-size:0.85rem;'>Waiting</div>
        </div>""", unsafe_allow_html=True)

    # Tier 3 — Agent
    if rca:
        verdict = rca.get("verdict","UNKNOWN") if isinstance(rca,dict) else "UNKNOWN"
        vc = FAULT_COLORS.get(verdict.lower(),"#6366F1")
        st.markdown(f"""
        <div class='detect-card detect-done'>
          <div class='detect-label' style='color:#10B981;'>Tier 3 · LangGraph Agent</div>
          <div class='detect-result' style='color:{vc};'>{verdict.replace("_"," ")}</div>
          <div class='detect-sub'>RCA complete — see report →</div>
        </div>""", unsafe_allow_html=True)
    elif lstm_v:
        st.markdown("""
        <div class='detect-card detect-active'>
          <div class='detect-label' style='color:#10B981;'>Tier 3 · LangGraph Agent</div>
          <div class='detect-result' style='color:#10B981;font-size:0.85rem;'>Investigating...</div>
          <div class='detect-sub'>Running diagnostic tool calls</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='detect-card detect-waiting'>
          <div class='detect-label' style='color:#475569;'>Tier 3 · LangGraph Agent</div>
          <div class='detect-result' style='color:#334155;font-size:0.85rem;'>Waiting</div>
        </div>""", unsafe_allow_html=True)

# ── Col 2: Live Alert Feed ────────────────────────────────────────────────────
with col2:
    st.markdown("<div class='card-title'>LIVE ANOMALY FEED</div>", unsafe_allow_html=True)
    if not alerts:
        st.markdown("""<div style='color:#334155;font-size:0.85rem;padding:12px 0;'>
            No anomalies detected yet.</div>""", unsafe_allow_html=True)
    else:
        for a in alerts[:8]:
            fault  = a.get("fault","unknown")
            color  = FAULT_COLORS.get(fault, "#6366F1")
            mse    = a.get("mse", 0)
            ts     = str(a.get("ts",""))[:16]
            st.markdown(f"""
            <div class='alert-row'>
              <div>
                <span style='color:#F1F5F9;font-weight:700;'>{a.get("device","?")}</span>
                <span style='color:#334155;margin:0 4px;'>/</span>
                <span style='color:#64748B;font-size:0.8rem;'>{a.get("interface","?")}</span><br>
                <span style='color:#334155;font-size:0.7rem;'>MSE {mse:.4f} · {ts}</span>
              </div>
              <span class='badge' style='background:{color}22;color:{color};border:1px solid {color};'>
                {fault.replace("_"," ")}
              </span>
            </div>""", unsafe_allow_html=True)

# ── Col 3: RCA Report ─────────────────────────────────────────────────────────
with col3:
    st.markdown("<div class='card-title'>ROOT CAUSE ANALYSIS REPORT</div>", unsafe_allow_html=True)

    if not rca and not lstm_v:
        st.markdown("""<div style='color:#334155;font-size:0.85rem;padding:12px 0;'>
            Waiting for an anomaly to investigate...</div>""", unsafe_allow_html=True)
    elif not rca and lstm_v:
        st.markdown("""
        <div style='background:#1E293B;border:1px solid #6366F1;border-radius:10px;
                    padding:20px;text-align:center;'>
          <div style='color:#6366F1;font-size:0.85rem;font-weight:600;'>
            Agent is investigating...
          </div>
          <div style='color:#475569;font-size:0.78rem;margin-top:6px;'>
            Calling diagnostic tools, building evidence chain
          </div>
        </div>""", unsafe_allow_html=True)

        # Show tool call progress
        steps = state.get("agent_steps", [])
        if steps:
            st.markdown("<br>", unsafe_allow_html=True)
            for step in steps[-6:]:
                stype = step.get("type","think")
                lc = {"think":"#6366F1","act":"#10B981","observe":"#3B82F6","error":"#EF4444"}.get(stype,"#64748B")
                label = stype.upper()
                text = str(step.get("text",""))[:120]
                st.markdown(f"""
                <div style='padding:6px 10px;border-radius:6px;margin-bottom:4px;
                            background:#0F172A;border-left:3px solid {lc};
                            font-family:monospace;font-size:0.75rem;color:#94A3B8;'>
                  <span style='color:{lc};font-weight:700;margin-right:8px;'>{label}</span>{text}
                </div>""", unsafe_allow_html=True)
    else:
        verdict = rca.get("verdict","UNKNOWN") if isinstance(rca,dict) else "UNKNOWN"
        summary = rca.get("summary", str(rca)) if isinstance(rca,dict) else str(rca)
        vc = FAULT_COLORS.get(verdict.lower(),"#6366F1")
        remediation = REMEDIATION.get(verdict, "Review affected interfaces and consult network team.")

        st.markdown(f"""
        <div class='rca-card'>
          <span class='verdict-pill' style='background:{vc}22;color:{vc};border:1px solid {vc};'>
            {verdict.replace("_"," ")}
          </span>

          <div class='rca-section'>
            <div class='rca-section-title'>Evidence Summary</div>
            <div class='rca-text'>{summary[:400]}</div>
          </div>

          <div class='rca-section' style='margin-top:14px;padding-top:14px;
                border-top:1px solid #334155;'>
            <div class='rca-section-title'>Recommended Remediation</div>
            <div class='rca-text' style='color:#10B981;'>{remediation}</div>
          </div>
        </div>""", unsafe_allow_html=True)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Demo Controls")
    st.code("python3 demo/live_pipeline.py --replay", language="bash")
    st.markdown("---")
    st.markdown("**Inject Fault**")
    fault_choice = st.selectbox("", [
        "link_failure","congestion","interface_flap","mtu_mismatch","packet_loss"
    ], label_visibility="collapsed")
    if st.button("Inject", use_container_width=True):
        subprocess.Popen(["python3","demo/inject_fault.py", fault_choice,"--sim"], cwd=str(ROOT))
        st.success(f"Injected: {fault_choice}")
    if st.button("Recover / Reset", use_container_width=True):
        subprocess.Popen(["python3","demo/inject_fault.py","recover","--sim"], cwd=str(ROOT))

# ── Auto-refresh ──────────────────────────────────────────────────────────────
time.sleep(2)
st.rerun()
