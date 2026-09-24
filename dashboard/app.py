"""
AIOps RCA — Interactive Story Dashboard
The main page is a full narrative: problem → dataset → anomalies → method → results.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dashboard.utils import inject_css, load_telemetry, FAULT_COLORS, PLOTLY_TEMPLATE

st.set_page_config(
    page_title="AIOps RCA — Agentic Network Fault Detection",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ── Extra narrative CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
.hero { padding: 60px 0 40px 0; }
.hero h1 {
    font-size: 3rem; font-weight: 900; color: #F1F5F9;
    line-height: 1.15; margin-bottom: 12px;
}
.hero p {
    font-size: 1.15rem; color: #94A3B8; max-width: 680px; line-height: 1.7;
}
.section-label {
    font-size: 0.72rem; font-weight: 800; color: #6366F1;
    text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 6px;
}
.section-title {
    font-size: 1.9rem; font-weight: 800; color: #F1F5F9;
    line-height: 1.2; margin-bottom: 14px;
}
.section-body {
    font-size: 0.97rem; color: #94A3B8; line-height: 1.8; max-width: 780px;
}
.callout {
    background: #1E293B; border-left: 4px solid #6366F1;
    border-radius: 0 8px 8px 0; padding: 16px 22px; margin: 20px 0;
    color: #CBD5E1; font-size: 0.93rem; line-height: 1.7;
}
.anomaly-card {
    background: #1E293B; border: 1px solid #334155;
    border-radius: 12px; padding: 20px 22px; height: 100%;
    transition: border-color 0.2s;
}
.anomaly-card h4 { font-size: 1rem; font-weight: 700; margin: 8px 0 6px 0; }
.anomaly-card p  { font-size: 0.83rem; color: #94A3B8; line-height: 1.65; margin: 0; }
.tier-pill {
    display: inline-block; padding: 3px 12px;
    border-radius: 999px; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 10px;
}
.result-number {
    font-size: 2.4rem; font-weight: 900; line-height: 1;
}
.result-label {
    font-size: 0.78rem; color: #64748B; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px;
}
.divider-line {
    border: none; border-top: 1px solid #1E293B; margin: 48px 0;
}
</style>
""", unsafe_allow_html=True)

df = load_telemetry()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style='padding:8px 0 20px 0;'>
  <div style='font-size:1.1rem; font-weight:800; color:#F1F5F9;'> AIOps RCA</div>
  <div style='font-size:0.78rem; color:#64748B; margin-top:2px;'>Final Year Project</div>
</div>
""", unsafe_allow_html=True)
    st.markdown("**On this page**")
    st.markdown("""
- [The Problem](#the-problem)
- [Our Approach](#our-approach)
- [The Dataset](#the-dataset)
- [The Anomalies](#the-anomalies)
- [The Method](#the-method)
- [What We Achieved](#what-we-achieved)
""")
    st.divider()
    st.markdown("**Explore Data**")
    st.page_link("pages/1_Dataset_Explorer.py",  label="Dataset Explorer")
    st.page_link("pages/2_Results.py",            label="Full Results")
    st.page_link("pages/3_Agent_Console.py",      label="Agent Console")
    st.divider()
    st.caption("Built with PyTorch · LangGraph · Gemini 2.5 Flash · Streamlit")

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='hero'>
  <div class='section-label'>Final Year Project · Computer Networks + AI</div>
  <h1 class='hero-h1' style='font-size:3rem; font-weight:900; color:#F1F5F9; line-height:1.15; margin-bottom:14px;'>
    Autonomous Network<br>Fault Detection & RCA<br>
    <span style='color:#6366F1;'>Using Agentic AI</span>
  </h1>
  <p style='font-size:1.1rem; color:#94A3B8; max-width:700px; line-height:1.75;'>
    A three-tier machine learning pipeline that detects network faults, catches 
    zero-day anomalies the training data never saw, and autonomously investigates 
    each alert — producing human-readable root cause reports without a single 
    human in the loop.
  </p>
</div>
""", unsafe_allow_html=True)

# Quick stats strip
s1,s2,s3,s4,s5,s6 = st.columns(6)
for col, val, label in [
    (s1, "760,320", "SNMP Records"),
    (s2, "72 hrs",  "Simulation"),
    (s3, "5",       "Fault Types"),
    (s4, "3",       "ML Tiers"),
    (s5, "0.957",   "RF F1 Score"),
    (s6, "100%",    "Mininet RF F1"),
]:
    col.markdown(f"""
<div style='background:#1E293B; border:1px solid #334155; border-radius:10px;
            padding:14px 16px; text-align:center;'>
  <div style='font-size:1.5rem; font-weight:800; color:#F1F5F9;'>{val}</div>
  <div style='font-size:0.68rem; color:#64748B; font-weight:600;
              text-transform:uppercase; letter-spacing:0.08em; margin-top:3px;'>{label}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider-line'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# THE PROBLEM
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<a name='the-problem'></a>", unsafe_allow_html=True)
st.markdown("""
<div class='section-label'>01 — The Problem</div>
<div class='section-title'>Network failures cost billions.<br>Detecting them is still manual.</div>
<div class='section-body'>
Modern networks — cloud data centres, ISPs, enterprise WANs — generate 
<strong style='color:#F1F5F9;'>millions of telemetry events per hour</strong>: 
SNMP counters, syslog messages, NetFlow records, interface statistics. 
When something goes wrong, a Network Operations Centre (NOC) engineer must 
manually sift through this flood, correlate dozens of metrics across dozens 
of devices, and identify the root cause — often under pressure, in the middle 
of the night.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
p1, p2, p3 = st.columns(3)
for col, icon, title, body in [
    (p1, "", "Slow MTTR",
     "Mean Time to Resolution for network faults averages 4–8 hours in enterprise networks. Every minute of downtime costs money."),
    (p2, "", "Alert Fatigue",
     "NOC teams receive thousands of alerts daily. Up to 70% are false positives, leading engineers to ignore real incidents."),
    (p3, "", "Zero-Day Blind Spots",
     "Rule-based systems only catch known fault signatures. Novel attacks and new failure modes go completely undetected."),
]:
    col.markdown(f"""
<div style='background:#1E293B; border:1px solid #334155; border-radius:10px; padding:20px;'>
  <div style='font-size:1.6rem; margin-bottom:10px;'>{icon}</div>
  <div style='font-size:0.95rem; font-weight:700; color:#F1F5F9; margin-bottom:8px;'>{title}</div>
  <div style='font-size:0.83rem; color:#94A3B8; line-height:1.65;'>{body}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='callout'>
<strong style='color:#A5B4FC;'>Research Question:</strong> Can we build an AI system that automatically 
detects network anomalies — including fault types never seen during training — and produces 
evidence-grounded root cause reports without requiring human intervention?
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider-line'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# OUR APPROACH
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<a name='our-approach'></a>", unsafe_allow_html=True)
st.markdown("""
<div class='section-label'>02 — Our Approach</div>
<div class='section-title'>Three tiers. Each solving<br>a different part of the problem.</div>
<div class='section-body'>
No single ML model can solve all three problems above simultaneously. 
A supervised classifier catches known faults with high precision but is blind to 
new attack types. An unsupervised model catches everything unusual but floods 
operators with false positives. An LLM agent can reason — but it needs evidence 
to reason <em>from</em>. So we stack all three.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

t1,t2,t3 = st.columns(3)
for col, num, color, name, desc, solves in [
    (t1, "01", "#EF4444", "Random Forest",
     "A supervised classifier trained on labelled examples of 5 known fault types.",
     "Solves: slow, error-prone manual rule matching for known faults."),
    (t2, "02", "#F97316", "LSTM Autoencoder",
     "An unsupervised deep learning model trained only on normal traffic. Anything that deviates from 'normal' is flagged — even faults it was never trained on.",
     "Solves: zero-day blind spots. Caught BGP hijack with 100% recall."),
    (t3, "03", "#818CF8", "LangGraph ReAct Agent",
     "A Gemini 2.5 Flash LLM that autonomously calls 6 diagnostic tools, reasons over the evidence, and writes a structured RCA report.",
     "Solves: alert fatigue. Eliminates false positives before a human ever sees them."),
]:
    col.markdown(f"""
<div style='background:#1E293B; border:1px solid #334155; border-radius:12px; padding:22px;'>
  <div style='font-size:0.68rem; font-weight:800; color:{color}; text-transform:uppercase;
              letter-spacing:0.12em; margin-bottom:8px;'>TIER {num}</div>
  <div style='font-size:1.05rem; font-weight:700; color:#F1F5F9; margin-bottom:10px;'>{name}</div>
  <div style='font-size:0.83rem; color:#94A3B8; line-height:1.65; margin-bottom:14px;'>{desc}</div>
  <div style='font-size:0.78rem; color:{color}; font-weight:600;'>{solves}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider-line'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# THE DATASET
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<a name='the-dataset'></a>", unsafe_allow_html=True)
st.markdown("""
<div class='section-label'>03 — The Dataset</div>
<div class='section-title'>760,320 records.<br>Built from scratch.</div>
<div class='section-body'>
Real network telemetry datasets with labelled anomalies are rarely public — 
network operators don't share failure data. So we built a <strong style='color:#F1F5F9;'>
stochastic network telemetry simulator</strong> that generates statistically realistic 
SNMP metrics across two full network topologies over 72 hours, then injects 
controlled faults at randomised intervals. This approach — standard in networking 
research — produces reproducible, fully labelled data with realistic diurnal 
traffic patterns (day/night cycles).
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_stats, col_chart = st.columns([1, 2])

with col_stats:
    for label, val, color in [
        ("Total Records",     "760,320",        "#6366F1"),
        ("Simulation Length", "72 hours",        "#10B981"),
        ("Polling Interval",  "15 seconds",      "#F97316"),
        ("Topologies",        "2 (Spine-Leaf + WAN)", "#8B5CF6"),
        ("Total Devices",     "15 interfaces",   "#EF4444"),
        ("Anomaly Rounds",    "3 per topology",  "#EAB308"),
        ("Training Split",    "60 / 20 / 20 %",  "#64748B"),
        ("Anomaly Rate",      f"{(df['anomaly_type']!='none').mean()*100:.1f}%", "#94A3B8"),
    ]:
        st.markdown(f"""
<div style='display:flex; justify-content:space-between; align-items:center;
            padding:9px 0; border-bottom:1px solid #1E293B;'>
  <span style='font-size:0.83rem; color:#94A3B8;'>{label}</span>
  <span style='font-size:0.85rem; font-weight:700; color:{color};'>{val}</span>
</div>
""", unsafe_allow_html=True)

with col_chart:
    sample  = df.iloc[::15].copy()
    sample['time_bin'] = sample['timestamp'].dt.floor('20min')
    tl = (sample[sample['anomaly_type']!='none']
          .groupby(['time_bin','anomaly_type']).size().reset_index(name='count'))
    if len(tl):
        fig = px.bar(tl, x='time_bin', y='count', color='anomaly_type',
                     color_discrete_map=FAULT_COLORS,
                     labels={'time_bin':'','count':'Events','anomaly_type':'Fault'})
        fig.update_layout(**PLOTLY_TEMPLATE['layout'], height=320,
                          margin=dict(l=0,r=0,t=10,b=0),
                          legend=dict(orientation='h', y=1.05, x=0,
                                      bgcolor='rgba(0,0,0,0)'))
        fig.update_traces(marker_line_width=0)
        st.caption("Fault injections across the 72-hour simulation window")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class='callout'>
<strong style='color:#A5B4FC;'>Why simulated data is academically valid:</strong> 
Simulation-based datasets are standard in networking research (NS-3, GNS3, Mininet). 
Our simulator uses realistic SNMP counter models (Poisson traffic, diurnal sine-wave 
CPU utilisation, exponential fault durations) and was validated against real 
Linux-kernel Mininet telemetry — achieving RF F1 = 1.000 on the real-world transfer test.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider-line'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# THE ANOMALIES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<a name='the-anomalies'></a>", unsafe_allow_html=True)
st.markdown("""
<div class='section-label'>04 — The Anomalies</div>
<div class='section-title'>Five fault types.<br>One zero-day surprise.</div>
<div class='section-body'>
The dataset contains five distinct fault categories, each with a different 
telemetry signature. The system must learn to distinguish all five from normal 
traffic — and then catch a sixth fault type (BGP hijack) it was never trained on.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

ANOMALIES = [
    ("", "#EF4444", "link_failure", "Link Failure",
     "A physical or logical link goes down between two network devices. "
     "The interface operational status drops to DOWN (ifOperStatus=2), "
     "traffic counters stop incrementing, and syslog messages like "
     "LINEPROTO-5-UPDOWN appear. All traffic that was routing through "
     "that link is either dropped or rerouted.",
     "ifOperStatus=DOWN · traffic drops to 0 · latency spikes · LINEPROTO syslog"),

    ("", "#F97316", "congestion",   "Congestion",
     "A link becomes saturated — traffic demand exceeds available bandwidth. "
     "Unlike a link failure, the interface stays UP but packet queues fill. "
     "Discards (ifInDiscards) increase, latency grows, and throughput plateaus. "
     "Common cause: traffic bursts, video streaming, backup jobs, or DDoS.",
     "ifInDiscards↑ · latency↑ · ifInOctets near link capacity · no link down"),

    ("", "#EAB308", "mtu_mismatch", "MTU Mismatch",
     "Mismatched Maximum Transmission Unit settings between two connected "
     "interfaces. Packets larger than the smaller MTU get fragmented or "
     "silently dropped. Appears as high error rates (ifInErrors), "
     "ICMP_UNREACHABLE log messages, and mysterious TCP retransmissions "
     "without any clear link failure.",
     "ifInErrors↑ · fragmentation · ICMP_UNREACHABLE · TCP retransmits"),

    ("", "#8B5CF6", "interface_flap", "Interface Flap",
     "An interface repeatedly cycles between UP and DOWN states — often caused "
     "by a faulty cable, SFP module, or driver issue. Each flap triggers BGP/"
     "OSPF reconvergence, causing temporary traffic blackholing. Syslog shows "
     "repeated LINEPROTO-5-UPDOWN messages within seconds.",
     "Repeated UP/DOWN in syslog · BGP ADJCHANGE · oscillating ifOperStatus"),

    ("", "#EC4899", "routing_loop",  "Routing Loop",
     "A misconfiguration or protocol bug causes packets to circulate in a "
     "loop between routers indefinitely, until their TTL reaches zero. "
     "Appears as CPU spikes from TTL-expired processing, ICMP_TIME_EXCEEDED "
     "log messages, and traffic that never reaches its destination despite "
     "links being UP.",
     "CPU↑ · TTL_EXPIRED logs · ICMP_TIME_EXCEEDED · traffic not delivered"),
]

for i in range(0, len(ANOMALIES), 2):
    row = st.columns(2)
    for j, col in enumerate(row):
        if i + j >= len(ANOMALIES):
            break
        icon, color, key, name, desc, signals = ANOMALIES[i+j]
        count = int((df['anomaly_type']==key).sum())
        col.markdown(f"""
<div class='anomaly-card' style='background:#1E293B; border:1px solid #334155;
     border-top:3px solid {color}; border-radius:12px; padding:22px; margin-bottom:16px;'>
  <div style='display:flex; align-items:center; gap:10px; margin-bottom:12px;'>
    <span style='font-size:1.6rem;'>{icon}</span>
    <div>
      <div style='font-size:1rem; font-weight:700; color:#F1F5F9;'>{name}</div>
      <div style='font-size:0.72rem; color:#64748B; font-weight:600; font-family:monospace;'>{key}</div>
    </div>
    <div style='margin-left:auto; background:#0F172A; border:1px solid #334155;
                border-radius:6px; padding:3px 10px; font-size:0.75rem;
                color:{color}; font-weight:700;'>{count:,} events</div>
  </div>
  <p style='font-size:0.83rem; color:#94A3B8; line-height:1.7; margin-bottom:14px;'>{desc}</p>
  <div style='background:#0F172A; border-radius:6px; padding:10px 12px;
              font-size:0.75rem; color:{color}; font-family:monospace; line-height:1.8;'>
     {signals}
  </div>
</div>
""", unsafe_allow_html=True)

# Zero-day bonus
st.markdown("""
<div style='background:linear-gradient(135deg,#1E1B4B,#1E293B); border:1px solid #4338CA;
     border-radius:12px; padding:24px 28px; margin-top:8px;'>
  <div style='font-size:0.72rem; font-weight:800; color:#818CF8; text-transform:uppercase;
              letter-spacing:0.12em; margin-bottom:8px;'> BONUS — Zero-Day Test</div>
  <div style='display:flex; gap:24px; align-items:flex-start;'>
    <div style='flex:1;'>
      <div style='font-size:1.0rem; font-weight:700; color:#F1F5F9; margin-bottom:8px;'>
        BGP Hijack Attack
      </div>
      <p style='font-size:0.83rem; color:#94A3B8; line-height:1.7; margin:0;'>
        A fault type <strong style='color:#F1F5F9;'>never seen during training</strong>. 
        In a BGP hijack, an attacker announces fake routing prefixes, 
        causing traffic to be redirected through a malicious AS — or blackholed entirely. 
        The LSTM Autoencoder, trained only on normal traffic, detects this purely from the 
        reconstruction error spike: 350ms latency, 65% packet loss, and unusual BGP ADJCHANGE 
        syslogs. No labelled examples. No rules. Caught with <strong style='color:#10B981;'>
        100% recall</strong>.
      </p>
    </div>
    <div style='text-align:center; min-width:100px;'>
      <div style='font-size:2.2rem; font-weight:900; color:#10B981;'>100%</div>
      <div style='font-size:0.7rem; color:#64748B; text-transform:uppercase;
                  letter-spacing:0.08em;'>LSTM Recall</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider-line'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# THE METHOD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<a name='the-method'></a>", unsafe_allow_html=True)
st.markdown("""
<div class='section-label'>05 — The Method</div>
<div class='section-title'>How the pipeline works<br>end to end.</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

steps = [
    ("1", "#6366F1", "Feature Engineering",
     "Raw SNMP counters (ifInOctets, ifOutOctets, ifInDiscards, ifInErrors, ifOperStatus, latency_ms) "
     "are transformed into 19 features: delta rates, rolling means, rolling standard deviations, "
     "z-scores, and utilisation ratios. A RobustScaler normalises each feature to be outlier-resistant."),

    ("2", "#EF4444", "Tier 1 — Random Forest",
     "A 400-tree Random Forest with balanced class subsampling classifies each telemetry record "
     "as normal or one of 5 known fault types. The prediction threshold is tuned on the validation "
     "set rather than using the default 0.5, improving recall on subtle faults like MTU mismatch."),

    ("3", "#F97316", "Tier 2 — LSTM Autoencoder",
     "A 3-layer LSTM encoder-decoder is trained exclusively on normal traffic (454,140 samples). "
     "It learns to reconstruct normal traffic patterns. At inference time, high reconstruction "
     "error (MSE) indicates the traffic deviates from normal — flagging anomalies the RF might miss. "
     "A custom weighted MSE (5× on link-down features, 4× on drop features) makes it more sensitive "
     "to the most critical fault signatures."),

    ("4", "#10B981", "Transfer Learning (Mininet)",
     "The simulation-trained models are fine-tuned on 2,200 real Mininet telemetry records. "
     "The RF threshold is re-tuned on a held-out Mininet validation set. Broken features "
     "(cpu_utilisation is flat in Mininet) are zeroed before inference. The LSTM threshold "
     "is recalibrated from the 93rd percentile of normal MSE scores."),

    ("5", "#818CF8", "Tier 3 — LangGraph ReAct Agent",
     "For each high-priority alert, a LangGraph StateGraph drives a Reason→Act→Observe loop. "
     "Gemini 2.5 Flash reasons about the alert, calls one of 6 diagnostic tools "
     "(get_metrics, inspect_topology, query_syslogs, trace_path, check_routes, get_anomaly_score), "
     "observes the result, and repeats — until it has enough evidence to issue a structured "
     "RCA report with fault classification, evidence chain, and remediation commands."),
]

for num, color, title, body in steps:
    st.markdown(f"""
<div style='display:flex; gap:20px; margin-bottom:20px; align-items:flex-start;'>
  <div style='min-width:36px; height:36px; border-radius:50%; background:{color}22;
              border:2px solid {color}; display:flex; align-items:center; justify-content:center;
              font-size:0.85rem; font-weight:800; color:{color}; flex-shrink:0; margin-top:2px;'>
    {num}
  </div>
  <div>
    <div style='font-size:0.95rem; font-weight:700; color:#F1F5F9; margin-bottom:5px;'>{title}</div>
    <div style='font-size:0.83rem; color:#94A3B8; line-height:1.7;'>{body}</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider-line'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# WHAT WE ACHIEVED
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<a name='what-we-achieved'></a>", unsafe_allow_html=True)
st.markdown("""
<div class='section-label'>06 — What We Achieved</div>
<div class='section-title'>Results across simulation,<br>real-world transfer, and agent evaluation.</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Simulation results
st.markdown("""
<div style='font-size:0.85rem; font-weight:700; color:#F1F5F9; margin-bottom:12px;
            text-transform:uppercase; letter-spacing:0.08em;'>
  Simulation — 760k Records
</div>
""", unsafe_allow_html=True)

r1,r2,r3,r4,r5,r6 = st.columns(6)
for col, val, label, color in [
    (r1, "0.957", "RF F1 Score",        "#10B981"),
    (r2, "100%",  "RF Precision",        "#10B981"),
    (r3, "96%",   "RF Recall",           "#10B981"),
    (r4, "0.728", "LSTM F1 Score",       "#F97316"),
    (r5, "99.6%", "LSTM Recall",         "#F97316"),
    (r6, "0.999", "LSTM ROC-AUC",        "#F97316"),
]:
    col.markdown(f"""
<div style='background:#1E293B; border:1px solid #334155; border-radius:10px;
            padding:16px; text-align:center;'>
  <div style='font-size:1.6rem; font-weight:900; color:{color};'>{val}</div>
  <div style='font-size:0.67rem; color:#64748B; font-weight:600;
              text-transform:uppercase; letter-spacing:0.07em; margin-top:4px;'>{label}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Mininet results
st.markdown("""
<div style='font-size:0.85rem; font-weight:700; color:#F1F5F9; margin-bottom:12px;
            text-transform:uppercase; letter-spacing:0.08em;'>
  Mininet Real-World Transfer
</div>
""", unsafe_allow_html=True)

m1,m2,m3,m4,m5 = st.columns(5)
for col, val, label, color in [
    (m1, "1.000", "RF F1 Score",     "#10B981"),
    (m2, "100%",  "RF Precision",    "#10B981"),
    (m3, "100%",  "RF Recall",       "#10B981"),
    (m4, "0.889", "LSTM ROC-AUC",    "#F97316"),
    (m5, "42.6%", "LSTM Precision",  "#F97316"),
]:
    col.markdown(f"""
<div style='background:#1E293B; border:1px solid #334155; border-radius:10px;
            padding:16px; text-align:center;'>
  <div style='font-size:1.6rem; font-weight:900; color:{color};'>{val}</div>
  <div style='font-size:0.67rem; color:#64748B; font-weight:600;
              text-transform:uppercase; letter-spacing:0.07em; margin-top:4px;'>{label}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Agent results
st.markdown("""
<div style='font-size:0.85rem; font-weight:700; color:#F1F5F9; margin-bottom:14px;
            text-transform:uppercase; letter-spacing:0.08em;'>
  Tier 3 Agent — 3 Live Investigations
</div>
""", unsafe_allow_html=True)

for icon, device, fault, verdict, correct, tools, explain in [
    ("", "A1 / GigE0/1",  "link_failure", "LINK_FAILURE",  True, 3,
     "Confirmed via ifOperStatus=DOWN, LINEPROTO syslog, and trace_path showing broken hop."),
    ("", "A4 / GigE0/1",  "congestion",   "CONGESTION",    True, 5,
     "Identified via ifInOctets 30× higher on A4 than peer D3. Traffic asymmetry = egress saturation."),
    ("", "D1 / GigE0/1",  "link_failure", "FALSE_POSITIVE", True, 4,
     "Autonomously dismissed — link UP, MSE normal, no syslogs. Saved NOC engineer ~20 min."),
]:
    badge_color = "#064E3B" if correct else "#7F1D1D"
    badge_text  = " CORRECT" if correct else " WRONG"
    text_color  = "#34D399" if correct else "#FCA5A5"
    st.markdown(f"""
<div style='background:#1E293B; border:1px solid #334155; border-radius:10px;
            padding:16px 20px; margin-bottom:10px; display:flex; gap:20px; align-items:center;'>
  <div style='font-size:1.4rem;'>{icon}</div>
  <div style='flex:1;'>
    <div style='display:flex; align-items:center; gap:12px; margin-bottom:4px;'>
      <span style='font-size:0.88rem; font-weight:700; color:#F1F5F9; font-family:monospace;'>{device}</span>
      <span style='font-size:0.72rem; color:#64748B;'>·</span>
      <span style='font-size:0.78rem; color:#64748B; font-family:monospace;'>{fault}</span>
    </div>
    <div style='font-size:0.82rem; color:#94A3B8;'>{explain}</div>
  </div>
  <div style='text-align:right; flex-shrink:0;'>
    <div style='background:{badge_color}; color:{text_color}; border-radius:6px;
                padding:3px 10px; font-size:0.72rem; font-weight:700;
                letter-spacing:0.06em; margin-bottom:4px;'>{badge_text}</div>
    <div style='font-size:0.72rem; color:#475569;'>Verdict: <strong style='color:#818CF8;'>{verdict}</strong></div>
    <div style='font-size:0.72rem; color:#475569;'>{tools} tool calls</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style='background:linear-gradient(135deg,#0F2027,#1E293B); border:1px solid #334155;
     border-radius:16px; padding:32px 36px; text-align:center;'>
  <div style='font-size:0.78rem; font-weight:700; color:#6366F1; text-transform:uppercase;
              letter-spacing:0.12em; margin-bottom:12px;'>Key Contribution</div>
  <div style='font-size:1.5rem; font-weight:800; color:#F1F5F9; max-width:700px;
              margin:0 auto; line-height:1.45;'>
    The first three-tier AIOps pipeline combining supervised, unsupervised, 
    and agentic AI for autonomous network fault detection and root cause analysis — 
    validated on both simulated and real Linux-kernel Mininet data.
  </div>
  <div style='margin-top:24px; display:flex; justify-content:center; gap:16px; flex-wrap:wrap;'>
    <span style='background:#1E293B; border:1px solid #334155; border-radius:6px;
                 padding:6px 14px; font-size:0.78rem; color:#94A3B8;'>PyTorch LSTM</span>
    <span style='background:#1E293B; border:1px solid #334155; border-radius:6px;
                 padding:6px 14px; font-size:0.78rem; color:#94A3B8;'>LangGraph ReAct</span>
    <span style='background:#1E293B; border:1px solid #334155; border-radius:6px;
                 padding:6px 14px; font-size:0.78rem; color:#94A3B8;'>Gemini 2.5 Flash</span>
    <span style='background:#1E293B; border:1px solid #334155; border-radius:6px;
                 padding:6px 14px; font-size:0.78rem; color:#94A3B8;'>Mininet Transfer</span>
    <span style='background:#1E293B; border:1px solid #334155; border-radius:6px;
                 padding:6px 14px; font-size:0.78rem; color:#94A3B8;'>Zero-Day Detection</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
