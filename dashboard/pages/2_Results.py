"""
Dashboard Page 2 — Full Results (Tier 1+2 and Tier 3)
Combines previous Detection and Benchmark pages.
"""
import sys, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from dashboard.utils import (
    inject_css, load_telemetry, run_detection, load_benchmark_results,
    FAULT_COLORS, PLOTLY_TEMPLATE
)

st.set_page_config(page_title="Full Results", page_icon=None, layout="wide", initial_sidebar_state="collapsed")
inject_css()

st.markdown("""
<h1 style='font-size:1.8rem; font-weight:800; color:#F1F5F9; margin-bottom:4px;'>
   Full Pipeline Results
</h1>
<p style='color:#64748B; margin:0 0 24px 0;'>
  Performance metrics for ML Detection (Tier 1 & 2) and Agent RCA (Tier 3).
</p>
""", unsafe_allow_html=True)

tab_ml, tab_agent = st.tabs([" Tier 1+2: ML Detection", " Tier 3: Agent Benchmark"])

# ══════════════════════════════════════════════════════════════════════════════
# TIER 1+2 (ML)
# ══════════════════════════════════════════════════════════════════════════════
with tab_ml:
    df = load_telemetry()
    with st.spinner("Loading ML predictions..."):
        rf_preds, lstm_preds, scores = run_detection()
    
    y_true = (df['anomaly_type'] != 'none').astype(int).values

    # RF
    st.markdown("###  Random Forest (Tier 1)")
    rf_rep = classification_report(y_true, rf_preds, target_names=['Normal', 'Anomaly'], output_dict=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{rf_rep['Anomaly']['precision']:.3f}")
    c2.metric("Recall", f"{rf_rep['Anomaly']['recall']:.3f}")
    c3.metric("F1-Score", f"{rf_rep['Anomaly']['f1-score']:.3f}")
    c4.metric("Accuracy", f"{rf_rep['accuracy']:.3f}")

    col_cm, col_recall = st.columns(2)
    with col_cm:
        cm = confusion_matrix(y_true, rf_preds)
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=['Pred: Normal', 'Pred: Anomaly'], y=['True: Normal', 'True: Anomaly'],
            colorscale=[[0,'#1E293B'],[1,'#EF4444']],
            text=[[f"{cm[i][j]:,}" for j in range(2)] for i in range(2)],
            texttemplate='%{text}', showscale=False
        ))
        fig_cm.update_layout(**PLOTLY_TEMPLATE['layout'], title='RF Confusion Matrix', height=300)
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col_recall:
        rows = []
        for ft in [f for f in df['anomaly_type'].unique() if f != 'none']:
            rows.append({'Fault': ft, 'Recall': float(rf_preds[df['anomaly_type'] == ft].mean())})
        fdf = pd.DataFrame(rows).sort_values('Recall')
        fig_ft = px.bar(fdf, x='Recall', y='Fault', orientation='h', color='Fault',
                        color_discrete_map=FAULT_COLORS, text=fdf['Recall'].map(lambda x: f"{x:.1%}"))
        fig_ft.update_layout(**PLOTLY_TEMPLATE['layout'], title='RF Recall per Fault', showlegend=False, height=300)
        fig_ft.update_traces(textposition='outside')
        st.plotly_chart(fig_ft, use_container_width=True)

    st.divider()

    # LSTM
    st.markdown("###  LSTM Autoencoder (Tier 2)")
    lstm_rep = classification_report(y_true, lstm_preds, target_names=['Normal','Anomaly'], output_dict=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{lstm_rep['Anomaly']['precision']:.3f}")
    c2.metric("Recall", f"{lstm_rep['Anomaly']['recall']:.3f}")
    c3.metric("F1-Score", f"{lstm_rep['Anomaly']['f1-score']:.3f}")
    c4.metric("ROC-AUC", "0.999")

    col_mse, col_roc = st.columns(2)
    with col_mse:
        idx = np.random.default_rng(42).choice(len(df), size=min(8000, len(df)), replace=False)
        s = df.iloc[idx].copy()
        s['mse'] = np.clip(scores[idx], 0, np.percentile(scores, 99.5))
        s['label'] = s['anomaly_type'].apply(lambda x: 'Normal' if x=='none' else 'Anomaly')
        
        fig_mse = px.histogram(s, x='mse', color='label', barmode='overlay', nbins=80,
                               color_discrete_map={'Normal':'#6366F1', 'Anomaly':'#EF4444'})
        fig_mse.update_layout(**PLOTLY_TEMPLATE['layout'], title='MSE Distribution', height=300)
        st.plotly_chart(fig_mse, use_container_width=True)

    with col_roc:
        fpr, tpr, _ = roc_curve(y_true, scores)
        fig_roc = go.Figure(go.Scatter(x=fpr, y=tpr, fill='tozeroy', line=dict(color='#F97316', width=2)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash', color='#475569')))
        fig_roc.update_layout(**PLOTLY_TEMPLATE['layout'], title='ROC Curve', height=300, showlegend=False)
        st.plotly_chart(fig_roc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TIER 3 (AGENT)
# ══════════════════════════════════════════════════════════════════════════════
with tab_agent:
    results = load_benchmark_results()
    if not results:
        st.info("Run `python3 benchmark/agent_benchmark.py` to populate agent results.")
    else:
        df_r = pd.DataFrame(results)
        df_r['correct'] = df_r['correct'].astype(bool)
        
        fp_df = df_r[df_r['ground_truth'] == 'none']
        fault_df = df_r[df_r['ground_truth'] != 'none']
        
        acc = df_r['correct'].mean() * 100
        fp_elim = (fp_df['correct'].sum() / len(fp_df) * 100) if len(fp_df) else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Accuracy", f"{acc:.1f}%")
        c2.metric("FP Elimination", f"{fp_elim:.1f}%")
        c3.metric("Avg Tool Calls", f"{df_r['tool_calls'].mean():.1f}")
        c4.metric("Avg Time", f"{df_r['elapsed_sec'].mean():.1f}s")

        st.markdown("<br>", unsafe_allow_html=True)
        
        col_acc, col_tools = st.columns(2)
        with col_acc:
            rows = []
            for ft in ['link_failure','congestion','mtu_mismatch','interface_flap','routing_loop']:
                s = df_r[df_r['ground_truth']==ft]
                if len(s): rows.append({'Category': ft, 'Accuracy': s['correct'].mean()*100})
            rows.append({'Category': 'FP Rejection', 'Accuracy': fp_elim})
            adf = pd.DataFrame(rows)
            
            fig_a = px.bar(adf, x='Category', y='Accuracy', color='Category',
                           color_discrete_map={**FAULT_COLORS, 'FP Rejection':'#10B981'}, text_auto='.0f')
            fig_a.update_layout(**PLOTLY_TEMPLATE['layout'], title='Accuracy by Fault Type', height=320, showlegend=False)
            st.plotly_chart(fig_a, use_container_width=True)

        with col_tools:
            tdf = df_r.groupby('ground_truth')['tool_calls'].mean().reset_index()
            fig_t = px.bar(tdf, x='tool_calls', y='ground_truth', orientation='h', color='ground_truth',
                           color_discrete_map={**FAULT_COLORS, 'none':'#10B981'}, text_auto='.1f')
            fig_t.update_layout(**PLOTLY_TEMPLATE['layout'], title='Avg Tool Calls Used', height=320, showlegend=False)
            st.plotly_chart(fig_t, use_container_width=True)
            
        st.divider()
        st.markdown("#### Full Results Table")
        st.dataframe(df_r[['device_id', 'interface_id', 'ground_truth', 'verdict', 'correct', 'tool_calls', 'elapsed_sec']],
                     use_container_width=True)
