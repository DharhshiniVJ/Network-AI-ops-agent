"""
Quick eval — loads the already-trained LSTM and RF from disk,
runs inference on the test split, and prints the comparison table.
No re-training needed.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import joblib, torch
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

df = pd.read_parquet("data/snmp_telemetry.parquet")
if 'is_anomaly' not in df.columns:
    df['is_anomaly'] = 0
if 'timestamp' in df.columns:
    df = df.sort_values('timestamp')

n = len(df)
test_df = df.iloc[int(0.8 * n):]
print(f"Test set: {len(test_df):,} rows  |  anomalies: {test_df['is_anomaly'].sum():,}")

# ── LSTM ──────────────────────────────────────────────────────────────────────
print("\n[LSTM] Loading saved model...")
lstm = torch.load("data/models/lstm_autoencoder.pth", map_location="cpu", weights_only=False)
lstm.device = torch.device("cpu")
lstm.model  = lstm.model.to("cpu")
preds_lstm, scores_lstm = lstm.predict(test_df)
_, y_true = lstm.pipeline.transform(test_df)

roc  = roc_auc_score(y_true, scores_lstm)
pr   = average_precision_score(y_true, scores_lstm)
rep  = classification_report(y_true, preds_lstm, output_dict=True, zero_division=0)
cls1 = rep.get('1', rep.get('1.0', {}))

print(f"\n  ROC-AUC:   {roc:.4f}")
print(f"  PR-AUC:    {pr:.4f}")
print(f"  Precision: {cls1.get('precision', 0):.4f}")
print(f"  Recall:    {cls1.get('recall', 0):.4f}")
print(f"  F1-Score:  {cls1.get('f1-score', 0):.4f}")

# ── RF ────────────────────────────────────────────────────────────────────────
print("\n[Random Forest] Loading saved model...")
rf = joblib.load("data/models/random_forest.pkl")
preds_rf, scores_rf = rf.predict(test_df)

roc_rf  = roc_auc_score(y_true, scores_rf)
pr_rf   = average_precision_score(y_true, scores_rf)
rep_rf  = classification_report(y_true, preds_rf, output_dict=True, zero_division=0)
cls1_rf = rep_rf.get('1', rep_rf.get('1.0', {}))

print(f"\n  ROC-AUC:   {roc_rf:.4f}")
print(f"  PR-AUC:    {pr_rf:.4f}")
print(f"  Precision: {cls1_rf.get('precision', 0):.4f}")
print(f"  Recall:    {cls1_rf.get('recall', 0):.4f}")
print(f"  F1-Score:  {cls1_rf.get('f1-score', 0):.4f}")

print("\nDone.")
