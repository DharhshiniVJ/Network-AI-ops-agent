import pandas as pd
import numpy as np
import torch
import joblib
from datetime import datetime, timedelta, timezone
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from network.topology import NetworkTopology
from network.anomaly_injector import AnomalyInjector
from network.telemetry_generator import TelemetryEngine
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

print("=== Generating Zero-Day (BGP Hijack) Dataset ===")
base_time = datetime(2026, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
topo = NetworkTopology()
topo.build_hierarchical_wan()
links = [("R1", "D1"), ("R2", "D2"), ("D1", "A1"), ("D3", "A4"), ("R1", "R2")]

injector = AnomalyInjector(topo)
anomalies = [
    injector.create_bgp_hijack(links[0], base_time + timedelta(hours=1), 15.0),
    injector.create_bgp_hijack(links[1], base_time + timedelta(hours=3), 15.0),
    injector.create_bgp_hijack(links[2], base_time + timedelta(hours=5), 15.0),
    injector.create_bgp_hijack(links[3], base_time + timedelta(hours=7), 15.0),
    injector.create_bgp_hijack(links[4], base_time + timedelta(hours=9), 15.0),
]

engine = TelemetryEngine(topo)
snmp_df, _, _ = engine.generate_telemetry_window(
    start_time=base_time, duration_seconds=43200, interval_seconds=15, anomalies=anomalies)

# Crucial: rename time to timestamp like main.py does!
snmp_df.rename(columns={'time': 'timestamp'}, inplace=True)

# Also MUST sort chronologically for evaluation otherwise sequences are broken
snmp_df = snmp_df.sort_values(["device_id", "interface_id", "timestamp"]).reset_index(drop=True)

print(f"Generated {len(snmp_df)} records.")
print(f"Total BGP Hijack anomalies: {snmp_df['is_anomaly'].sum()}")

print("\n=== Loading Trained Models ===")
lstm = torch.load("data/models/lstm_autoencoder.pth", map_location="cpu", weights_only=False)
lstm.device = torch.device("cpu")
if hasattr(lstm, 'model'):
    lstm.model.to("cpu")
rf = joblib.load("data/models/random_forest.pkl")

print("\n=== Evaluating Zero-Day Anomaly Detection ===")

print("\n--- Deep LSTM-Autoencoder (Unsupervised) ---")
lstm_preds, lstm_scores = lstm.predict(snmp_df)

# Because we sorted snmp_df before predict, the indices should match exactly
y_true = snmp_df['is_anomaly'].values

lstm_pr = average_precision_score(y_true, lstm_scores)
print(f"LSTM PR-AUC: {lstm_pr:.4f}")
print("LSTM Classification Report:")
print(classification_report(y_true, lstm_preds, digits=4, zero_division=0))

print("\n--- Random Forest (Supervised) ---")
rf_preds, rf_scores = rf.predict(snmp_df)
rf_pr = average_precision_score(y_true, rf_scores)
print(f"RF PR-AUC: {rf_pr:.4f}")
print("RF Classification Report:")
print(classification_report(y_true, rf_preds, digits=4, zero_division=0))

