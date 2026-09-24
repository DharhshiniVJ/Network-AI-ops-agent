import pandas as pd
import numpy as np
import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from network.topology import NetworkTopology
from network.anomaly_injector import AnomalyInjector
from network.telemetry_generator import TelemetryEngine
from datetime import datetime, timedelta, timezone

base_time = datetime(2026, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
topo = NetworkTopology()
topo.build_hierarchical_wan()
links = [("R1", "D1")]

injector = AnomalyInjector(topo)
anomalies = [injector.create_bgp_hijack(links[0], base_time + timedelta(hours=1), 15.0)]
engine = TelemetryEngine(topo)
snmp_df, _, _ = engine.generate_telemetry_window(
    start_time=base_time, duration_seconds=7200, interval_seconds=15, anomalies=anomalies)

lstm = torch.load("data/models/lstm_autoencoder.pth", map_location="cpu", weights_only=False)
lstm.device = torch.device("cpu")
lstm.model.to("cpu")

_, mse = lstm.predict(snmp_df)
snmp_df['mse'] = mse

print(f"Optimal Threshold: {lstm.optimal_threshold:.6f}")
print("\nNormal samples MSE stats:")
print(snmp_df[snmp_df['is_anomaly'] == 0]['mse'].describe())
print("\nAnomaly samples MSE stats:")
print(snmp_df[snmp_df['is_anomaly'] == 1]['mse'].describe())

