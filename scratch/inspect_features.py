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

X_robust, _ = lstm.pipeline.transform(snmp_df)
X_scaled = lstm.dl_scaler.transform(X_robust)
feature_names = lstm.pipeline.get_feature_names()

df_scaled = pd.DataFrame(X_scaled, columns=feature_names)
df_scaled['is_anomaly'] = snmp_df['is_anomaly'].values

print("Mean Scaled Features for Normal:")
print(df_scaled[df_scaled['is_anomaly'] == 0].mean())

print("\nMean Scaled Features for Anomaly:")
print(df_scaled[df_scaled['is_anomaly'] == 1].mean())

