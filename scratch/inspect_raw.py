import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from network.topology import NetworkTopology
from network.anomaly_injector import AnomalyInjector
from network.telemetry_generator import TelemetryEngine

base_time = datetime(2026, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
topo = NetworkTopology()
topo.build_hierarchical_wan()
links = [("R1", "D1")]

injector = AnomalyInjector(topo)
anomalies = [injector.create_bgp_hijack(links[0], base_time + timedelta(hours=1), 15.0)]
engine = TelemetryEngine(topo)
snmp_df, _, _ = engine.generate_telemetry_window(
    start_time=base_time, duration_seconds=7200, interval_seconds=15, anomalies=anomalies)

print("Raw Anomaly Samples:")
print(snmp_df[snmp_df['is_anomaly'] == 1][['device_id', 'interface_id', 'ifInOctets', 'ifOutOctets', 'latency_ms', 'packet_loss_rate']].head(10))

