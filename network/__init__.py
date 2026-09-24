"""
Network simulation package.

Provides topology modeling, synthetic telemetry generation,
and controlled anomaly injection for network RCA research.
"""

from network.topology import NetworkTopology
from network.telemetry_generator import TelemetryEngine
from network.anomaly_injector import AnomalyInjector

__all__ = ["NetworkTopology", "TelemetryEngine", "AnomalyInjector"]
