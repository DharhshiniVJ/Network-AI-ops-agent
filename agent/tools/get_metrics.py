"""
Tool for fetching metrics.
"""
from typing import Dict, Any
from agent.tools.registry import diagnostic_tool

_network_state = None

def set_network_state(state: Dict[str, Any]):
    """Injects the global network state."""
    global _network_state
    _network_state = state

@diagnostic_tool(
    name="get_metrics",
    description="Fetches current health metrics for a network device including CPU utilization, packet loss, interface errors, discards, memory usage, and latency. Use this to check if a specific device is under stress or experiencing performance degradation."
)
def get_metrics(device_id: str) -> dict:
    """Fetches metrics for a specific device."""
    if _network_state is None or "metrics" not in _network_state:
        return {"error": "Network state not initialized or missing metrics."}
    
    metrics = _network_state["metrics"].get(device_id)
    if not metrics:
        return {"error": f"No metrics found for device '{device_id}'"}
    
    return {"device_id": device_id, "metrics": metrics}

def get_all_device_metrics() -> dict:
    """Fetches metrics for all devices."""
    if _network_state is None or "metrics" not in _network_state:
        return {"error": "Network state not initialized or missing metrics."}
    
    return {"all_metrics": _network_state["metrics"]}
