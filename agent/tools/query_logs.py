"""
Tool for querying logs.
"""
from typing import Dict, Any
from agent.tools.registry import diagnostic_tool

_network_state = None

def set_network_state(state: Dict[str, Any]):
    """Injects the global network state."""
    global _network_state
    _network_state = state

@diagnostic_tool(
    name="query_logs",
    description="Retrieves recent system logs (syslog messages) for a network device, including error messages, warnings, and state change notifications. Use this to find error patterns, link state changes, and diagnostic messages."
)
def query_logs(device_id: str, severity: str = "all") -> dict:
    """Queries logs for a given device."""
    if _network_state is None or "logs" not in _network_state:
        return {"error": "Network state not initialized or missing logs."}
        
    device_logs = _network_state["logs"].get(device_id, [])
    
    if severity != "all":
        device_logs = [log for log in device_logs if log.get("severity", "").lower() == severity.lower()]
        
    recent_logs = device_logs[-20:] if len(device_logs) > 20 else device_logs
    
    return {
        "device": device_id,
        "log_count": len(recent_logs),
        "logs": recent_logs
    }
