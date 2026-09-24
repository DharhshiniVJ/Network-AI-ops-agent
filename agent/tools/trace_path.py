"""
Tool for tracing path.
"""
from typing import Dict, Any
from agent.tools.registry import diagnostic_tool

_network_state = None

def set_network_state(state: Dict[str, Any]):
    """Injects the global network state."""
    global _network_state
    _network_state = state

@diagnostic_tool(
    name="trace_path",
    description="Simulates a traceroute from source to destination device, showing each hop along the path with its status (OK/PACKET_DROP/TIMEOUT/LOOP_DETECTED), latency, and any issues encountered. Use this to find exactly where packets are being dropped or delayed."
)
def trace_path(source: str, destination: str) -> dict:
    """Simulates a traceroute from source to destination."""
    if _network_state is None or "topology" not in _network_state:
        return {"error": "Network state not initialized."}
        
    traces = _network_state.get("mock_traces", {})
    trace_key = f"{source}-{destination}"
    
    if trace_key in traces:
        return traces[trace_key]
        
    return {
        "source": source,
        "destination": destination,
        "hops": [],
        "completed": False,
        "total_latency_ms": 0,
        "error": "Path computation not implemented in mock network state."
    }
