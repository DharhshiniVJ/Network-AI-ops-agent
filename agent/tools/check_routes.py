"""
Tool for checking routes.
"""
from typing import Dict, Any
from agent.tools.registry import diagnostic_tool

_network_state = None

def set_network_state(state: Dict[str, Any]):
    """Injects the global network state."""
    global _network_state
    _network_state = state

@diagnostic_tool(
    name="check_routes",
    description="Inspects the routing table for a network device, showing destination networks, next-hop addresses, route metrics, and routing protocols (OSPF/BGP/STATIC). Use this to detect routing anomalies, missing routes, or suboptimal paths."
)
def check_routes(device_id: str) -> dict:
    """Checks the routing table for a device."""
    if _network_state is None or "routes" not in _network_state:
        return {"error": "Network state not initialized or missing routes data."}
    
    routes = _network_state["routes"].get(device_id)
    if routes is None:
        return {"error": f"No routes found for device '{device_id}'"}
        
    return {"device_id": device_id, "routes": routes}
