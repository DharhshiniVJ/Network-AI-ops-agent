"""
Tool for inspecting topology.
"""
from typing import Dict, Any
from agent.tools.registry import diagnostic_tool

# Reference to the network state
_network_state = None

def set_network_state(state: Dict[str, Any]):
    """Injects the global network state."""
    global _network_state
    _network_state = state

@diagnostic_tool(
    name="inspect_topology",
    description="Returns the network topology showing all devices, their connections, link statuses (UP/DOWN), bandwidth, and interface configurations. Use this to understand the network layout and identify down links or topology issues."
)
def inspect_topology(device_id: str = "all") -> dict:
    """Inspects the network topology."""
    if _network_state is None or "topology" not in _network_state:
        return {"error": "Network state not initialized or missing topology data."}
    
    topology = _network_state["topology"]
    
    if device_id == "all":
        return {"topology": topology}
    
    # Simple extraction logic: node info and connected links
    nodes = topology.get("nodes", [])
    links = topology.get("links", [])
    
    device_info = next((n for n in nodes if n.get("id") == device_id), None)
    if not device_info:
        return {"error": f"Device '{device_id}' not found in topology."}
        
    connected_links = [l for l in links if l.get("source") == device_id or l.get("target") == device_id]
    
    return {
        "device": device_info,
        "connected_links": connected_links
    }
