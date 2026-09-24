"""
Diagnostic tools for the RCA agent.
"""
from agent.tools.registry import TOOL_REGISTRY, TOOLS_SCHEMA, diagnostic_tool
from agent.tools.get_metrics import get_metrics, get_all_device_metrics, set_network_state as set_metrics_state
from agent.tools.inspect_topology import inspect_topology, set_network_state as set_topology_state
from agent.tools.check_routes import check_routes, set_network_state as set_routes_state
from agent.tools.trace_path import trace_path, set_network_state as set_trace_state
from agent.tools.query_logs import query_logs, set_network_state as set_logs_state

def inject_network_state_to_tools(network_state):
    """Injects the shared network state into all diagnostic tools."""
    set_metrics_state(network_state)
    set_topology_state(network_state)
    set_routes_state(network_state)
    set_trace_state(network_state)
    set_logs_state(network_state)
