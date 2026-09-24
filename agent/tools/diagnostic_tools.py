"""
Diagnostic Tools for the LangGraph RCA Agent
=============================================
Each tool is a standard LangChain @tool decorated function.
All tools read from a shared NetworkState object that is populated
before the agent runs by loading the real telemetry/topology data.

Tools:
  get_metrics        — SNMP interface stats for a device
  inspect_topology   — Neighbors, link statuses, interface names
  query_syslogs      — Recent syslog entries for a device
  trace_path         — End-to-end path trace between two nodes
  check_routes       — Routing table / shortest path analysis
  get_anomaly_score  — LSTM MSE score for a specific interface
"""

from __future__ import annotations
import json
from typing import Optional
from langchain_core.tools import tool


# ─── Shared State (injected before agent runs) ────────────────────────────────

class NetworkState:
    """
    Singleton container for all live network data.
    Populated once from telemetry CSV + topology before the agent starts.
    """
    topology: dict   = {}   # node-link dict from NetworkTopology.to_dict()
    metrics:  dict   = {}   # {device_id: {interface_id: {stat: value}}}
    syslogs:  list   = []   # list of syslog strings
    scores:   dict   = {}   # {device_id: {interface_id: mse_score}}


_state = NetworkState()


def inject_network_state(topology: dict, metrics: dict,
                         syslogs: list, scores: dict):
    """Called by main.py before invoking the agent."""
    _state.topology = topology
    _state.metrics  = metrics
    _state.syslogs  = syslogs
    _state.scores   = scores


# ─── Tool Implementations ─────────────────────────────────────────────────────

@tool
def get_metrics(device_id: str, interface_id: Optional[str] = None) -> str:
    """
    Fetch SNMP health metrics for a network device.
    Returns per-interface stats: ifOperStatus, latency_ms, packet_loss_rate,
    ifInOctets, ifOutOctets, ifInDiscards, ifInErrors, cpu_utilization.
    If interface_id is provided, returns stats for that interface only.
    """
    if not _state.metrics:
        return json.dumps({"error": "Network state not initialized."})

    device_data = _state.metrics.get(device_id)
    if not device_data:
        # List available devices to help the agent
        return json.dumps({
            "error": f"No metrics for '{device_id}'.",
            "available_devices": list(_state.metrics.keys())
        })

    if interface_id:
        intf_data = device_data.get(interface_id)
        if not intf_data:
            return json.dumps({
                "error": f"No metrics for '{device_id}/{interface_id}'.",
                "available_interfaces": list(device_data.keys())
            })
        return json.dumps({"device_id": device_id,
                           "interface_id": interface_id,
                           "metrics": intf_data})

    return json.dumps({"device_id": device_id, "interfaces": device_data})


@tool
def inspect_topology(device_id: str = "all") -> str:
    """
    Inspect network topology. Returns device info and its connected links
    with their current status (UP/DOWN), bandwidth, and interface names.
    Use device_id='all' for the full topology overview.
    """
    if not _state.topology:
        return json.dumps({"error": "Topology not initialized."})

    nodes = _state.topology.get("nodes", [])
    links = _state.topology.get("links", [])

    if device_id == "all":
        summary = {
            "total_devices": len(nodes),
            "total_links":   len(links),
            "devices": [{"id": n["id"],
                         "type": n.get("device_type", "unknown"),
                         "status": n.get("status", "UP")}
                        for n in nodes],
            "down_links": [{"from": l["source"], "to": l["target"],
                            "intf_u": l.get("intf_u"), "intf_v": l.get("intf_v")}
                           for l in links if l.get("status") == "DOWN"]
        }
        return json.dumps(summary)

    device_node = next((n for n in nodes if n.get("id") == device_id), None)
    if not device_node:
        return json.dumps({"error": f"Device '{device_id}' not found.",
                           "available": [n["id"] for n in nodes]})

    connected = [
        {
            "peer":       l["target"] if l["source"] == device_id else l["source"],
            "intf_local": l.get("intf_u") if l["source"] == device_id else l.get("intf_v"),
            "intf_peer":  l.get("intf_v") if l["source"] == device_id else l.get("intf_u"),
            "status":     l.get("status", "UP"),
            "bandwidth_mbps": l.get("bandwidth_mbps"),
        }
        for l in links
        if l["source"] == device_id or l["target"] == device_id
    ]

    return json.dumps({"device": device_node, "connected_links": connected})


@tool
def query_syslogs(device_id: str, keyword: Optional[str] = None,
                  last_n: int = 20) -> str:
    """
    Query syslog messages for a specific device.
    Optionally filter by keyword (e.g., 'DOWN', 'ERROR', 'BGP', 'MTU', 'LOOP').
    Returns the last_n matching log entries.
    """
    if not _state.syslogs:
        return json.dumps({"error": "No syslogs available.",
                           "hint": "Syslogs are generated during simulate phase."})

    # Filter by device
    device_logs = [s for s in _state.syslogs if device_id in s]

    if not device_logs:
        return json.dumps({"warning": f"No logs found for '{device_id}'.",
                           "total_logs": len(_state.syslogs),
                           "sample": _state.syslogs[:3]})

    # Optional keyword filter
    if keyword:
        device_logs = [s for s in device_logs if keyword.upper() in s.upper()]

    return json.dumps({
        "device_id":    device_id,
        "keyword":      keyword,
        "total_matches": len(device_logs),
        "logs":         device_logs[-last_n:]
    })


@tool
def trace_path(source_device: str, destination_device: str) -> str:
    """
    Trace the network path between two devices using topology routing.
    Returns each hop, its status, latency contribution, and whether any
    link along the path is DOWN or congested.
    """
    if not _state.topology:
        return json.dumps({"error": "Topology not initialized."})

    import networkx as nx

    nodes = _state.topology.get("nodes", [])
    links = _state.topology.get("links", [])

    G = nx.Graph()
    for n in nodes:
        G.add_node(n["id"], **n)
    for l in links:
        # Only add UP links for routing — DOWN links are failures
        if l.get("status", "UP") == "UP":
            G.add_edge(l["source"], l["target"],
                       weight=l.get("weight", 1),
                       bandwidth_mbps=l.get("bandwidth_mbps", 1000),
                       intf_u=l.get("intf_u"), intf_v=l.get("intf_v"))

    if source_device not in G or destination_device not in G:
        return json.dumps({"error": "Source or destination not in topology.",
                           "available": list(G.nodes())})

    try:
        path = nx.shortest_path(G, source_device, destination_device, weight="weight")
    except nx.NetworkXNoPath:
        # Try including DOWN links to identify the broken hop
        G_full = nx.Graph()
        for n in nodes:
            G_full.add_node(n["id"])
        for l in links:
            G_full.add_edge(l["source"], l["target"])

        try:
            full_path = nx.shortest_path(G_full, source_device, destination_device)
        except nx.NetworkXNoPath:
            return json.dumps({"error": "No path exists even in full topology.",
                               "source": source_device,
                               "destination": destination_device})

        # Identify broken hop
        broken_hops = []
        for i in range(len(full_path) - 1):
            link = next((l for l in links
                         if (l["source"] == full_path[i] and l["target"] == full_path[i+1]) or
                            (l["source"] == full_path[i+1] and l["target"] == full_path[i])), None)
            if link and link.get("status") == "DOWN":
                broken_hops.append({"hop": i+1,
                                    "from": full_path[i],
                                    "to":   full_path[i+1],
                                    "status": "DOWN"})

        return json.dumps({"reachable": False,
                           "intended_path": full_path,
                           "broken_hops":   broken_hops,
                           "diagnosis": "PATH_BROKEN — link failure detected"})

    # Annotate each hop with metrics
    hops = []
    for i in range(len(path) - 1):
        edge = G.get_edge_data(path[i], path[i+1], {})
        dev_metrics = _state.metrics.get(path[i], {})
        # Pick the interface on this hop
        intf = edge.get("intf_u") if G.nodes[path[i]] else edge.get("intf_v")
        intf_metrics = dev_metrics.get(intf, {}) if intf else {}

        hops.append({
            "hop":     i + 1,
            "from":    path[i],
            "to":      path[i+1],
            "status":  "UP",
            "latency_ms":        intf_metrics.get("latency_ms"),
            "packet_loss_rate":  intf_metrics.get("packet_loss_rate"),
            "bandwidth_mbps":    edge.get("bandwidth_mbps"),
        })

    return json.dumps({"reachable":   True,
                       "source":      source_device,
                       "destination": destination_device,
                       "hop_count":   len(path) - 1,
                       "path":        path,
                       "hops":        hops})


@tool
def check_routes(device_id: str, destination: Optional[str] = None) -> str:
    """
    Verify routing table for a device. Returns all reachable destinations
    and their next-hops via shortest-path routing. If destination is provided,
    returns just that specific route. Useful for detecting routing loops or
    missing routes caused by link failures.
    """
    if not _state.topology:
        return json.dumps({"error": "Topology not initialized."})

    import networkx as nx

    nodes = _state.topology.get("nodes", [])
    links = _state.topology.get("links", [])

    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n["id"])
    for l in links:
        if l.get("status", "UP") == "UP":
            w = l.get("weight", 1)
            G.add_edge(l["source"], l["target"], weight=w)
            G.add_edge(l["target"], l["source"], weight=w)

    if device_id not in G:
        return json.dumps({"error": f"Device '{device_id}' not found."})

    if destination:
        try:
            path = nx.shortest_path(G, device_id, destination, weight="weight")
            return json.dumps({"device": device_id, "destination": destination,
                               "next_hop": path[1] if len(path) > 1 else "local",
                               "full_path": path, "reachable": True})
        except nx.NetworkXNoPath:
            return json.dumps({"device": device_id, "destination": destination,
                               "reachable": False,
                               "diagnosis": "Route not found — possible link failure"})

    # Full routing table
    try:
        lengths, paths = nx.single_source_dijkstra(G, device_id, weight="weight")
        routing_table = {
            dest: {"next_hop": paths[dest][1] if len(paths[dest]) > 1 else "local",
                   "hops": len(paths[dest]) - 1,
                   "cost": lengths[dest]}
            for dest in paths if dest != device_id
        }
        unreachable = [n["id"] for n in nodes if n["id"] not in routing_table
                       and n["id"] != device_id]
        return json.dumps({"device": device_id,
                           "routing_table": routing_table,
                           "unreachable_devices": unreachable})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_anomaly_score(device_id: str, interface_id: Optional[str] = None) -> str:
    """
    Get the LSTM-Autoencoder MSE anomaly score for a device or specific interface.
    A higher score means the traffic pattern deviates more from normal.
    Threshold is ~0.000428 (simulation) or ~0.15 (Mininet-adapted).
    """
    if not _state.scores:
        return json.dumps({"error": "No anomaly scores available.",
                           "hint": "Scores are populated from the LSTM predict() output."})

    device_scores = _state.scores.get(device_id)
    if not device_scores:
        return json.dumps({"error": f"No scores for '{device_id}'.",
                           "available": list(_state.scores.keys())})

    if interface_id:
        score = device_scores.get(interface_id)
        if score is None:
            return json.dumps({"error": f"No score for '{device_id}/{interface_id}'.",
                               "available_interfaces": list(device_scores.keys())})
        return json.dumps({"device_id": device_id, "interface_id": interface_id,
                           "mse_score": score,
                           "anomalous": score > 0.000428})

    # Return all interface scores for device sorted by severity
    sorted_scores = sorted(device_scores.items(), key=lambda x: x[1], reverse=True)
    return json.dumps({"device_id": device_id,
                       "interface_scores": [{"interface": k, "mse": v,
                                             "anomalous": v > 0.000428}
                                            for k, v in sorted_scores]})


# ─── Tool List (import this in main.py) ───────────────────────────────────────

ALL_TOOLS = [
    get_metrics,
    inspect_topology,
    query_syslogs,
    trace_path,
    check_routes,
    get_anomaly_score,
]
