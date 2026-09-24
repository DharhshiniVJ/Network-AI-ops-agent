"""
Network Topology Engine.

This module provides a NetworkX-based topology engine for the Agentic AI Network RCA project.
It supports building spine-leaf and hierarchical WAN topologies, shortest path routing,
and topology mutation.
"""

import json
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple, Set


class NetworkTopology:
    """Core NetworkX-based topology engine."""

    def __init__(self) -> None:
        """Initialize the network topology."""
        self.graph = nx.Graph()
        self._routing_cache: Dict[Tuple[str, str], List[str]] = {}

    def _calculate_ospf_cost(self, bandwidth_mbps: float) -> float:
        """Calculate OSPF cost based on bandwidth in Mbps."""
        if bandwidth_mbps <= 0:
            return float('inf')
        # Cost = 10^8 / bandwidth_bps = 100 / bandwidth_mbps
        return 100.0 / bandwidth_mbps

    def build_spine_leaf(self, num_spines: int = 2, num_leafs: int = 4) -> None:
        """
        Build a Datacenter Clos topology.
        Every leaf connects to every spine. 40 Gbps fabric links, 0.5ms delay.
        
        Args:
            num_spines: Number of spine switches.
            num_leafs: Number of leaf switches.
        """
        self.graph.clear()
        self.invalidate_routing_cache()

        # Add spines
        for i in range(1, num_spines + 1):
            node_id = f"Spine{i}"
            self.graph.add_node(
                node_id,
                device_type="spine",
                management_ip=f"10.0.1.{i}",
                cpu_baseline=10.0,
                memory_mb=16384,
                status="UP"
            )

        # Add leafs
        for i in range(1, num_leafs + 1):
            node_id = f"Leaf{i}"
            self.graph.add_node(
                node_id,
                device_type="leaf",
                management_ip=f"10.0.2.{i}",
                cpu_baseline=15.0,
                memory_mb=8192,
                status="UP"
            )

        # Add links
        for i in range(1, num_spines + 1):
            for j in range(1, num_leafs + 1):
                spine_id = f"Spine{i}"
                leaf_id = f"Leaf{j}"
                bw_mbps = 40000.0
                self.graph.add_edge(
                    spine_id,
                    leaf_id,
                    bandwidth_mbps=bw_mbps,
                    delay_ms=0.5,
                    loss_rate=0.0,
                    status="UP",
                    weight=self._calculate_ospf_cost(bw_mbps),
                    node_u=spine_id,
                    node_v=leaf_id,
                    intf_u=f"Ethernet1/{j}",
                    intf_v=f"Ethernet1/{i}",
                    queue_capacity_pkts=10000
                )

    def build_hierarchical_wan(self) -> None:
        """
        Build an Enterprise WAN topology.
        - 2 core routers (100 Gbps interconnect)
        - 3 distribution switches (10 Gbps uplinks to cores)
        - 4 access switches (1 Gbps uplinks to distribution)
        - Redundant cross-links between distribution switches
        """
        self.graph.clear()
        self.invalidate_routing_cache()

        # Core routers
        for i in range(1, 3):
            self.graph.add_node(
                f"R{i}",
                device_type="core_router",
                management_ip=f"10.1.1.{i}",
                cpu_baseline=20.0,
                memory_mb=32768,
                status="UP"
            )

        # Distribution switches
        for i in range(1, 4):
            self.graph.add_node(
                f"D{i}",
                device_type="dist_switch",
                management_ip=f"10.1.2.{i}",
                cpu_baseline=25.0,
                memory_mb=16384,
                status="UP"
            )

        # Access switches
        for i in range(1, 5):
            self.graph.add_node(
                f"A{i}",
                device_type="access_switch",
                management_ip=f"10.1.3.{i}",
                cpu_baseline=10.0,
                memory_mb=8192,
                status="UP"
            )

        def add_wan_edge(u: str, v: str, bw_mbps: float, delay: float, intf_u: str, intf_v: str) -> None:
            self.graph.add_edge(
                u, v,
                bandwidth_mbps=bw_mbps,
                delay_ms=delay,
                loss_rate=0.0,
                status="UP",
                weight=self._calculate_ospf_cost(bw_mbps),
                node_u=u,
                node_v=v,
                intf_u=intf_u,
                intf_v=intf_v,
                queue_capacity_pkts=5000
            )

        # Core to Core (100 Gbps)
        add_wan_edge("R1", "R2", 100000.0, 1.0, "HundredGigE0/1", "HundredGigE0/1")

        # Distribution to Core (10 Gbps)
        for i in range(1, 4):
            add_wan_edge(f"D{i}", "R1", 10000.0, 2.0, "TenGigE0/1", f"TenGigE0/{i+1}")
            add_wan_edge(f"D{i}", "R2", 10000.0, 2.0, "TenGigE0/2", f"TenGigE0/{i+1}")

        # Distribution Cross-links (10 Gbps)
        add_wan_edge("D1", "D2", 10000.0, 1.0, "TenGigE0/3", "TenGigE0/3")
        add_wan_edge("D2", "D3", 10000.0, 1.0, "TenGigE0/4", "TenGigE0/3")
        add_wan_edge("D3", "D1", 10000.0, 1.0, "TenGigE0/4", "TenGigE0/4")

        # Access to Distribution (1 Gbps)
        add_wan_edge("A1", "D1", 1000.0, 5.0, "GigE0/1", "GigE0/1")
        add_wan_edge("A2", "D1", 1000.0, 5.0, "GigE0/1", "GigE0/2")
        add_wan_edge("A3", "D2", 1000.0, 5.0, "GigE0/1", "GigE0/1")
        add_wan_edge("A4", "D3", 1000.0, 5.0, "GigE0/1", "GigE0/1")

    def _active_edges_view(self) -> nx.Graph:
        """Return a view of the graph with only UP nodes and edges."""
        def edge_filter(u: str, v: str) -> bool:
            return (self.graph.nodes[u].get('status') == 'UP' and 
                    self.graph.nodes[v].get('status') == 'UP' and
                    self.graph[u][v].get('status') == 'UP')
                    
        def node_filter(n: str) -> bool:
            return self.graph.nodes[n].get('status') == 'UP'

        return nx.subgraph_view(self.graph, filter_node=node_filter, filter_edge=edge_filter)

    def compute_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """
        Compute shortest path using Dijkstra over active (UP) edges only.
        Uses caching for performance.
        
        Args:
            source: Source node ID.
            target: Target node ID.
            
        Returns:
            List of node IDs representing the path, or None if no path exists.
        """
        if source not in self.graph or target not in self.graph:
            return None

        cache_key = (source, target)
        if cache_key in self._routing_cache:
            return self._routing_cache[cache_key]

        active_graph = self._active_edges_view()
        try:
            path = nx.shortest_path(active_graph, source=source, target=target, weight='weight')
            self._routing_cache[cache_key] = path
            return path
        except nx.NetworkXNoPath:
            return None

    def compute_all_paths(self, source: str, target: str) -> List[List[str]]:
        """
        Compute all simple paths between source and target for redundancy analysis.
        Uses active (UP) edges only.
        
        Args:
            source: Source node ID.
            target: Target node ID.
            
        Returns:
            List of lists of node IDs representing the paths.
        """
        if source not in self.graph or target not in self.graph:
            return []

        active_graph = self._active_edges_view()
        return list(nx.all_simple_paths(active_graph, source=source, target=target))

    def invalidate_routing_cache(self) -> None:
        """Clear the routing cache."""
        self._routing_cache.clear()

    def set_link_status(self, u: str, v: str, status: str) -> None:
        """
        Toggle link UP/DOWN and invalidate routing cache.
        
        Args:
            u: Source node ID.
            v: Target node ID.
            status: 'UP' or 'DOWN'.
        """
        if self.graph.has_edge(u, v):
            self.graph[u][v]['status'] = status
            self.invalidate_routing_cache()

    def set_node_status(self, node: str, status: str) -> None:
        """
        Toggle node UP/DOWN and invalidate routing cache.
        
        Args:
            node: Node ID.
            status: 'UP' or 'DOWN'.
        """
        if self.graph.has_node(node):
            self.graph.nodes[node]['status'] = status
            self.invalidate_routing_cache()

    def get_neighbors(self, device_id: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Get neighbors of a device and the connecting link data.
        
        Args:
            device_id: Node ID.
            
        Returns:
            List of tuples (neighbor_id, link_attributes_dict).
        """
        if device_id not in self.graph:
            return []
        
        neighbors = []
        for neighbor in self.graph.neighbors(device_id):
            edge_data = self.graph[device_id][neighbor]
            neighbors.append((neighbor, dict(edge_data)))
        return neighbors

    def get_link_data(self, u: str, v: str) -> Optional[Dict[str, Any]]:
        """
        Get edge attributes.
        
        Args:
            u: Source node ID.
            v: Target node ID.
            
        Returns:
            Dictionary of edge attributes or None if edge doesn't exist.
        """
        if self.graph.has_edge(u, v):
            return dict(self.graph[u][v])
        return None

    def get_node_data(self, device_id: str) -> Optional[Dict[str, Any]]:
        """
        Get node attributes.
        
        Args:
            device_id: Node ID.
            
        Returns:
            Dictionary of node attributes or None if node doesn't exist.
        """
        if self.graph.has_node(device_id):
            return dict(self.graph.nodes[device_id])
        return None

    def get_all_devices(self) -> List[str]:
        """
        List all node IDs.
        
        Returns:
            List of node IDs.
        """
        return list(self.graph.nodes)

    def get_all_links(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        List all edges with their data.
        
        Returns:
            List of tuples (u, v, link_attributes_dict).
        """
        return [(u, v, dict(data)) for u, v, data in self.graph.edges(data=True)]

    def get_device_interfaces(self, device_id: str) -> List[str]:
        """
        Get all interfaces on a device based on link configurations.
        
        Args:
            device_id: Node ID.
            
        Returns:
            List of interface names configured on the device.
        """
        interfaces = set()
        if device_id in self.graph:
            for u, v, data in self.graph.edges(device_id, data=True):
                if data.get('node_u') == device_id and 'intf_u' in data:
                    interfaces.add(data['intf_u'])
                elif data.get('node_v') == device_id and 'intf_v' in data:
                    interfaces.add(data['intf_v'])
                
        return list(interfaces)

    def get_betweenness_centrality(self) -> Dict[str, float]:
        """
        Calculate node betweenness centrality.
        
        Returns:
            Dictionary of node betweenness centrality scores.
        """
        active_graph = self._active_edges_view()
        return nx.betweenness_centrality(active_graph, weight='weight')

    def get_critical_links(self) -> List[Tuple[str, str]]:
        """
        Find bridges (single points of failure) in the active topology.
        
        Returns:
            List of edges (u, v) that are bridges.
        """
        active_graph = self._active_edges_view()
        return list(nx.bridges(active_graph))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert topology to a JSON-compatible dictionary (node-link format).
        
        Returns:
            Dictionary representation of the graph.
        """
        return nx.node_link_data(self.graph)

    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Reconstruct topology from a dictionary.
        
        Args:
            data: Dictionary representation of the graph (node-link format).
        """
        self.graph = nx.node_link_graph(data)
        self.invalidate_routing_cache()

    def export_graphml(self, filepath: str) -> None:
        """
        Export topology to a GraphML file.
        
        Args:
            filepath: Path to the output GraphML file.
        """
        nx.write_graphml(self.graph, filepath)

    def __repr__(self) -> str:
        """String representation of the topology."""
        return f"<NetworkTopology: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges>"
