import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

class AnomalyInjector:
    """
    A high-level class that creates anomaly configurations for the TelemetryEngine.
    """

    def __init__(self, topology: Any):
        """
        Initialize the AnomalyInjector with a NetworkTopology instance.
        
        Args:
            topology: The NetworkTopology instance.
        """
        self.topology = topology

    def _validate_link(self, link: Tuple[str, str]) -> None:
        """
        Validates that the link exists in the topology.
        
        Args:
            link: A tuple of two device names (node1, node2).
            
        Raises:
            ValueError: If the link is not found in the topology.
        """
        node1, node2 = link
        # Attempt to validate based on common topology interfaces
        if hasattr(self.topology, 'has_link'):
            if not self.topology.has_link(node1, node2):
                raise ValueError(f"Link {link} does not exist in the topology.")
        elif hasattr(self.topology, 'links'):
            if link not in self.topology.links and (node2, node1) not in self.topology.links:
                raise ValueError(f"Link {link} does not exist in the topology.")
        elif hasattr(self.topology, 'graph'):
            if not self.topology.graph.has_edge(node1, node2):
                raise ValueError(f"Link {link} does not exist in the topology.")
        # If we cannot validate, we proceed assuming it's valid.

    def _create_base_anomaly(self, anomaly_type: str, link: Tuple[str, str], start_time: datetime, duration_minutes: float, severity: str, description: str) -> Dict[str, Any]:
        """Helper to create a base anomaly dict."""
        self._validate_link(link)
        end_time = start_time + timedelta(minutes=duration_minutes)
        return {
            "type": anomaly_type,
            "target_link": link,
            "start_time": start_time,
            "end_time": end_time,
            "anomaly_id": str(uuid.uuid4()),
            "severity": severity,
            "description": description
        }

    def create_link_failure(self, link: Tuple[str, str], start_time: datetime, duration_minutes: float) -> Dict[str, Any]:
        """Creates a link failure anomaly."""
        return self._create_base_anomaly(
            anomaly_type="link_failure",
            link=link,
            start_time=start_time,
            duration_minutes=duration_minutes,
            severity="critical",
            description=f"Complete link failure between {link[0]} and {link[1]}."
        )

    def create_congestion(self, link: Tuple[str, str], start_time: datetime, duration_minutes: float) -> Dict[str, Any]:
        """Creates a congestion anomaly."""
        return self._create_base_anomaly(
            anomaly_type="congestion",
            link=link,
            start_time=start_time,
            duration_minutes=duration_minutes,
            severity="major",
            description=f"High congestion detected on link between {link[0]} and {link[1]}."
        )

    def create_interface_flap(self, link: Tuple[str, str], start_time: datetime, duration_minutes: float) -> Dict[str, Any]:
        """Creates an interface flap anomaly."""
        return self._create_base_anomaly(
            anomaly_type="interface_flap",
            link=link,
            start_time=start_time,
            duration_minutes=duration_minutes,
            severity="major",
            description=f"Interface flapping observed on link between {link[0]} and {link[1]}."
        )

    def create_mtu_mismatch(self, link: Tuple[str, str], start_time: datetime, duration_minutes: float) -> Dict[str, Any]:
        """Creates an MTU mismatch anomaly."""
        return self._create_base_anomaly(
            anomaly_type="mtu_mismatch",
            link=link,
            start_time=start_time,
            duration_minutes=duration_minutes,
            severity="minor",
            description=f"MTU mismatch causing packet drops on link between {link[0]} and {link[1]}."
        )

    def create_routing_loop(self, link: Tuple[str, str], start_time: datetime, duration_minutes: float) -> Dict[str, Any]:
        """Creates a routing loop anomaly."""
        return self._create_base_anomaly(
            anomaly_type="routing_loop",
            link=link,
            start_time=start_time,
            duration_minutes=duration_minutes,
            severity="critical",
            description=f"Routing loop detected involving link between {link[0]} and {link[1]}."
        )

    def create_bgp_hijack(self, link: Tuple[str, str], start_time: datetime, duration_minutes: float) -> Dict[str, Any]:
        """Creates a BGP hijack anomaly."""
        return self._create_base_anomaly(
            anomaly_type="bgp_hijack",
            link=link,
            start_time=start_time,
            duration_minutes=duration_minutes,
            severity="critical",
            description=f"BGP route hijack detected redirecting traffic over link between {link[0]} and {link[1]}."
        )

    def create_cascading_failure(self, primary_link: Tuple[str, str], secondary_link: Tuple[str, str], start_time: datetime, gap_minutes: float, duration_minutes: float) -> List[Dict[str, Any]]:
        """
        Creates a cascading failure: primary link fails, then secondary gets congested due to rerouting.
        """
        primary = self.create_link_failure(primary_link, start_time, duration_minutes)
        secondary_start = start_time + timedelta(minutes=gap_minutes)
        secondary_duration = duration_minutes - gap_minutes
        if secondary_duration < 0:
            secondary_duration = 1.0  # default to 1 min if gap is larger than duration
        secondary = self.create_congestion(secondary_link, secondary_start, secondary_duration)
        
        # Modify description to indicate cascade
        secondary["description"] = f"High congestion on link between {secondary_link[0]} and {secondary_link[1]} due to rerouting from primary failure."
        
        return [primary, secondary]

    def get_ground_truth(self, anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Returns a ground truth catalog.
        
        Args:
            anomalies: List of anomaly configuration dicts.
            
        Returns:
            A ground truth dict containing anomaly_id, type, affected entities, time window, expected cascading effects.
        """
        ground_truth = {}
        for anomaly in anomalies:
            anomaly_id = anomaly["anomaly_id"]
            ground_truth[anomaly_id] = {
                "anomaly_id": anomaly_id,
                "type": anomaly["type"],
                "affected_entities": list(anomaly["target_link"]),
                "time_window": {
                    "start": anomaly["start_time"].isoformat(),
                    "end": anomaly["end_time"].isoformat()
                },
                "expected_cascading_effects": []
            }
            if anomaly["type"] == "link_failure":
                ground_truth[anomaly_id]["expected_cascading_effects"].append("Potential congestion on alternative paths")
            elif anomaly["type"] == "routing_loop":
                ground_truth[anomaly_id]["expected_cascading_effects"].append("High CPU utilization, latency spikes")
                
        return {
            "anomalies": ground_truth,
            "total_anomalies": len(anomalies)
        }
