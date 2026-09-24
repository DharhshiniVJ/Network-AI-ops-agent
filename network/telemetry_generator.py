"""
Telemetry Engine for Network Anomaly Detection Framework.
Generates synchronized multi-modal synthetic network telemetry data.
"""

import math
import datetime
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import numpy as np


class TelemetryEngine:
    """
    Generates synchronized multi-modal synthetic network telemetry data.
    Takes a NetworkTopology instance and generates SNMP metrics, Syslog messages,
    and NetFlow/IPFIX records with realistic baseline modeling and anomaly signatures.
    """

    def __init__(self, topology: Any, seed: int = 42):
        """
        Initialize the TelemetryEngine.

        Args:
            topology: NetworkTopology instance containing the network graph and details.
            seed: Random seed for reproducibility.
        """
        self.topology = topology
        self.rng = np.random.default_rng(seed)
        
        # State variables for Ornstein-Uhlenbeck noise
        self.ou_states: Dict[str, float] = {}
        self.theta = 0.15  # Reversion rate
        self.sigma = 0.08  # Volatility
        
        # Zipf-distributed destination ports
        self.zipf_ports = [443, 80, 8080, 53, 22, 9200]
        self.zipf_probs = self._calculate_zipf()

    def _calculate_zipf(self) -> List[float]:
        """Calculates Zipf probabilities for the destination ports."""
        s = 1.5  # Zipf parameter
        N = len(self.zipf_ports)
        probs = [1.0 / (i**s) for i in range(1, N + 1)]
        sum_probs = sum(probs)
        return [p / sum_probs for p in probs]

    def _get_diurnal_factor(self, dt: datetime.datetime) -> float:
        """
        Calculates diurnal pattern based on time of day.
        baseline(t) = 0.65 + 0.35 * sin(2π(t - 32400) / 86400)
        Peak at 14:00, trough at 04:00.
        """
        t = dt.hour * 3600 + dt.minute * 60 + dt.second
        return 0.65 + 0.35 * math.sin(2 * math.pi * (t - 32400) / 86400)

    def _ou_step(self, key: str, dt_seconds: float) -> float:
        """
        Calculates the next step for the Ornstein-Uhlenbeck process.
        x_new = x_old * exp(-θΔt) + μ(1-exp(-θΔt)) + σ*sqrt((1-exp(-2θΔt))/(2θ)) * Z
        """
        if key not in self.ou_states:
            self.ou_states[key] = 0.0
            
        x_old = self.ou_states[key]
        Z = self.rng.standard_normal()
        
        decay = math.exp(-self.theta * dt_seconds)
        vol = self.sigma * math.sqrt((1 - math.exp(-2 * self.theta * dt_seconds)) / (2 * self.theta))
        
        x_new = x_old * decay + vol * Z
        self.ou_states[key] = x_new
        return x_new

    def _extract_links(self) -> List[Tuple[str, str, int, str, str]]:
        """Extracts link information from the topology.
        
        Returns:
            List of (source, destination, capacity_bps, intf_src, intf_dst) tuples for each
            direction of every link in the topology.
        """
        links = []
        # Extract from NetworkX graph (attribute is .graph in our topology engine)
        graph = getattr(self.topology, 'graph', getattr(self.topology, 'G', None))
        if graph is not None and hasattr(graph, 'edges'):
            for u, v, data in graph.edges(data=True):
                # Support both bandwidth_mbps and capacity attributes
                if 'bandwidth_mbps' in data:
                    capacity_bps = int(data['bandwidth_mbps'] * 1_000_000)
                else:
                    capacity_bps = data.get('capacity', 10_000_000_000)
                    
                intf_u = data.get('intf_u', 'GigabitEthernet0/1')
                intf_v = data.get('intf_v', 'GigabitEthernet0/1')
                
                links.append((u, v, capacity_bps, intf_u, intf_v))
                links.append((v, u, capacity_bps, intf_v, intf_u))  # bidirectional
        else:
            # Fallback mock links
            links = [
                ("Spine1", "Leaf1", 40_000_000_000, "Ethernet1/1", "Ethernet1/1"),
                ("Leaf1", "Spine1", 40_000_000_000, "Ethernet1/1", "Ethernet1/1"),
                ("Spine1", "Leaf2", 40_000_000_000, "Ethernet1/2", "Ethernet1/1"),
                ("Leaf2", "Spine1", 40_000_000_000, "Ethernet1/1", "Ethernet1/2"),
            ]
        return links

    def generate_telemetry_window(
        self,
        start_time: datetime.datetime,
        duration_seconds: int = 3600,
        interval_seconds: int = 15,
        anomalies: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
        """
        Generates multi-modal telemetry data for a specific time window.
        
        Args:
            start_time: Start time of the window.
            duration_seconds: Total duration to simulate.
            interval_seconds: Granularity of SNMP metrics and simulation steps.
            anomalies: List of anomaly definitions to inject.
            
        Returns:
            Tuple containing (SNMP DataFrame, Syslog List, NetFlow DataFrame).
        """
        if anomalies is None:
            anomalies = []

        timestamps = [
            start_time + datetime.timedelta(seconds=i) 
            for i in range(0, duration_seconds, interval_seconds)
        ]

        snmp_records = []
        syslog_messages = []
        netflow_records = []

        links = self._extract_links()
        devices = list(set([u for u, _, _, _, _ in links]))

        def add_syslog(ts: datetime.datetime, device: str, process: str, msg: str, priority: int = 6):
            """Formats and appends RFC 5424 syslog message."""
            ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            syslog_messages.append(f"<{priority}>1 {ts_str} {device} {process} - - - {msg}")

        # Initial normal state syslog
        for d in devices:
            add_syslog(start_time, d, "SYS", "%SYS-6-NORMAL: System running normally, all interfaces operational", 6)

        for ts in timestamps:
            dt_seconds = float(interval_seconds)
            diurnal = self._get_diurnal_factor(ts)

            # Periodic background syslogs
            if ts.minute % 30 == 0 and ts.second < interval_seconds:
                for d in devices:
                    add_syslog(ts, d, "OSPF", "%OSPF-5-HELLO: Routine OSPF Hello", 6)

            for src, dst, capacity, intf_src, intf_dst in links:
                key = f"{src}->{dst}"
                
                # Baseline traffic modeling
                base_utilization = diurnal * 0.45  # scale down to leave room for spikes
                noise = self._ou_step(key, dt_seconds)
                current_utilization = max(0.01, min(0.99, base_utilization + noise))

                # Default healthy metrics
                ifOperStatus = 1
                is_anomaly = 0
                anomaly_type = "none"
                ifInOctets = int(current_utilization * capacity * interval_seconds / 8)
                ifOutOctets = int(current_utilization * capacity * interval_seconds / 8)
                ifInDiscards = 0
                ifOutDiscards = 0
                ifInErrors = 0
                
                # CPU utilization based on traffic and own noise
                cpu_noise = self._ou_step(f"{src}_cpu", dt_seconds)
                cpu_utilization = max(5.0, min(95.0, 15.0 + current_utilization * 40.0 + cpu_noise * 10.0))
                
                base_delay = 1.2  # ms
                # Queue mechanics for latency
                latency_ms = base_delay / (1 - min(current_utilization, 0.95))
                packet_loss_rate = 0.0

                # Check for active anomalies affecting this link/time
                for anom in anomalies:
                    if anom["start_time"] <= ts <= anom["end_time"]:
                        if "target_link" in anom and anom["target_link"] in [(src, dst), (dst, src)]:
                            is_anomaly = 1
                            anomaly_type = anom["type"]

                            if anomaly_type == "link_failure":
                                ifOperStatus = 2
                                ifInOctets = 0
                                ifOutOctets = 0
                                latency_ms = 0.0
                                
                                # Trigger syslog exactly once at start
                                if ts == anom["start_time"] or (ts - anom["start_time"]).total_seconds() < interval_seconds:
                                    # Ensure we don't spam if multiple links match or interval misalignment
                                    if not any(f"%LINK-3-UPDOWN" in m for m in syslog_messages[-10:]):
                                        add_syslog(ts, src, "LINK", f"%LINK-3-UPDOWN: Interface {intf_src}, changed state to down", 3)
                                        add_syslog(ts, src, "OSPF", f"%OSPF-5-ADJCHANGE: Process 1, Nbr on {intf_src} from FULL to DOWN", 5)

                            elif anomaly_type == "congestion":
                                current_utilization = 1.25
                                ifInOctets = int(capacity * interval_seconds / 8)
                                ifOutOctets = int(capacity * interval_seconds / 8)
                                ifOutDiscards = int((0.25 * capacity * interval_seconds) / (8 * 1500))
                                packet_loss_rate = 0.15
                                latency_ms = 80.0
                                
                                # Periodic syslog during congestion
                                if ts.second % 60 < interval_seconds:
                                    add_syslog(ts, src, "QOS", f"%QOS-4-BUFFER_OVERFLOW: Interface {intf_src} tail drops exceeded threshold", 4)

                            elif anomaly_type == "interface_flap":
                                elapsed = (ts - anom["start_time"]).total_seconds()
                                if (elapsed // 30) % 2 == 1:
                                    ifOperStatus = 2
                                    ifInOctets = 0
                                    ifOutOctets = 0
                                    latency_ms = 0.0
                                    if elapsed % 30 < interval_seconds:
                                        add_syslog(ts, src, "LINK", f"%LINK-3-UPDOWN: Interface {intf_src}, changed state to down", 3)
                                else:
                                    if elapsed % 30 < interval_seconds and elapsed >= 30:
                                        add_syslog(ts, src, "LINK", f"%LINK-3-UPDOWN: Interface {intf_src}, changed state to up", 3)

                            elif anomaly_type == "mtu_mismatch":
                                ifInErrors = int(10 + self.rng.poisson(30))
                                packet_loss_rate = 0.05
                                
                            elif anomaly_type == "routing_loop":
                                current_utilization = 0.99
                                cpu_utilization = 98.0
                                ifInOctets = int(0.99 * capacity * interval_seconds / 8)
                                ifOutOctets = ifInOctets
                                packet_loss_rate = 0.10
                                latency_ms = 120.0
                                
                                if ts.second % 30 < interval_seconds:
                                    add_syslog(ts, src, "ROUTING", f"%ROUTING-4-LOOP_DETECTED: Loop detected on {intf_src}", 4)
                                    add_syslog(ts, src, "IP", "%IP-4-TTL_EXPIRED: TTL expired in transit", 4)
                                    
                            elif anomaly_type == "bgp_hijack":
                                # BGP Hijack drops outbound octets (blackholing) but spikes latency massively
                                # as traffic is misrouted externally before dropping.
                                current_utilization = 0.80
                                ifInOctets = int(0.80 * capacity * interval_seconds / 8)
                                ifOutOctets = int(0.05 * capacity * interval_seconds / 8) # traffic blackholed
                                packet_loss_rate = 0.65
                                latency_ms = 350.0  # Massive delay via unauthorized ASN
                                
                                if ts.second % 60 < interval_seconds:
                                    add_syslog(ts, src, "BGP", f"%BGP-5-ADJCHANGE: neighbor 192.0.2.1 Down - Peer closed the session", 5)
                                    add_syslog(ts, src, "BGP", f"%BGP-3-NOTIFICATION: sent to neighbor 192.0.2.1 6/2 (Administrative Shutdown)", 3)

                # Record SNMP metrics
                snmp_records.append({
                    "timestamp": ts,
                    "device_id": src,
                    "interface_id": intf_src,
                    "peer_device_id": dst,
                    "ifOperStatus": ifOperStatus,
                    "ifInOctets": ifInOctets,
                    "ifOutOctets": ifOutOctets,
                    "ifInDiscards": ifInDiscards,
                    "ifOutDiscards": ifOutDiscards,
                    "ifInErrors": ifInErrors,
                    "latency_ms": latency_ms,
                    "packet_loss_rate": packet_loss_rate,
                    "cpu_utilization": cpu_utilization,
                    "is_anomaly": is_anomaly,
                    "anomaly_type": anomaly_type
                })

                # Generate NetFlow if link is UP (sample every 10th interval to keep dataset manageable)
                if ifOperStatus == 1 and self.rng.random() < 0.1:
                    num_flows = self.rng.poisson(10 + current_utilization * 50)
                    for _ in range(num_flows):
                        packet_count = max(1, int(self.rng.lognormal(mean=3.0, sigma=1.5)))
                        byte_count = packet_count * self.rng.integers(64, 1500)
                        
                        flow_dur = self.rng.uniform(0.1, max(0.2, interval_seconds - 0.1))
                        
                        netflow_records.append({
                            "flow_start": ts,
                            "flow_end": ts + datetime.timedelta(seconds=flow_dur),
                            "device_id": src,
                            "src_ip": f"10.0.{self.rng.integers(1, 254)}.{self.rng.integers(1, 254)}",
                            "dst_ip": f"10.0.{self.rng.integers(1, 254)}.{self.rng.integers(1, 254)}",
                            "src_port": self.rng.integers(1024, 65535),
                            "dst_port": self.rng.choice(self.zipf_ports, p=self.zipf_probs),
                            "protocol": self.rng.choice([6, 17], p=[0.8, 0.2]),
                            "packet_count": packet_count,
                            "byte_count": byte_count,
                            "tcp_flags": self.rng.choice([2, 18, 24, 16], p=[0.1, 0.5, 0.3, 0.1])
                        })

        df_snmp = pd.DataFrame(snmp_records)
        df_netflow = pd.DataFrame(netflow_records)

        return df_snmp, syslog_messages, df_netflow
