#!/usr/bin/env python3
"""
Full 5-Fault Mininet Benchmark Dataset Generator
==================================================
Generates a production-quality ML training/testing dataset using real
Linux kernel network telemetry. This script must be run on Ubuntu/Linux
with Mininet installed.

Run:
    sudo python3 mininet_topo.py

Output:
    mininet_telemetry.csv  — ~50,000 records, all 5 fault types labeled
    mininet_summary.txt    — Dataset statistics

Total runtime: ~45 minutes
All 5 fault types: link_failure, congestion, interface_flap, mtu_mismatch, packet_loss
"""

import os
import sys
import time
import threading
import subprocess
import math
from datetime import datetime, timezone

import pandas as pd

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, Controller
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink

# ─── Topology ────────────────────────────────────────────────────────────────

class SpineLeafTopo(Topo):
    """2-Spine, 4-Leaf Datacenter Spine-Leaf Topology"""
    def build(self):
        spine1 = self.addSwitch('s1')
        spine2 = self.addSwitch('s2')

        leaf1 = self.addSwitch('l1')
        leaf2 = self.addSwitch('l2')
        leaf3 = self.addSwitch('l3')
        leaf4 = self.addSwitch('l4')

        h1 = self.addHost('h1', ip='10.0.0.1/24')
        h2 = self.addHost('h2', ip='10.0.0.2/24')
        h3 = self.addHost('h3', ip='10.0.0.3/24')
        h4 = self.addHost('h4', ip='10.0.0.4/24')

        # Hosts to Leafs (1 Gbps)
        self.addLink(h1, leaf1, bw=1000)
        self.addLink(h2, leaf2, bw=1000)
        self.addLink(h3, leaf3, bw=1000)
        self.addLink(h4, leaf4, bw=1000)

        # Full-mesh Leafs to both Spines (10 Gbps fabric)
        for leaf in [leaf1, leaf2, leaf3, leaf4]:
            self.addLink(spine1, leaf, bw=10000, delay='0.5ms')
            self.addLink(spine2, leaf, bw=10000, delay='0.5ms')


# ─── Helpers ─────────────────────────────────────────────────────────────────

def read_intf_stat(intf_name: str, stat: str) -> int:
    """Read a kernel interface statistic from sysfs."""
    try:
        path = f'/sys/class/net/{intf_name}/statistics/{stat}'
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return 0


def read_operstate(intf_name: str) -> int:
    """Return 1 (UP) or 2 (DOWN) for an interface."""
    try:
        state = open(f'/sys/class/net/{intf_name}/operstate').read().strip()
        return 1 if state in ('up', 'unknown') else 2
    except Exception:
        return 1


def read_cpu_utilization() -> float:
    """Read overall CPU utilization from /proc/stat (0.0 – 1.0)."""
    try:
        with open('/proc/stat') as f:
            line = f.readline()
        parts = list(map(int, line.split()[1:]))
        idle = parts[3]
        total = sum(parts)
        # Store previous reading between calls using a mutable default
        prev = read_cpu_utilization._prev
        if prev is None:
            read_cpu_utilization._prev = (idle, total)
            return 0.0
        prev_idle, prev_total = prev
        read_cpu_utilization._prev = (idle, total)
        d_idle  = idle  - prev_idle
        d_total = total - prev_total
        if d_total == 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - d_idle / d_total))
    except Exception:
        return 0.0

read_cpu_utilization._prev = None


def measure_latency(host, target_ip: str) -> float:
    """Run a single ping from host and return RTT in ms. Returns -1 on failure."""
    try:
        result = host.cmd(f'ping -c 1 -W 1 {target_ip}')
        for line in result.split('\n'):
            if 'time=' in line:
                return float(line.split('time=')[1].split(' ')[0])
    except Exception:
        pass
    return -1.0


def measure_packet_loss(host, target_ip: str, count: int = 5) -> float:
    """Run ping with count packets and return loss rate (0.0 – 1.0)."""
    try:
        result = host.cmd(f'ping -c {count} -W 1 {target_ip}')
        for line in result.split('\n'):
            if '% packet loss' in line:
                pct = float(line.split('%')[0].split()[-1])
                return pct / 100.0
    except Exception:
        pass
    return 0.0


# ─── Telemetry Collector ─────────────────────────────────────────────────────

class TelemetryCollector(threading.Thread):
    """
    Background thread that polls every POLL_INTERVAL seconds and records:
      - Per-interface kernel counters (ifInOctets, ifOutOctets, drops, errors)
      - Per-switch CPU utilization from /proc/stat
      - Current fault label set by the orchestrator
    """
    POLL_INTERVAL = 1   # 1 second — gives ~100k records over 45 min (was 2s)

    def __init__(self, net, probe_host, probe_target_ip: str):
        super().__init__(daemon=True)
        self.net              = net
        self.probe_host       = probe_host
        self.probe_target_ip  = probe_target_ip
        self.running          = True
        self.data             = []

        # Mutable fault state — orchestrator writes, collector reads
        self.current_anomaly_type = 'none'
        self.current_is_anomaly   = 0

        # Rolling latency/loss updated by a dedicated background probe thread
        self._latest_latency_ms   = 0.5
        self._latest_packet_loss  = 0.0
        self._probe_thread        = threading.Thread(target=self._probe_loop, daemon=True)
        self._probe_thread.start()

        # Live CSV streaming — flushed every FLUSH_INTERVAL seconds for demo pipeline
        self.live_csv      = 'mininet_telemetry_live.csv'
        self._flush_every  = 5     # seconds
        self._last_flush   = 0
        self._flushed_rows = 0
        self._csv_header_written = False

    def _probe_loop(self):
        """Runs continuous ping probes in background — non-blocking on collector."""
        while self.running:
            try:
                result = self.probe_host.cmd(
                    f'ping -c 5 -W 1 -q {self.probe_target_ip}')
                for line in result.split('\n'):
                    if 'rtt' in line and 'avg' in line:
                        avg_ms = float(line.split('/')[4])
                        self._latest_latency_ms = avg_ms
                    if '% packet loss' in line:
                        pct = float(line.split('%')[0].split()[-1])
                        self._latest_packet_loss = pct / 100.0
            except Exception:
                pass
            time.sleep(3)

    def _flush_live_csv(self):
        """Append any unflushed rows to the live CSV file."""
        new_rows = self.data[self._flushed_rows:]
        if not new_rows:
            return
        import csv, os
        write_header = not self._csv_header_written
        with open(self.live_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=new_rows[0].keys())
            if write_header:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerows(new_rows)
        self._flushed_rows = len(self.data)

    def run(self):
        while self.running:
            ts          = datetime.now(timezone.utc)
            cpu_util    = read_cpu_utilization()
            latency_ms  = self._latest_latency_ms
            packet_loss = self._latest_packet_loss

            for switch in self.net.switches:
                for intf in switch.intfList():
                    if intf.name == 'lo':
                        continue
                    n = intf.name
                    self.data.append({
                        'timestamp'        : ts,
                        'device_id'        : switch.name,
                        'interface_id'     : n,
                        'ifOperStatus'     : read_operstate(n),
                        'ifInOctets'       : read_intf_stat(n, 'rx_bytes'),
                        'ifOutOctets'      : read_intf_stat(n, 'tx_bytes'),
                        'ifInDiscards'     : read_intf_stat(n, 'rx_dropped'),
                        'ifOutDiscards'    : read_intf_stat(n, 'tx_dropped'),
                        'ifInErrors'       : read_intf_stat(n, 'rx_errors'),
                        'latency_ms'       : max(0.0, latency_ms),
                        'packet_loss_rate' : packet_loss,
                        'cpu_utilization'  : cpu_util,
                        'is_anomaly'       : self.current_is_anomaly,
                        'anomaly_type'     : self.current_anomaly_type,
                    })

            # Flush to live CSV every FLUSH_INTERVAL seconds
            now = time.time()
            if now - self._last_flush >= self._flush_every:
                self._flush_live_csv()
                self._last_flush = now

            time.sleep(self.POLL_INTERVAL)

    def set_fault(self, anomaly_type: str):
        self.current_anomaly_type = anomaly_type
        self.current_is_anomaly   = 0 if anomaly_type == 'none' else 1

    def save(self, path='mininet_telemetry.csv'):
        df = pd.DataFrame(self.data)
        # Derive additional columns the feature pipeline expects
        for col in ['ifInOctets', 'ifOutOctets', 'ifInDiscards']:
            df[f'delta_{col}'] = (
                df.groupby(['device_id', 'interface_id'])[col]
                  .diff()
                  .fillna(0)
            )
        df.to_csv(path, index=False)

        # Print summary
        total  = len(df)
        anoms  = df['is_anomaly'].sum()
        by_type = df.groupby('anomaly_type')['is_anomaly'].count()

        summary = (
            f"\n{'='*55}\n"
            f"  DATASET SUMMARY\n"
            f"{'='*55}\n"
            f"  Total records    : {total:,}\n"
            f"  Normal records   : {total - anoms:,}\n"
            f"  Anomaly records  : {anoms:,} ({anoms/total*100:.1f}%)\n\n"
            f"  Records per fault type:\n"
        )
        for atype, count in by_type.items():
            summary += f"    {atype:<20}: {count:,}\n"
        summary += f"{'='*55}\n"
        info(summary)

        with open('mininet_summary.txt', 'w') as f:
            f.write(summary)

        info(f"*** Saved {total:,} records to {path}\n")


# ─── Fault Injectors ─────────────────────────────────────────────────────────

def inject_link_failure(net, collector, switch_a='s1', switch_b='l1',
                        duration=300):
    """Phase 1: Physically bring down a spine-leaf link."""
    info(f"\n{'='*55}\n")
    info(f"  INJECTING FAULT: link_failure ({switch_a} ↔ {switch_b})\n")
    info(f"{'='*55}\n")
    collector.set_fault('link_failure')
    net.configLinkStatus(switch_a, switch_b, 'down')
    time.sleep(duration)
    net.configLinkStatus(switch_a, switch_b, 'up')
    info(f"*** Link {switch_a}↔{switch_b} restored\n")


def inject_congestion(net, collector, intf_name='s2-eth2',
                      delay_ms=80, duration=300):
    """Phase 2: Use Linux tc netem to add 80ms delay + jitter."""
    info(f"\n{'='*55}\n")
    info(f"  INJECTING FAULT: congestion (tc netem on {intf_name})\n")
    info(f"{'='*55}\n")
    collector.set_fault('congestion')
    os.system(f'tc qdisc add dev {intf_name} root netem delay {delay_ms}ms '
              f'{delay_ms//4}ms distribution normal loss 5%')
    time.sleep(duration)
    os.system(f'tc qdisc del dev {intf_name} root 2>/dev/null')
    info(f"*** Congestion on {intf_name} cleared\n")


def inject_interface_flap(net, collector, switch_a='s1', switch_b='l2',
                          flap_interval=5, duration=300):
    """Phase 3: Rapidly toggle a link UP/DOWN every flap_interval seconds."""
    info(f"\n{'='*55}\n")
    info(f"  INJECTING FAULT: interface_flap ({switch_a} ↔ {switch_b})\n")
    info(f"{'='*55}\n")
    collector.set_fault('interface_flap')
    end_time = time.time() + duration
    state = True   # True = up, False = down
    while time.time() < end_time:
        status = 'up' if state else 'down'
        net.configLinkStatus(switch_a, switch_b, status)
        state = not state
        time.sleep(flap_interval)
    net.configLinkStatus(switch_a, switch_b, 'up')
    info(f"*** Interface flap on {switch_a}↔{switch_b} stopped\n")


def inject_mtu_mismatch(net, collector, intf_name='s1-eth5',
                        mtu=500, duration=300):
    """Phase 4: Force MTU to 500 bytes causing fragmentation errors."""
    info(f"\n{'='*55}\n")
    info(f"  INJECTING FAULT: mtu_mismatch (MTU={mtu} on {intf_name})\n")
    info(f"{'='*55}\n")
    collector.set_fault('mtu_mismatch')
    os.system(f'ip link set dev {intf_name} mtu {mtu}')
    time.sleep(duration)
    os.system(f'ip link set dev {intf_name} mtu 1500')
    info(f"*** MTU on {intf_name} restored to 1500\n")


def inject_packet_loss(net, collector, intf_name='s2-eth4',
                       loss_pct=40, duration=300):
    """Phase 5: Force 40% random packet loss via tc netem."""
    info(f"\n{'='*55}\n")
    info(f"  INJECTING FAULT: packet_loss ({loss_pct}% on {intf_name})\n")
    info(f"{'='*55}\n")
    collector.set_fault('packet_loss')
    os.system(f'tc qdisc add dev {intf_name} root netem loss {loss_pct}%')
    time.sleep(duration)
    os.system(f'tc qdisc del dev {intf_name} root 2>/dev/null')
    info(f"*** Packet loss on {intf_name} cleared\n")


def recovery_phase(collector, duration=180, label='Recovery'):
    """Normal traffic window between faults."""
    info(f"\n*** {label} — {duration}s of normal traffic...\n")
    collector.set_fault('none')
    time.sleep(duration)


# ─── Main Emulation ──────────────────────────────────────────────────────────

def run_emulation():
    setLogLevel('info')

    info("*** Building Spine-Leaf Topology\n")
    topo = SpineLeafTopo()
    net  = Mininet(topo=topo, switch=OVSKernelSwitch, link=TCLink,
                   controller=Controller)
    net.start()

    info("*** Testing baseline connectivity\n")
    net.pingAll()

    h1, h2, h3, h4 = net.get('h1', 'h2', 'h3', 'h4')

    # Start iperf servers on all hosts
    for h in [h2, h3, h4]:
        h.cmd('iperf3 -s -D')
    time.sleep(1)

    info("*** Starting background traffic (h1→h2, h1→h3, h1→h4, h2→h3)\n")
    h1.cmd('iperf3 -c 10.0.0.2 -t 9999 -b 200M &')
    h1.cmd('iperf3 -c 10.0.0.3 -t 9999 -b 200M &')
    h1.cmd('iperf3 -c 10.0.0.4 -t 9999 -b 200M &')
    h2.cmd('iperf3 -c 10.0.0.3 -t 9999 -b 100M &')  # extra cross traffic
    time.sleep(3)

    # Probe pair for latency/loss measurement
    collector = TelemetryCollector(net, probe_host=h1, probe_target_ip='10.0.0.4')
    collector.start()

    # ── Phase 0: Warmup — Normal Baseline (15 min) ────────────────────────
    info("\n*** PHASE 0: Warmup — 15 minutes of clean normal traffic\n")
    recovery_phase(collector, duration=900, label='Warmup')

    # ── Phase 1: Link Failure (3 min) ─────────────────────────────────────
    inject_link_failure(net, collector, switch_a='s1', switch_b='l1', duration=180)
    recovery_phase(collector, duration=480, label='Post-LinkFailure Recovery')

    # ── Phase 2: Congestion (3 min) ───────────────────────────────────────
    inject_congestion(net, collector, intf_name='s2-eth2', delay_ms=80, duration=180)
    recovery_phase(collector, duration=480, label='Post-Congestion Recovery')

    # ── Phase 3: Interface Flap (3 min) ───────────────────────────────────
    inject_interface_flap(net, collector, switch_a='s1', switch_b='l2',
                          flap_interval=3, duration=180)
    recovery_phase(collector, duration=480, label='Post-Flap Recovery')

    # ── Phase 4: MTU Mismatch (3 min) ─────────────────────────────────────
    inject_mtu_mismatch(net, collector, intf_name='s1-eth5', mtu=500, duration=180)
    recovery_phase(collector, duration=480, label='Post-MTU Recovery')

    # ── Phase 5: Packet Loss (3 min) ──────────────────────────────────────
    inject_packet_loss(net, collector, intf_name='s2-eth4', loss_pct=40, duration=180)
    recovery_phase(collector, duration=480, label='Post-PacketLoss Recovery')

    # ── Done ──────────────────────────────────────────────────────────────
    info("\n*** Stopping telemetry collector and saving dataset...\n")
    collector.running = False
    collector.join()
    collector.save('mininet_telemetry.csv')

    info("\n*** Dropping into CLI — type 'exit' to shut down\n")
    CLI(net)
    net.stop()


if __name__ == '__main__':
    if os.geteuid() != 0:
        print("ERROR: This script must be run as root: sudo python3 mininet_topo.py")
        sys.exit(1)
    run_emulation()
