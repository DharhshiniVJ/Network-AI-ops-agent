#!/usr/bin/env python3
"""
Push-button Fault Injector
===========================
Injects named faults into a running Mininet network, or simulates the
injection event in the shared state.json for replay/demo mode on Mac.

Usage (Ubuntu with live Mininet):
    sudo python3 demo/inject_fault.py link_failure
    sudo python3 demo/inject_fault.py congestion
    sudo python3 demo/inject_fault.py interface_flap
    sudo python3 demo/inject_fault.py mtu_mismatch
    sudo python3 demo/inject_fault.py packet_loss
    sudo python3 demo/inject_fault.py recover        # clears all faults

Usage (Mac demo mode — just marks state.json):
    python3 demo/inject_fault.py link_failure --sim
"""

import sys, os, json, time, subprocess, argparse
from pathlib import Path
from datetime import datetime

ROOT       = Path(__file__).parent.parent
STATE_FILE = ROOT / "demo" / "state.json"

FAULT_DESCRIPTIONS = {
    "link_failure":    "Link between Spine1 and Leaf1 set to DOWN (ip link set s1-eth1 down)",
    "congestion":      "Traffic generator flooded Spine2-Leaf2 link to 95% capacity",
    "interface_flap":  "Spine1-Leaf2 link toggling UP/DOWN every 3 seconds",
    "mtu_mismatch":    "MTU on s1-eth5 set to 512 bytes (causes fragmentation errors)",
    "packet_loss":     "tc netem applied 15% random packet loss on s2-eth4",
    "recover":         "All faults cleared — network restored to baseline",
}

FAULT_COLORS = {
    "link_failure":   "#ef4444",
    "congestion":     "#f97316",
    "interface_flap": "#8b5cf6",
    "mtu_mismatch":   "#06b6d4",
    "packet_loss":    "#eab308",
    "recover":        "#10b981",
}

def update_state(fault: str):
    """Write the injected fault into the shared demo state."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
    else:
        state = {}

    if fault == "recover":
        state["injected_fault"] = "recover"
        state["rca_report"]     = None
        state["agent_steps"]    = []
        state["rf_verdict"]     = None
        state["lstm_verdict"]   = None
    else:
        state["injected_fault"] = fault

    state["last_update"] = str(datetime.now())

    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, default=str)
    tmp.replace(STATE_FILE)

    print(f"[inject] State updated: {fault}")


def inject_mininet(fault: str):
    """Run the actual Linux commands against a live Mininet network."""
    commands = {
        "link_failure": [
            "ip link set s1-eth1 down",
        ],
        "congestion": [
            # Flood s2-eth2 with iperf-style traffic using tc
            "tc qdisc add dev s2-eth2 root tbf rate 100kbit burst 10kb latency 50ms",
        ],
        "interface_flap": [
            # Background flap loop
            "bash -c 'for i in 1 2 3 4 5; do ip link set s1-eth2 down; sleep 1.5; ip link set s1-eth2 up; sleep 1.5; done' &",
        ],
        "mtu_mismatch": [
            "ip link set dev s1-eth5 mtu 512",
        ],
        "packet_loss": [
            "tc qdisc add dev s2-eth4 root netem loss 15%",
        ],
        "recover": [
            "ip link set s1-eth1 up",
            "ip link set s1-eth2 up",
            "tc qdisc del dev s2-eth2 root 2>/dev/null || true",
            "tc qdisc del dev s2-eth4 root 2>/dev/null || true",
            "ip link set dev s1-eth5 mtu 1500",
        ],
    }

    cmds = commands.get(fault, [])
    if not cmds:
        print(f"[inject] Unknown fault: {fault}")
        return

    for cmd in cmds:
        print(f"[inject] $ {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0 and "No such file" not in result.stderr:
            print(f"[inject] WARNING: {result.stderr.strip()}")
        time.sleep(0.2)

    print(f"[inject] Fault '{fault}' injected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIOps Fault Injector")
    parser.add_argument(
        "fault",
        choices=list(FAULT_DESCRIPTIONS.keys()),
        help="Fault type to inject"
    )
    parser.add_argument(
        "--sim", action="store_true",
        help="Simulation mode — only updates state.json, no Linux commands"
    )
    args = parser.parse_args()

    if args.sim:
        print(f"[inject] Simulation mode — marking '{args.fault}' in state.json")
        update_state(args.fault)
    else:
        inject_mininet(args.fault)
        update_state(args.fault)

    print(f"[inject] Done: {FAULT_DESCRIPTIONS[args.fault]}")
