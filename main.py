import os
import sys
import json
import argparse
import logging
import pandas as pd
from datetime import datetime

from network.topology import NetworkTopology
from network.telemetry_generator import TelemetryEngine
from network.anomaly_injector import AnomalyInjector
from network.scenarios import build_all_scenarios
from detection.isolation_forest import IsolationForestDetector
from detection.baseline_models import BaselineDetector
from detection.evaluator import DetectionEvaluator
from benchmark.runner import BenchmarkRunner
from benchmark.report_generator import ReportGenerator

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def ensure_data_dir():
    os.makedirs("data", exist_ok=True)

def cmd_simulate(args):
    print("Running comprehensive 24-hour dataset simulation...")
    ensure_data_dir()
    from datetime import datetime, timezone, timedelta
    
    base_time = datetime(2026, 9, 18, 0, 0, 0, tzinfo=timezone.utc)
    all_snmp, all_syslogs, all_flows = [], [], []
    
    # Run both simulations concurrently in time (0-12h) so time-based splits get both topologies
    configs = [
        {"name": "spine_leaf", "offset_hours": 0},
        {"name": "wan", "offset_hours": 0}
    ]
    
    final_topology = None
    
    for cfg in configs:
        print(f"Generating 12-hour telemetry for {cfg['name']} topology...")
        topo = NetworkTopology()
        if cfg['name'] == "spine_leaf":
            topo.build_spine_leaf()
            links = [("Spine1", "Leaf1"), ("Spine2", "Leaf2"), ("Spine1", "Leaf3"), ("Spine2", "Leaf4"), ("Spine1", "Leaf2")]
        else:
            topo.build_hierarchical_wan()
            links = [("R1", "D1"), ("R2", "D2"), ("D1", "A1"), ("D3", "A4"), ("R1", "R2")]
            
        final_topology = topo  # Save last one for topology.json
        injector = AnomalyInjector(topo)
        phase_time = base_time + timedelta(hours=cfg['offset_hours'])
        
        # Inject 3 rounds of anomalies across 72 hours for proper density
        anomalies = [
            # Round 1 (Hours 2-12)
            injector.create_link_failure(links[0], phase_time + timedelta(hours=2), 15.0),
            injector.create_congestion(links[1], phase_time + timedelta(hours=4), 20.0),
            injector.create_interface_flap(links[2], phase_time + timedelta(hours=6), 10.0),
            injector.create_mtu_mismatch(links[3], phase_time + timedelta(hours=8), 15.0),
            injector.create_routing_loop(links[4], phase_time + timedelta(hours=10), 10.0),
            
            # Round 2 (Hours 24-36)
            injector.create_link_failure(links[2], phase_time + timedelta(hours=24), 15.0),
            injector.create_congestion(links[3], phase_time + timedelta(hours=27), 20.0),
            injector.create_interface_flap(links[4], phase_time + timedelta(hours=30), 10.0),
            injector.create_mtu_mismatch(links[0], phase_time + timedelta(hours=33), 15.0),
            injector.create_routing_loop(links[1], phase_time + timedelta(hours=35), 10.0),

            # Round 3 (Hours 48-60)
            injector.create_link_failure(links[4], phase_time + timedelta(hours=48), 15.0),
            injector.create_congestion(links[0], phase_time + timedelta(hours=51), 20.0),
            injector.create_interface_flap(links[1], phase_time + timedelta(hours=54), 10.0),
            injector.create_mtu_mismatch(links[2], phase_time + timedelta(hours=57), 15.0),
            injector.create_routing_loop(links[3], phase_time + timedelta(hours=59), 10.0),
        ]
        
        engine = TelemetryEngine(topo)
        snmp, syslogs, flow = engine.generate_telemetry_window(
            start_time=phase_time,
            duration_seconds=259200, # 72 hours
            interval_seconds=15,     # Back to 15s — this is what gave good results
            anomalies=anomalies
        )
        all_snmp.append(snmp)
        all_syslogs.extend(syslogs)
        all_flows.append(flow)
        
    print("Merging datasets...")
    final_snmp = pd.concat(all_snmp, ignore_index=True)
    final_flow = pd.concat(all_flows, ignore_index=True)
    
    # Save to data/
    snmp_path = "data/snmp_telemetry.parquet"
    final_snmp.to_parquet(snmp_path)
    
    syslog_path = "data/syslog.jsonl"
    with open(syslog_path, "w") as f:
        for log in all_syslogs:
            f.write(json.dumps({"log": log}) + "\n")
            
    flow_path = "data/flow_telemetry.parquet"
    final_flow.to_parquet(flow_path)
    
    topo_path = "data/topology.json"
    with open(topo_path, "w") as f:
        json.dump(final_topology.to_dict(), f, indent=4)
        
    print(f"Simulation complete. Data saved to data/ directory.")
    print(f"Total SNMP records: {len(final_snmp):,}")
    print(f"Total Syslog records: {len(all_syslogs):,}")
    print(f"Total Flow records: {len(final_flow):,}")

def cmd_detect(args):
    print("Running detection benchmark...")
    snmp_path = "data/snmp_telemetry.parquet"
    if not os.path.exists(snmp_path):
        print(f"Error: Data file {snmp_path} not found. Run 'simulate' first.")
        return
        
    df = pd.read_parquet(snmp_path)
    
    if 'is_anomaly' not in df.columns:
        df['is_anomaly'] = 0
        
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    
    # ----------------------------------------------------------------
    # FIX 1: Stratified Split (ensures all 5 anomaly types in every split)
    # ----------------------------------------------------------------
    # Separate normal and anomaly samples
    normal_df = df[df['is_anomaly'] == 0].copy()
    anomaly_df = df[df['is_anomaly'] == 1].copy()
    
    # Split normal samples chronologically (60/20/20)
    n_train = int(len(normal_df) * 0.6)
    n_val = int(len(normal_df) * 0.8)
    normal_train = normal_df.iloc[:n_train]
    normal_val = normal_df.iloc[n_train:n_val]
    normal_test = normal_df.iloc[n_val:]
    
    # Stratified split of anomaly samples by anomaly_type
    anomaly_train_parts, anomaly_val_parts, anomaly_test_parts = [], [], []
    for atype, group in anomaly_df.groupby('anomaly_type'):
        group = group.sort_values('timestamp')
        a_train_idx = int(len(group) * 0.6)
        a_val_idx = int(len(group) * 0.8)
        anomaly_train_parts.append(group.iloc[:a_train_idx])
        anomaly_val_parts.append(group.iloc[a_train_idx:a_val_idx])
        anomaly_test_parts.append(group.iloc[a_val_idx:])
    
    anomaly_train = pd.concat(anomaly_train_parts)
    anomaly_val = pd.concat(anomaly_val_parts)
    anomaly_test = pd.concat(anomaly_test_parts)
    
    # Combine and sort by timestamp
    train_df = pd.concat([normal_train, anomaly_train]).sort_values('timestamp').reset_index(drop=True)
    val_df = pd.concat([normal_val, anomaly_val]).sort_values('timestamp').reset_index(drop=True)
    test_df = pd.concat([normal_test, anomaly_test]).sort_values('timestamp').reset_index(drop=True)
    
    print(f"Data Split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Anomaly counts -> Train: {train_df['is_anomaly'].sum()}, Val: {val_df['is_anomaly'].sum()}, Test: {test_df['is_anomaly'].sum()}")
    
    if 'anomaly_type' in train_df.columns:
        for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            types = split_df[split_df['is_anomaly']==1]['anomaly_type'].value_counts()
            print(f"  {split_name} anomaly types: {dict(types)}")
    
    runner = BenchmarkRunner()
    results = runner.run_detection_benchmark(train_df, val_df, test_df)
    
    print("\n--- Detection Results ---")
    if 'comparison_df' in results:
        print(results['comparison_df'].to_string())
        
    ReportGenerator().generate_detection_report(results, "results/detection")

def cmd_investigate(args):
    """Run Tier 3 LangGraph ReAct Agent on live detection output."""
    from agent.run_agent import run_agent_pipeline
    max_alerts = getattr(args, "max_alerts", 3)
    run_agent_pipeline(max_alerts=max_alerts)


def cmd_benchmark(args):
    print("Running full agent benchmark...")
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is not set.")
        return
        
    runner = BenchmarkRunner(verbose=False)
    try:
        results = runner.run_all_scenarios()
        ReportGenerator().generate_agent_report(results, "results/agent")
    except Exception as e:
        print(f"Benchmark failed: {str(e)}")

def cmd_demo(args):
    scenarios = build_all_scenarios()
    print("Available Scenarios:")
    for s in scenarios:
        print(f"  {s.id}: {s.name}")
        
    choice = input("\nEnter scenario ID to investigate: ").strip()
    scenario = next((s for s in scenarios if s.id == choice), None)
    
    if not scenario:
        print("Invalid scenario ID.")
        return
        
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is not set.")
        return
        
    print(f"\nStarting investigation for {scenario.id}...")
    runner = BenchmarkRunner(verbose=True)
    try:
        runner.run_single_scenario(scenario)
    except Exception as e:
        print(f"Demo failed: {str(e)}")

def cmd_full(args):
    print("Running end-to-end full pipeline...")
    cmd_simulate(args)
    cmd_detect(args)
    cmd_benchmark(args)
    
    # We could theoretically combine the reports here, but cmd_detect and cmd_benchmark
    # already generate their respective pieces in the results directory.
    print("\nFull pipeline complete. Results saved in data/ and results/ directories.")

def cmd_triage(args):
    print("=== Running Tiered Triage Architecture ===")
    from benchmark.runner import BenchmarkRunner
    from detection.tiered_triage import TieredTriagePipeline
    from network.topology import NetworkTopology
    import pandas as pd
    
    sl_topo = NetworkTopology()
    sl_topo.build_spine_leaf()
    wan_topo = NetworkTopology()
    wan_topo.build_hierarchical_wan()
    base_time = pd.Timestamp("2026-09-18 00:00:00", tz="UTC")
    
    scenarios = build_all_scenarios(sl_topo, wan_topo, base_time)
    
    # We choose Scenario 3 (Zero-Day: Routing Loop)
    target_scenario = scenarios[2]
    print(f"Testing Scenario: {target_scenario.id} - {target_scenario.description}")
    
    llm_provider = getattr(args, 'llm_provider', 'gemini')
    runner = BenchmarkRunner(llm_provider=llm_provider, verbose=False)
    
    # Generate the network state and telemetry for this specific scenario
    from network.topology import NetworkTopology
    from network.telemetry_generator import TelemetryEngine
    
    topology = NetworkTopology()
    if target_scenario.topology_type == "spine_leaf":
        topology.build_spine_leaf()
    else:
        topology.build_hierarchical_wan()
        
    telemetry_engine = TelemetryEngine(topology)
    network_state = runner.setup_network_state(target_scenario, topology, telemetry_engine)
    
    # We need the raw telemetry dataframe to pass through the models
    telemetry_data = telemetry_engine.generate_telemetry_window(
        start_time=target_scenario.anomaly_configs[0]['start_time'] - pd.Timedelta(minutes=5),
        duration_seconds=600,
        interval_seconds=15,
        anomalies=target_scenario.anomaly_configs
    )
    df = telemetry_data[0] # snmp data
    
    print("\n--- Executing 3-Tier Pipeline ---")
    pipeline = TieredTriagePipeline(llm_provider=llm_provider)
    result = pipeline.process_telemetry_window(df, network_state)
    
    print("\n================ TIERED TRIAGE RESULTS ================")
    print(f"Tier 1 (Supervised Filter) : {result['tier_1_status']}")
    print(f"Tier 2 (Unsupervised)      : {result['tier_2_status']}")
    print(f"Tier 3 (Agentic AI)        : {result['tier_3_status']}")
    print("\nFINAL RCA REPORT:")
    print("-------------------------------------------------------")
    print(result['final_rca'])
    print("=======================================================\n")

def main():
    parser = argparse.ArgumentParser(description="Network RCA Capstone Project Entry Point")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    subparsers.add_parser("simulate", help="Generate topology + telemetry data")
    subparsers.add_parser("detect", help="Run anomaly detection (IF + baselines)")
    
    inv_parser = subparsers.add_parser("investigate", help="Run agent on a single scenario")
    inv_parser.add_argument("--scenario", required=True, help="Scenario ID (e.g., SCEN-01)")
    
    bench_parser = subparsers.add_parser("benchmark", help="Run full benchmark (all 10 scenarios)")
    bench_parser.add_argument("--llm_provider", type=str, default="gemini", choices=["gemini", "openai", "ollama"], help="LLM Provider to use")
    bench_parser.add_argument("--verbose", action="store_true", help="Enable verbose agent logging")

    triage_parser = subparsers.add_parser("triage", help="Run the advanced Supervised -> Unsupervised -> Agentic pipeline")
    triage_parser.add_argument("--llm_provider", type=str, default="gemini", choices=["gemini", "openai", "ollama"], help="LLM Provider to use")

    subparsers.add_parser("demo", help="Interactive demo: pick a scenario, watch agent investigate")
    subparsers.add_parser("full", help="Run everything end-to-end")
    
    args = parser.parse_args()
    
    commands = {
        "simulate": cmd_simulate,
        "detect": cmd_detect,
        "investigate": cmd_investigate,
        "benchmark": cmd_benchmark,
        "triage": cmd_triage,
        "demo": cmd_demo,
        "full": cmd_full
    }
    
    commands[args.command](args)

if __name__ == "__main__":
    main()
