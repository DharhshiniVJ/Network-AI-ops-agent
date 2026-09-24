import os
import json
import logging
from typing import Dict, List, Any, Optional
import networkx as nx
import pandas as pd

from network.topology import NetworkTopology
from network.telemetry_generator import TelemetryEngine
from network.anomaly_injector import AnomalyInjector
from network.scenarios import FailureScenario, build_all_scenarios
from detection.isolation_forest import IsolationForestDetector
from detection.baseline_models import BaselineDetector
from detection.evaluator import DetectionEvaluator
from agent.react_agent import ReActAgent
from agent.llm_provider import LLMProviderFactory
from agent.prompts import RCA_SYSTEM_PROMPT, ALERT_TEMPLATE
from agent.tools.registry import TOOL_REGISTRY, TOOLS_SCHEMA
from agent.rca_evaluator import RCAEvaluator

import agent.tools.get_metrics as get_metrics
import agent.tools.inspect_topology as inspect_topology
import agent.tools.check_routes as check_routes
import agent.tools.trace_path as trace_path
import agent.tools.query_logs as query_logs

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    def __init__(self, llm_provider: str = "gemini", verbose: bool = True):
        self.llm_provider = llm_provider
        self.verbose = verbose

    def setup_network_state(self, scenario: FailureScenario, topology: NetworkTopology, telemetry_engine: TelemetryEngine) -> Dict:
        # 1. Build the appropriate topology
        topology.build_spine_leaf() if scenario.topology_type == "spine_leaf" else topology.build_hierarchical_wan()
        
        # 2. Anomaly configs are passed directly to generate_telemetry_window below
            
        from datetime import datetime, timezone
        telemetry_data_tuple = telemetry_engine.generate_telemetry_window(
            start_time=datetime.now(timezone.utc),
            duration_seconds=600,
            interval_seconds=15,
            anomalies=scenario.anomaly_configs
        )
        df = telemetry_data_tuple[0]
        syslogs = telemetry_data_tuple[1]
        
        # Create a mock dataframe for syslogs just to satisfy the next lines
        logs_df = pd.DataFrame({'device_id': [log.split(' ')[2] for log in syslogs if len(log.split(' ')) > 2], 'message': syslogs}) if syslogs else pd.DataFrame(columns=['device_id', 'message'])
        
        # 3. Construct a network_state dict
        metrics = {}
        if not df.empty:
            for device in df['device_id'].unique():
                device_df = df[df['device_id'] == device]
                latest = device_df.iloc[-1]
                metrics[device] = {
                    "cpu_utilization": latest.get("cpu_util", 0),
                    "memory_utilization": latest.get("memory_util", 0),
                    "packet_loss": device_df["packet_loss"].max() if "packet_loss" in device_df.columns else 0,
                    "discards": device_df["discards"].sum() if "discards" in device_df.columns else 0,
                    "errors": device_df["errors"].sum() if "errors" in device_df.columns else 0,
                    "latency": device_df["latency"].max() if "latency" in device_df.columns else 0
                }
                
        routes = {}
        for node in topology.graph.nodes():
            routes[node] = nx.single_source_shortest_path(topology.graph, node)
            
        logs = {}
        if not logs_df.empty:
            for device in logs_df['device_id'].unique():
                logs[device] = logs_df[logs_df['device_id'] == device]['message'].tolist()
                
        network_state = {
            "metrics": metrics,
            "topology": topology.to_dict(),
            "routes": routes,
            "logs": logs,
            "topology_obj": topology
        }
        
        # 4. Inject this state into all tool modules
        from agent.tools import inject_network_state_to_tools
        inject_network_state_to_tools(network_state)
        
        # 5. Return the network_state
        return network_state

    def run_single_scenario(self, scenario: FailureScenario) -> Dict:
        topology = NetworkTopology()
        if scenario.topology_type == "spine_leaf":
            topology.build_spine_leaf()
        else:
            topology.build_hierarchical_wan()
            
        telemetry_engine = TelemetryEngine(topology)
        self.setup_network_state(scenario, topology, telemetry_engine)
        
        client, model_name = LLMProviderFactory.get_client(self.llm_provider)
        agent = ReActAgent(
            llm_client=client,
            model=model_name,
            system_prompt=RCA_SYSTEM_PROMPT,
            tools_registry=TOOL_REGISTRY,
            tools_schema=TOOLS_SCHEMA,
            verbose=self.verbose
        )
        
        alert_msg = ALERT_TEMPLATE.format(alert_message=scenario.alert_message)
        agent_result = agent.run(alert_msg)
        
        evaluator = RCAEvaluator()
        eval_result = evaluator.evaluate(scenario, agent_result)
        
        return {
            "scenario": scenario.id,
            "agent_result": agent_result,
            "evaluation": eval_result
        }

    def run_all_scenarios(self, scenarios: List[FailureScenario] = None) -> Dict:
        if scenarios is None:
            scenarios = build_all_scenarios()
            
        results = []
        for scenario in scenarios:
            if self.verbose:
                print(f"Running scenario: {scenario.id} - {scenario.name}")
            result = self.run_single_scenario(scenario)
            results.append(result)
            
        evaluator = RCAEvaluator()
        aggregate_evals = [r["evaluation"] for r in results]
        aggregate_results = evaluator.evaluate_batch(aggregate_evals)
        
        print("\n--- Benchmark Summary ---")
        print(f"Total Scenarios: {aggregate_results['total_scenarios']}")
        print(f"Average Accuracy: {aggregate_results['average_accuracy']:.2f}")
        print(f"Average Steps: {aggregate_results['average_steps']:.2f}")
        
        return {
            "individual_results": results,
            "aggregate_results": aggregate_results
        }

    def run_detection_benchmark(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run anomaly detection models on generated telemetry using Train/Val/Test split.
        """
        evaluator = DetectionEvaluator()
        results = evaluator.run_full_evaluation(train_df, val_df, test_df)
        
        comparison_df = evaluator.generate_comparison_table(results)
        
        return {
            "results": results,
            "comparison_df": comparison_df,
            "evaluator": evaluator
        }
