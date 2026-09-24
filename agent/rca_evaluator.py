"""
RCA Evaluator.
"""
from typing import Dict, Any, List, Tuple
import pandas as pd

class FailureScenario:
    """Mock class representing a failure scenario."""
    def __init__(
        self, 
        name: str, 
        ground_truth_entity: str, 
        ground_truth_fault_class: str,
        ground_truth_remediation_keywords: List[str],
        optimal_tool_sequence: List[str]
    ):
        self.name = name
        self.ground_truth_entity = ground_truth_entity
        self.ground_truth_fault_class = ground_truth_fault_class
        self.ground_truth_remediation_keywords = ground_truth_remediation_keywords
        self.optimal_tool_sequence = optimal_tool_sequence

class RCAEvaluator:
    """Evaluates agent RCA performance against ground truth."""

    def evaluate_single(self, agent_output: Dict[str, Any], scenario: Any) -> Dict[str, Any]:
        """
        Evaluate a single RCA result.
        
        Args:
            agent_output: The output dictionary from ReActAgent.run().
            scenario: The FailureScenario ground truth object.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        final_answer = agent_output.get("final_answer", "").lower()
        trajectory = agent_output.get("trajectory", [])
        
        entity_identified = scenario.ground_truth_entity.lower() in final_answer
        fault_class_identified = scenario.ground_truth_fault_class.lower() in final_answer
        rca_accuracy = entity_identified and fault_class_identified
        
        remediation_relevant = any(
            kw.lower() in final_answer 
            for kw in scenario.ground_truth_remediation_keywords
        )
        
        tools_used = [item.get("tool") for item in trajectory if "tool" in item]
        total_steps = agent_output.get("steps", 0)
        
        optimal_steps = len(scenario.optimal_tool_sequence)
        trajectory_efficiency = 1.0
        if total_steps > 0 and optimal_steps > 0:
            trajectory_efficiency = min(1.0, optimal_steps / total_steps)
            
        evidence_grounding_score = 0.0
        tools_with_observations = [t for t in trajectory if "observation" in t]
        if tools_with_observations:
            cited = 0
            for item in tools_with_observations:
                if item.get("tool", "").lower() in final_answer:
                    cited += 1
            evidence_grounding_score = cited / len(tools_with_observations)
            
        return {
            "scenario_name": scenario.name,
            "rca_accuracy": rca_accuracy,
            "entity_identified": entity_identified,
            "fault_class_identified": fault_class_identified,
            "remediation_relevant": remediation_relevant,
            "evidence_grounding_score": evidence_grounding_score,
            "trajectory_efficiency": trajectory_efficiency,
            "tools_used": tools_used,
            "total_steps": total_steps
        }

    def evaluate_batch(self, results: List[Tuple[Dict[str, Any], Any]]) -> Dict[str, Any]:
        """
        Evaluate a batch of RCA results.
        
        Args:
            results: List of (agent_output, scenario) tuples.
            
        Returns:
            Dictionary containing aggregate evaluation metrics.
        """
        if not results:
            return {}
            
        evaluations = [self.evaluate_single(out, scen) for out, scen in results]
        
        overall_rca_accuracy = sum(1 for e in evaluations if e["rca_accuracy"]) / len(evaluations)
        avg_evidence_grounding = sum(e["evidence_grounding_score"] for e in evaluations) / len(evaluations)
        avg_trajectory_efficiency = sum(e["trajectory_efficiency"] for e in evaluations) / len(evaluations)
        remediation_precision = sum(1 for e in evaluations if e["remediation_relevant"]) / len(evaluations)
        
        return {
            "overall_rca_accuracy": overall_rca_accuracy,
            "avg_evidence_grounding": avg_evidence_grounding,
            "avg_trajectory_efficiency": avg_trajectory_efficiency,
            "remediation_precision": remediation_precision,
            "per_scenario": evaluations
        }
        
    def print_evaluation_report(self, batch_results: Dict[str, Any]) -> None:
        """Prints a formatted evaluation report."""
        print("=== RCA Evaluation Report ===")
        print(f"Overall RCA Accuracy: {batch_results.get('overall_rca_accuracy', 0):.2%}")
        print(f"Average Evidence Grounding: {batch_results.get('avg_evidence_grounding', 0):.2f}")
        print(f"Average Trajectory Efficiency: {batch_results.get('avg_trajectory_efficiency', 0):.2f}")
        print(f"Remediation Precision: {batch_results.get('remediation_precision', 0):.2%}")
        print("="*27)
        
    def to_dataframe(self, batch_results: Dict[str, Any]) -> pd.DataFrame:
        """Converts batch results per-scenario list to a pandas DataFrame."""
        scenarios = batch_results.get("per_scenario", [])
        return pd.DataFrame(scenarios)
