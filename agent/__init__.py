"""
Agentic AI package for autonomous network root cause analysis.

Provides a LangGraph ReAct agent with diagnostic tools
for investigating network anomalies and generating evidence-grounded RCA reports.
"""
from agent.react_agent import build_rca_graph, run_rca_investigation

__all__ = ["build_rca_graph", "run_rca_investigation"]
