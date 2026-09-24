"""
LangGraph ReAct Agent for Autonomous Network Root Cause Analysis
================================================================
Replaces the OpenAI-based react_agent.py with a proper LangGraph
StateGraph implementation.

Architecture:
  Input Alert
      │
      ▼
  [agent node] ── LLM reasons, picks a tool ──► [tools node] ── executes ──┐
      ▲                                                                      │
      └──────────────────────────── loop ───────────────────────────────────┘
      │
      ▼  (LLM returns final answer — no more tool calls)
  [finalize node] ── formats structured RCA report ──► Output
"""

import json
import os
from typing import Annotated, Sequence, TypedDict, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool as lc_tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import config

# ─── Agent State ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Immutable-ish state threaded through every graph node."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    steps:    int        # how many tool calls so far
    alert:    dict       # original alert dict for reference


# ─── LangGraph Graph Builder ──────────────────────────────────────────────────

def build_rca_graph(tools: list, llm):
    """
    Builds and compiles the LangGraph ReAct graph.

    Args:
        tools:  List of LangChain @tool decorated functions.
        llm:    A LangChain chat model already bound with the tools.

    Returns:
        A compiled LangGraph app (call .invoke() on it).
    """
    llm_with_tools = llm.bind_tools(tools)
    tool_node      = ToolNode(tools)

    # ── Node: Agent (LLM reasoning step) ─────────────────────────────────────
    def agent_node(state: AgentState) -> dict:
        if state["steps"] >= config.MAX_AGENT_STEPS:
            # Force the LLM to wrap up by injecting a nudge
            forced = HumanMessage(
                content="You have used the maximum number of tool calls. "
                        "Please provide your final RCA report now based on "
                        "the evidence gathered so far."
            )
            msgs = list(state["messages"]) + [forced]
            response = llm_with_tools.invoke(msgs)
        else:
            response = llm_with_tools.invoke(state["messages"])

        return {
            "messages": [response],
            "steps":    state["steps"],
        }

    # ── Node: Tools (execution step) ─────────────────────────────────────────
    def tool_execution_node(state: AgentState) -> dict:
        result = tool_node.invoke(state)
        return {
            "messages": result["messages"],
            "steps":    state["steps"] + 1,
        }

    # ── Edge condition: should we call a tool, or are we done? ────────────────
    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        # If the LLM returned tool calls → route to tool node
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    # ── Build the graph ───────────────────────────────────────────────────────
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_execution_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ─── High-Level Runner ────────────────────────────────────────────────────────

def run_rca_investigation(alert: dict, tools: list, llm, system_prompt: str,
                          verbose: bool = True) -> dict:
    """
    Run a full ReAct RCA investigation for a given alert.

    Args:
        alert:         Alert dict with keys: device_id, interface_id,
                       anomaly_type, mse_score, tier (1 or 2), timestamp.
        tools:         List of LangChain @tool functions.
        llm:           LangChain chat model.
        system_prompt: Agent system prompt string.
        verbose:       Print step-by-step reasoning.

    Returns:
        dict with keys: final_report, steps, trajectory, status
    """
    from agent.prompts import ALERT_TEMPLATE

    alert_message = ALERT_TEMPLATE.format(
        device_id    = alert.get("device_id",    "unknown"),
        interface_id = alert.get("interface_id", "unknown"),
        anomaly_type = alert.get("anomaly_type", "unknown"),
        mse_score    = alert.get("mse_score",    "N/A"),
        tier         = alert.get("tier",          2),
        timestamp    = alert.get("timestamp",    "unknown"),
    )

    app = build_rca_graph(tools, llm)

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=alert_message),
        ],
        "steps": 0,
        "alert": alert,
    }

    final_state = app.invoke(initial_state)

    # Extract trajectory from ToolMessages in the message history
    trajectory = []
    messages   = final_state["messages"]
    for i, msg in enumerate(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                # Find the corresponding ToolMessage response
                obs = next(
                    (m.content for m in messages[i+1:]
                     if isinstance(m, ToolMessage) and m.tool_call_id == tc["id"]),
                    "No response found"
                )
                trajectory.append({
                    "tool":       tc["name"],
                    "arguments":  tc["args"],
                    "observation": obs,
                })
                if verbose:
                    print(f"\n  🔧 Tool : {tc['name']}({json.dumps(tc['args'])})")
                    print(f"  👁 Result: {obs[:300]}{'...' if len(str(obs)) > 300 else ''}")

    final_msg = final_state["messages"][-1]
    final_report = final_msg.content if hasattr(final_msg, "content") else str(final_msg)

    if verbose:
        print("\n" + "═" * 60)
        print("  📋 FINAL RCA REPORT")
        print("═" * 60)
        print(final_report)

    return {
        "final_report": final_report,
        "steps":        final_state["steps"],
        "trajectory":   trajectory,
        "status":       "completed" if final_state["steps"] < config.MAX_AGENT_STEPS else "max_steps",
    }
