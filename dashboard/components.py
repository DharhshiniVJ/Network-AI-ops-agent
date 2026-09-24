import streamlit as st
from pyvis.network import Network
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any
import networkx as nx
import tempfile
import os

def render_topology_pyvis(topology: Any, height: str = "500px") -> str:
    """Creates a pyvis HTML string for the network topology."""
    # Create pyvis network
    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="black")
    net.barnes_hut()
    
    G = topology.graph
    
    # Add nodes
    for node_id, node_data in G.nodes(data=True):
        device_type = node_data.get("device_type", "unknown")
        
        # Color nodes by device type
        color_map = {
            "core": "red",
            "dist": "orange",
            "access": "blue",
            "spine": "purple",
            "leaf": "green",
            "unknown": "gray"
        }
        color = color_map.get(device_type, "gray")
        
        # Add node with label
        label = f"{node_id}\n({device_type})"
        title = f"Device: {node_id}\nType: {device_type}\nIP: {node_data.get('mgmt_ip', 'N/A')}"
        net.add_node(node_id, label=label, title=title, color=color, shape="dot")
        
    # Add edges
    for u, v, edge_data in G.edges(data=True):
        status = edge_data.get("status", "up")
        color = "green" if status.lower() == "up" else "red"
        title = f"Link: {u} - {v}\nBandwidth: {edge_data.get('bandwidth_mbps', 'N/A')} Mbps\nStatus: {status}"
        net.add_edge(u, v, color=color, title=title)
        
    # Generate HTML string
    # Pyvis doesn't easily return just string in older versions without saving, so we use a temp file
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        temp_filename = f.name
    
    net.save_graph(temp_filename)
    
    with open(temp_filename, "r", encoding="utf-8") as f:
        html_str = f.read()
        
    os.remove(temp_filename)
    
    return html_str

def plot_telemetry_timeseries(snmp_df: pd.DataFrame, device_id: str, interface_id: str) -> plt.Figure:
    """Plots ifInOctets, latency, discards with anomaly regions shaded."""
    # Filter data
    df = snmp_df[(snmp_df["device_id"] == device_id) & (snmp_df["interface_id"] == interface_id)]
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return fig
        
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
    df = df.sort_values("timestamp")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot ifInOctets
    axes[0].plot(df["timestamp"], df["ifInOctets"], color="blue", label="ifInOctets")
    axes[0].set_ylabel("Bytes")
    axes[0].set_title(f"ifInOctets for {device_id} : {interface_id}")
    
    # Plot latency
    if "latency_ms" in df.columns:
        axes[1].plot(df["timestamp"], df["latency_ms"], color="orange", label="Latency")
        axes[1].set_ylabel("ms")
    axes[1].set_title(f"Latency")
    
    # Plot discards
    if "ifInDiscards" in df.columns:
        axes[2].plot(df["timestamp"], df["ifInDiscards"], color="red", label="ifInDiscards")
        axes[2].set_ylabel("Packets")
    axes[2].set_title(f"Discards")
    
    # Highlight anomalies
    if "is_anomaly" in df.columns:
        anomalies = df[df["is_anomaly"] == 1]
        for ax in axes:
            for _, row in anomalies.iterrows():
                ax.axvline(x=row["timestamp"], color='red', alpha=0.3, linestyle='--')
                
    plt.tight_layout()
    return fig

def render_agent_trajectory(trajectory: List[Dict]) -> None:
    """Renders step-by-step expanders in Streamlit for the agent's trajectory."""
    for i, step in enumerate(trajectory, 1):
        action = step.get("action", "")
        action_input = step.get("action_input", {})
        thought = step.get("thought", "")
        observation = step.get("observation", "")
        
        expander_title = f"Step {i}: {action}"
        if action_input:
            # truncate input for title
            args_str = str(action_input)[:50] + ("..." if len(str(action_input)) > 50 else "")
            expander_title += f"({args_str})"
            
        with st.expander(expander_title):
            if thought:
                st.markdown("**🤔 Thought:**")
                st.info(thought)
            
            st.markdown(f"**🛠️ Tool Call:** `{action}`")
            st.json(action_input)
            
            st.markdown("**👀 Observation:**")
            st.markdown(f"```\n{observation}\n```")

def render_evaluation_metrics(eval_result: Dict) -> None:
    """Displays metrics as Streamlit metric cards in columns."""
    st.subheader("Evaluation Metrics")
    cols = st.columns(3)
    
    metrics = {
        "RCA Accuracy": eval_result.get("accuracy", 0.0),
        "Evidence Grounding": eval_result.get("grounding_score", 0.0),
        "Trajectory Efficiency": eval_result.get("efficiency_score", 0.0)
    }
    
    for i, (name, value) in enumerate(metrics.items()):
        with cols[i % 3]:
            # Format as percentage or rounded float
            st.metric(label=name, value=f"{value:.2f}")
            
    if "feedback" in eval_result:
        st.markdown("**Evaluator Feedback:**")
        st.info(eval_result["feedback"])

def format_rca_report(final_answer: str) -> str:
    """Cleans and formats the agent's RCA report for display."""
    if not final_answer:
        return "*No RCA report provided.*"
    return final_answer.strip()
