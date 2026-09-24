"""
Tier 3 Agent Entry Point
========================
Runs the full 3-tier AIOps pipeline:
  1. Load trained models + telemetry
  2. Run Tier 1 (RF) + Tier 2 (LSTM) detection
  3. For each high-confidence alert, invoke the LangGraph ReAct agent
  4. Print/save structured RCA reports

Usage:
    export GOOGLE_API_KEY=your_key_here
    python3 main.py agent

Or run standalone:
    python3 agent/run_agent.py
"""

import os
import sys
import json
import torch
import joblib
import pandas as pd
import numpy as np
from datetime import timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.tools.diagnostic_tools import ALL_TOOLS, inject_network_state
from agent.react_agent import run_rca_investigation
from agent.prompts import RCA_SYSTEM_PROMPT
from network.topology import NetworkTopology


# ─── LLM Provider ─────────────────────────────────────────────────────────────

def get_llm():
    """Returns a LangChain chat model based on env config."""
    provider = os.getenv("LLM_PROVIDER", "gemini")

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Set GOOGLE_API_KEY environment variable.\n"
                "  export GOOGLE_API_KEY=your_key_here"
            )
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.1,
            thinking_budget=0,   # disable thinking — avoids thought_signature errors with LangGraph
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model="qwen2.5:7b-instruct", temperature=0.1)

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Choose: gemini, openai, ollama")


# ─── State Builder ────────────────────────────────────────────────────────────

def build_network_state(snmp_df: pd.DataFrame, syslogs: list,
                        topology: NetworkTopology, lstm_scores: np.ndarray,
                        snmp_df_sorted: pd.DataFrame) -> None:
    """
    Converts raw telemetry data into the NetworkState format that tools read.
    """
    # 1. Metrics: latest reading per (device, interface)
    latest = (
        snmp_df.sort_values("timestamp")
               .groupby(["device_id", "interface_id"])
               .last()
               .reset_index()
    )
    metrics = {}
    for _, row in latest.iterrows():
        dev  = row["device_id"]
        intf = row["interface_id"]
        metrics.setdefault(dev, {})[intf] = {
            "ifOperStatus":    int(row.get("ifOperStatus", 1)),
            "latency_ms":      float(row.get("latency_ms", 0)),
            "packet_loss_rate":float(row.get("packet_loss_rate", 0)),
            "ifInOctets":      int(row.get("ifInOctets", 0)),
            "ifOutOctets":     int(row.get("ifOutOctets", 0)),
            "ifInDiscards":    int(row.get("ifInDiscards", 0)),
            "ifInErrors":      int(row.get("ifInErrors", 0)),
            "cpu_utilization": float(row.get("cpu_utilization", 0)),
        }

    # 2. Anomaly scores per (device, interface)
    scores = {}
    df_s = snmp_df_sorted.copy()
    df_s["mse_score"] = lstm_scores
    score_latest = (
        df_s.sort_values("timestamp")
            .groupby(["device_id", "interface_id"])["mse_score"]
            .last()
    )
    for (dev, intf), score in score_latest.items():
        scores.setdefault(dev, {})[intf] = float(score)

    # 3. Topology as node-link dict
    topo_dict = topology.to_dict()

    inject_network_state(
        topology = topo_dict,
        metrics  = metrics,
        syslogs  = syslogs,
        scores   = scores,
    )


# ─── Alert Picker ─────────────────────────────────────────────────────────────

def pick_alerts(df: pd.DataFrame, rf_preds: np.ndarray,
                lstm_preds: np.ndarray, lstm_scores: np.ndarray,
                top_n: int = 5) -> list:
    """
    Select the top_n most severe alerts for the agent to investigate.

    Strategy: per (device, interface), find the HIGHEST-MSE record —
    not the last one, which may be post-recovery normal traffic.
    Then rank by priority:
      1. Both RF and LSTM agreed at that peak point (highest confidence)
      2. RF only (known fault, high precision)
      3. LSTM only (zero-day, sorted by MSE)
    """
    df = df.copy()
    df["rf_pred"]    = rf_preds
    df["lstm_pred"]  = lstm_preds
    df["lstm_score"] = lstm_scores

    # For each (device, interface) pick the row with the highest MSE score
    # — this is the peak anomaly moment, not whatever happened last.
    idx_worst = df.groupby(["device_id", "interface_id"])["lstm_score"].idxmax()
    worst = df.loc[idx_worst].reset_index(drop=True)

    both_agree = worst[(worst["rf_pred"] == 1) & (worst["lstm_pred"] == 1)]
    rf_only    = worst[(worst["rf_pred"] == 1) & (worst["lstm_pred"] == 0)]
    lstm_only  = worst[(worst["rf_pred"] == 0) & (worst["lstm_pred"] == 1)]
    lstm_only  = lstm_only.sort_values("lstm_score", ascending=False)

    alerts = []
    for tier_label, tier_df, tier_num in [
        ("Both Tier1+2 (HIGH CONFIDENCE)", both_agree, "1+2"),
        ("Tier 1 — RF Known Fault",         rf_only,   1),
        ("Tier 2 — LSTM Zero-Day",           lstm_only, 2),
    ]:
        for _, row in tier_df.iterrows():
            if len(alerts) >= top_n:
                break
            alerts.append({
                "device_id":    row["device_id"],
                "interface_id": row["interface_id"],
                "anomaly_type": row.get("anomaly_type", "unknown"),
                "mse_score":    f"{row['lstm_score']:.6f}",
                "tier":         tier_num,
                "timestamp":    str(row.get("timestamp", "")),
                "tier_label":   tier_label,
            })
        if len(alerts) >= top_n:
            break

    return alerts




# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_agent_pipeline(snmp_csv=None, max_alerts: int = 3):
    """
    Full 3-tier pipeline: load data → detect → investigate with LangGraph agent.

    Args:
        snmp_csv:   Path to telemetry CSV. Defaults to latest in data/telemetry/.
        max_alerts: How many alerts to investigate (to control API costs).
    """
    print("\n" + "═" * 60)
    print("  TIER 3: LangGraph RCA Agent")
    print("═" * 60)

    # ── Load data ─────────────────────────────────────────────────
    if snmp_csv is None:
        # Check both data/ root (main.py output) and data/telemetry/ subdirectory
        candidates = []
        for search_dir in [Path("data"), Path("data/telemetry")]:
            candidates += sorted(search_dir.glob("snmp_*.parquet"))
            candidates += sorted(search_dir.glob("snmp_*.csv"))
        # Also check for un-prefixed file
        for search_dir in [Path("data"), Path("data/telemetry")]:
            if (search_dir / "snmp_telemetry.parquet").exists():
                candidates.append(search_dir / "snmp_telemetry.parquet")
            if (search_dir / "snmp_telemetry.csv").exists():
                candidates.append(search_dir / "snmp_telemetry.csv")

        if not candidates:
            raise FileNotFoundError(
                "No telemetry files found. Run: python3 main.py simulate"
            )
        snmp_csv = str(candidates[-1])


    ext = Path(snmp_csv).suffix
    if ext == ".parquet":
        snmp_df = pd.read_parquet(snmp_csv)
    else:
        snmp_df = pd.read_csv(snmp_csv)

    snmp_df["timestamp"] = pd.to_datetime(snmp_df["timestamp"])
    snmp_df_sorted = snmp_df.sort_values(
        ["device_id", "interface_id", "timestamp"]
    ).reset_index(drop=True)

    print(f"[✓] Loaded {len(snmp_df_sorted):,} telemetry records from {snmp_csv}")

    # ── Load models ───────────────────────────────────────────────
    rf_path   = Path("data/models/random_forest.pkl")
    lstm_path = Path("data/models/lstm_autoencoder.pth")

    if not rf_path.exists() or not lstm_path.exists():
        raise FileNotFoundError(
            "Models not found. Run: python3 main.py detect"
        )

    rf   = joblib.load(rf_path)
    lstm = torch.load(lstm_path, map_location="cpu", weights_only=False)
    lstm.device = torch.device("cpu")
    if hasattr(lstm, "model"):
        lstm.model.to("cpu")
    print("[✓] Loaded Tier 1 (RF) and Tier 2 (LSTM) models")

    # ── Run Tier 1 + 2 detection ──────────────────────────────────
    print("\n[*] Running Tier 1 + 2 anomaly detection...")
    rf_preds,   rf_scores   = rf.predict(snmp_df_sorted)
    lstm_preds, lstm_scores = lstm.predict(snmp_df_sorted)

    n_rf   = rf_preds.sum()
    n_lstm = lstm_preds.sum()
    n_both = ((rf_preds == 1) & (lstm_preds == 1)).sum()
    print(f"    RF flagged   : {n_rf} events")
    print(f"    LSTM flagged : {n_lstm} events")
    print(f"    Both agreed  : {n_both} events (highest priority)")

    # ── Load topology + syslogs ───────────────────────────────────
    topo = NetworkTopology()
    topo.build_hierarchical_wan()

    # ── Load syslog (handle both .txt and .jsonl) ─────────────────
    syslogs = []
    syslog_candidates = (
        sorted(Path("data").glob("syslog*.jsonl")) +
        sorted(Path("data").glob("syslog*.txt")) +
        sorted(Path("data/telemetry").glob("syslog*"))
    )
    if syslog_candidates:
        syslog_file = syslog_candidates[-1]
        with open(syslog_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    import json as _json
                    obj = _json.loads(line)
                    # Convert jsonl record to plain syslog string
                    syslogs.append(
                        f"[{obj.get('timestamp','')}] {obj.get('host','')} "
                        f"{obj.get('facility','')} {obj.get('message','')}"
                    )
                except Exception:
                    syslogs.append(line)
    print(f"[✓] Loaded topology ({len(topo.get_all_devices())} devices) "
          f"and {len(syslogs):,} syslog entries")

    # ── Build NetworkState for tools ──────────────────────────────
    build_network_state(snmp_df, syslogs, topo, lstm_scores, snmp_df_sorted)

    # ── Pick top alerts ───────────────────────────────────────────
    alerts = pick_alerts(snmp_df_sorted, rf_preds, lstm_preds,
                         lstm_scores, top_n=max_alerts)


    if not alerts:
        print("\n[✓] No anomalies detected. Network appears healthy.")
        return []

    print(f"\n[*] Investigating top {len(alerts)} alert(s) with LangGraph Agent...")

    # ── Get LLM ───────────────────────────────────────────────────
    llm = get_llm()

    # ── Run agent for each alert ──────────────────────────────────
    results = []
    for i, alert in enumerate(alerts, 1):
        print(f"\n{'─'*60}")
        print(f"  Alert {i}/{len(alerts)}: {alert['tier_label']}")
        print(f"  Device: {alert['device_id']} | Interface: {alert['interface_id']}")
        print(f"  Anomaly: {alert['anomaly_type']} | MSE: {alert['mse_score']}")
        print(f"{'─'*60}")

        result = run_rca_investigation(
            alert         = alert,
            tools         = ALL_TOOLS,
            llm           = llm,
            system_prompt = RCA_SYSTEM_PROMPT,
            verbose       = True,
        )
        result["alert"] = alert
        results.append(result)

    # ── Save results ──────────────────────────────────────────────
    out_path = Path("results/rca_reports.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[✓] {len(results)} RCA report(s) saved to {out_path}")
    return results


if __name__ == "__main__":
    run_agent_pipeline(max_alerts=3)
