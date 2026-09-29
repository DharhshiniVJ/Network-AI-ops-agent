#!/usr/bin/env python3
"""
Live Demo Pipeline
==================
Replays the saved SNMP parquet in real-time (simulated live mode for Mac),
or reads from a live mininet_telemetry.csv on Ubuntu.

Runs the 3-tier pipeline on each arriving "row batch" and writes state to
demo/state.json so the Streamlit dashboard can read it without websockets.

Usage (Mac — replay mode):
    python3 demo/live_pipeline.py --replay

Usage (Ubuntu — live Mininet):
    python3 demo/live_pipeline.py --live --csv emulation/mininet_telemetry_live.csv
"""

import os, sys, json, time, argparse, traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib, torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.feature_engineering import NetworkFeaturePipeline

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
STATE_FILE = ROOT / "demo" / "state.json"
STATE_FILE.parent.mkdir(exist_ok=True)

# ── State helpers ─────────────────────────────────────────────────────────────

def write_state(state: dict):
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, default=str)
    tmp.replace(STATE_FILE)

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}

def fresh_state() -> dict:
    return {
        "status":         "idle",
        "rows_seen":      0,
        "tier1_flags":    0,
        "tier2_flags":    0,
        "alerts":         [],
        "agent_steps":    [],
        "rca_report":     None,
        "rf_verdict":     None,
        "lstm_verdict":   None,
        "injected_fault": None,
        "last_update":    str(datetime.now()),
    }

# ── Model loader ──────────────────────────────────────────────────────────────

def load_models():
    print("[pipeline] Loading RF model...")
    rf = joblib.load(ROOT / "data/models/random_forest.pkl")

    print("[pipeline] Loading LSTM model...")
    lstm = torch.load(
        ROOT / "data/models/lstm_autoencoder.pth",
        map_location="cpu", weights_only=False
    )
    lstm.device = torch.device("cpu")
    lstm.model  = lstm.model.to("cpu")

    return rf, lstm

# ── Agent runner ──────────────────────────────────────────────────────────────

def run_agent(alert_row: pd.DataFrame, full_df: pd.DataFrame,
              lstm_scores: np.ndarray, state: dict):
    """Invoke Tier 3. Streams think/act/observe steps to state.json."""
    try:
        from agent.run_agent     import get_llm, build_network_state
        from agent.react_agent   import run_rca_investigation
        from agent.tools.diagnostic_tools import ALL_TOOLS, inject_network_state
        from agent.prompts       import RCA_SYSTEM_PROMPT
        from network.topology    import NetworkTopology

        state["agent_steps"] = [{"type": "think",
                                  "text": "Alert confirmed. Beginning investigation..."}]
        write_state(state)

        # Build minimal topology + network state
        topo = NetworkTopology()
        topo.build_spine_leaf()

        try:
            syslogs_raw = (ROOT / "data/syslog.jsonl").read_text().strip().split("\n")
            syslogs = [json.loads(l) for l in syslogs_raw[-50:] if l.strip()]
        except Exception:
            syslogs = []

        # Trim scores to match full_df length to avoid length mismatch
        scores_aligned = lstm_scores[-len(full_df):] if len(lstm_scores) >= len(full_df) \
                         else np.pad(lstm_scores, (len(full_df) - len(lstm_scores), 0))
        try:
            build_network_state(full_df, syslogs, topo, scores_aligned, full_df)
        except Exception:
            pass   # agent still runs without MSE in network state

        llm = get_llm()

        device_id  = str(alert_row["device_id"].iloc[0])
        intf_id    = str(alert_row["interface_id"].iloc[0])
        mse_score  = float(lstm_scores[min(alert_row.index[0], len(lstm_scores)-1)])

        alert = {
            "device_id":    device_id,
            "interface_id": intf_id,
            "anomaly_type": str(alert_row.get("anomaly_type", pd.Series(["unknown"])).iloc[0]),
            "mse_score":    round(mse_score, 4),
            "tier":         2,
            "timestamp":    str(alert_row["timestamp"].iloc[0]),
        }

        state["agent_steps"].append({"type": "act",
            "text": f"Investigating device={device_id} interface={intf_id} MSE={mse_score:.4f}"})
        write_state(state)

        result = run_rca_investigation(
            alert=alert,
            tools=ALL_TOOLS,
            llm=llm,
            system_prompt=RCA_SYSTEM_PROMPT,
            verbose=False,
        )

        # Stream trajectory steps into agent_steps
        steps_log = state["agent_steps"].copy()
        for step in result.get("trajectory", []):
            steps_log.append({"type": "act",
                               "text": f"Tool: {step['tool']}({json.dumps(step['arguments'])})"})
            obs_text = str(step.get("observation", ""))[:300]
            steps_log.append({"type": "observe", "text": obs_text})

        steps_log.append({"type": "think", "text": "Investigation complete. Writing report..."})

        # Parse verdict from final report text
        report_text = result.get("final_report", "")
        if isinstance(report_text, list):
            report_text = "\n".join(str(x) for x in report_text)
        report_text = str(report_text)

        verdict = "UNKNOWN"
        for v in ["LINK_FAILURE","CONGESTION","MTU_MISMATCH","INTERFACE_FLAP",
                  "PACKET_LOSS","FALSE_POSITIVE","ROUTING_LOOP"]:
            if v in report_text.upper():
                verdict = v
                break

        state["agent_steps"] = steps_log
        state["rca_report"]  = {
            "verdict": verdict,
            "summary": report_text[:600],
        }
        write_state(state)

    except Exception as e:
        state["agent_steps"].append({"type": "error", "text": str(e)})
        state["rca_report"]  = {"verdict": "ERROR", "summary": traceback.format_exc()[:400]}
        write_state(state)


# ── Main pipeline loop ────────────────────────────────────────────────────────

def run_replay(rf, lstm, speed: float = 10.0):
    """
    Interactive demo loop:
      - IDLE phase: streams normal-only rows, ticking up the SNMP counter
      - FAULT phase: when inject_fault.py --sim sets injected_fault in state.json,
        pull real rows of that fault type from the parquet, run through RF→LSTM→Agent
      - RECOVER: reset detection state, go back to idle
    """
    print("[pipeline] Loading parquet...")
    df = pd.read_parquet(ROOT / "data/snmp_telemetry.parquet")
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Separate clean vs fault rows by type
    normal_df = df[df["is_anomaly"] == 0].reset_index(drop=True)
    fault_pools = {}
    for ft in df["anomaly_type"].dropna().unique():
        if ft not in ("none", "", "normal"):
            pool = df[df["anomaly_type"] == ft].reset_index(drop=True)
            if len(pool) > 0:
                fault_pools[ft] = pool
    print(f"[pipeline] Normal rows: {len(normal_df):,} | Fault types: {list(fault_pools.keys())}")

    state = fresh_state()
    state["status"] = "running"
    write_state(state)

    BATCH       = 50
    normal_idx  = 0
    context_buf = []       # rolling buffer of recent rows for LSTM context

    print("[pipeline] Streaming normal traffic. Click a fault button in the dashboard.")

    while True:
        # ── Check for injected fault ───────────────────────────────────────────
        current_state = load_state()
        injected = current_state.get("injected_fault")

        if injected and injected != "recover":
            fault_type = injected
            print(f"[pipeline] FAULT INJECTED: {fault_type}")

            # Load fault scenario rows (up to 300 rows)
            if fault_type in fault_pools:
                fault_rows = fault_pools[fault_type].head(300).copy()
            else:
                # Unknown fault type — pick any anomalous rows
                fault_rows = df[df["is_anomaly"] == 1].head(300).copy()

            # Build context: last 200 normal rows + fault rows
            ctx_normal = normal_df.iloc[max(0, normal_idx - 200): normal_idx].copy()
            full_context = pd.concat([ctx_normal, fault_rows], ignore_index=True)

            # ── Tier 1: RF ────────────────────────────────────────────────────
            state["status"] = "running"
            try:
                preds_rf, scores_rf = rf.predict(full_context)
                tier1_hit = int(preds_rf[-len(fault_rows):].sum())
                state["tier1_flags"] += tier1_hit

                if tier1_hit > 0:
                    idx = np.where(preds_rf[-len(fault_rows):])[0][0]
                    fr  = fault_rows.iloc[idx]
                    rf_conf = float(scores_rf[-len(fault_rows):][idx])
                    state["rf_verdict"] = {
                        "fault":      fault_type.replace("_", " ").title(),
                        "confidence": round(rf_conf * 100, 1),
                        "device":     str(fr.get("device_id", "?")),
                        "interface":  str(fr.get("interface_id", "?")),
                    }
                    print(f"[pipeline] RF flagged: {state['rf_verdict']}")
                write_state(state)
            except Exception as e:
                print(f"[pipeline] RF error: {e}")

            time.sleep(1.5)   # let dashboard show RF result

            # ── Tier 2: LSTM ──────────────────────────────────────────────────
            try:
                _, scores_lstm = lstm.predict(full_context)
                batch_scores = scores_lstm[-len(fault_rows):]
                tier2_hit = int((batch_scores >= lstm.optimal_threshold).sum())
                state["tier2_flags"] += tier2_hit

                if tier2_hit > 0:
                    top_mse = float(batch_scores.max())
                    state["lstm_verdict"] = {
                        "mse":       round(top_mse, 6),
                        "threshold": round(float(lstm.optimal_threshold), 6),
                        "ratio":     round(top_mse / max(float(lstm.optimal_threshold), 1e-9), 2),
                    }
                    print(f"[pipeline] LSTM confirmed: MSE={top_mse:.6f}")
                write_state(state)
            except Exception as e:
                print(f"[pipeline] LSTM error: {e}")
                scores_lstm = np.zeros(len(full_context))
                tier2_hit = 0

            time.sleep(1.5)   # let dashboard show LSTM result

            # ── Tier 3: Agent ─────────────────────────────────────────────────
            if tier2_hit > 0:
                flagged = fault_rows.iloc[np.where(batch_scores >= lstm.optimal_threshold)[0]]
                alert_row = flagged.head(1)
                alert_entry = {
                    "ts":        str(fault_rows.iloc[0].get("timestamp", str(datetime.now()))),
                    "device":    str(alert_row.iloc[0].get("device_id", "unknown")),
                    "interface": str(alert_row.iloc[0].get("interface_id", "?")),
                    "mse":       round(float(batch_scores.max()), 6),
                    "fault":     fault_type,
                }
                state["alerts"].insert(0, alert_entry)
                state["alerts"] = state["alerts"][:20]

                state["status"] = "investigating"
                write_state(state)

                all_scores = np.zeros(len(full_context))
                all_scores[-len(scores_lstm):] = scores_lstm
                run_agent(alert_row, full_context, all_scores, state)
                state["status"] = "running"

            # Clear the injected fault so it doesn't re-trigger
            state["injected_fault"] = None
            write_state(state)
            print("[pipeline] Fault cycle complete. Waiting for next injection or Recover.")

            # Wait here until user clicks Recover
            while True:
                cs = load_state()
                if cs.get("injected_fault") == "recover" or cs.get("injected_fault") is None:
                    break
                time.sleep(1)

        elif injected == "recover":
            # Reset detection state, keep SNMP counter
            print("[pipeline] Recovering — resetting detection state.")
            rows_seen = state.get("rows_seen", 0)
            t1 = state.get("tier1_flags", 0)
            t2 = state.get("tier2_flags", 0)
            alerts = state.get("alerts", [])
            state = fresh_state()
            state["status"] = "running"
            state["rows_seen"]   = rows_seen
            state["tier1_flags"] = t1
            state["tier2_flags"] = t2
            state["alerts"]      = alerts
            write_state(state)

        else:
            # ── IDLE: stream normal rows ───────────────────────────────────────
            end = min(normal_idx + BATCH, len(normal_df))
            batch = normal_df.iloc[normal_idx: end].copy()
            normal_idx = end if end < len(normal_df) else 0   # loop

            state["rows_seen"]  += len(batch)
            state["last_update"] = str(datetime.now())
            write_state(state)
            time.sleep(BATCH / (30.0 * speed))




def run_live(csv_path: str, rf, lstm):
    """
    Live mode: watches a CSV that Mininet writes to in real-time,
    processes new rows as they appear.
    """
    print(f"[pipeline] Live mode — watching {csv_path}")
    state   = fresh_state()
    state["status"] = "running"
    write_state(state)

    last_pos   = 0
    seen_header = False
    all_rows   = []

    while True:
        try:
            with open(csv_path) as f:
                lines = f.readlines()

            if not seen_header and len(lines) > 1:
                header = lines[0]
                seen_header = True

            new_lines = lines[last_pos + 1:]
            if not new_lines:
                time.sleep(1)
                continue

            last_pos = len(lines) - 1

            import io
            chunk = pd.read_csv(io.StringIO(header + "".join(new_lines)))
            all_rows.append(chunk)
            full_df = pd.concat(all_rows, ignore_index=True)

            # Run detection on last 200 rows for context
            context = full_df.tail(200).copy()
            _, scores_lstm = lstm.predict(context)
            preds_rf, _    = rf.predict(context)

            tier2_hits = (scores_lstm >= lstm.optimal_threshold).sum()
            state["rows_seen"]   = len(full_df)
            state["tier1_flags"] += int(preds_rf.sum())
            state["tier2_flags"] += int(tier2_hits)
            state["last_update"]  = str(datetime.now())

            if tier2_hits > 0 and state["rca_report"] is None:
                state["status"] = "investigating"
                write_state(state)
                flagged = context.iloc[np.where(scores_lstm >= lstm.optimal_threshold)[0]]
                run_agent(flagged.head(1), full_df,
                          np.pad(scores_lstm, (len(full_df) - len(scores_lstm), 0)),
                          state)
                state["status"] = "running"

            write_state(state)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[pipeline] Error: {e}")
            time.sleep(2)

    state["status"] = "done"
    write_state(state)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIOps Live Demo Pipeline")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--replay", action="store_true",
                       help="Replay saved parquet (Mac demo mode)")
    group.add_argument("--live",   action="store_true",
                       help="Watch a live CSV from Mininet (Ubuntu)")
    parser.add_argument("--csv",   default="emulation/mininet_telemetry_live.csv",
                        help="Path to live CSV (only with --live)")
    parser.add_argument("--speed", type=float, default=15.0,
                        help="Replay speed multiplier (default 15×)")
    args = parser.parse_args()

    if not os.environ.get("GOOGLE_API_KEY"):
        raise EnvironmentError("Set GOOGLE_API_KEY before running: export GOOGLE_API_KEY=your_key")

    rf_model, lstm_model = load_models()

    if args.replay:
        run_replay(rf_model, lstm_model, speed=args.speed)
    else:
        run_live(args.csv, rf_model, lstm_model)
