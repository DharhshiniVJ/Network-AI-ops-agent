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
        "status":       "idle",
        "rows_seen":    0,
        "tier1_flags":  0,
        "tier2_flags":  0,
        "alerts":       [],        # list of {device, interface, fault, ts}
        "agent_steps":  [],        # list of {type, text} — streams agent reasoning
        "rca_report":   None,
        "injected_fault": None,
        "last_update":  str(datetime.now()),
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

        build_network_state(full_df, syslogs, topo, lstm_scores, full_df)

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
    Replay the saved parquet in batches of 30 rows (≈ 1 second of data at 30Hz).
    speed: multiplier — 10 = replay 10× faster than real-time.
    """
    print("[pipeline] Replay mode — loading parquet...")
    df = pd.read_parquet(ROOT / "data/snmp_telemetry.parquet")
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    state = fresh_state()
    state["status"] = "running"
    write_state(state)

    BATCH = 50    # rows per tick
    total = len(df)
    rf_feat_names = rf.pipeline.get_feature_names() if hasattr(rf, 'pipeline') else None

    all_scores = np.zeros(total)

    print(f"[pipeline] Streaming {total:,} rows in batches of {BATCH}...")

    for start in range(0, total, BATCH):
        batch   = df.iloc[start: start + BATCH].copy()
        context = df.iloc[max(0, start - 200): start + BATCH].copy()

        # ── Tier 1: Random Forest ──────────────────────────────────────────────
        try:
            preds_rf, scores_rf = rf.predict(context)
            tier1_hit = int(preds_rf[-len(batch):].sum())
        except Exception:
            tier1_hit = 0

        # ── Tier 2: LSTM ──────────────────────────────────────────────────────
        try:
            _, scores_lstm = lstm.predict(context)
            batch_scores = scores_lstm[-len(batch):]
            all_scores[start: start + len(batch)] = batch_scores
            tier2_hit = int((batch_scores >= lstm.optimal_threshold).sum())
        except Exception:
            tier2_hit  = 0
            batch_scores = np.zeros(len(batch))

        state["rows_seen"]   += len(batch)
        state["tier1_flags"] += tier1_hit
        state["tier2_flags"] += tier2_hit
        state["last_update"]  = str(datetime.now())

        # New alerts
        if tier2_hit > 0:
            flagged = batch.iloc[np.where(batch_scores >= lstm.optimal_threshold)[0]]
            for _, row in flagged.head(3).iterrows():
                alert = {
                    "ts":        str(row.get("timestamp", "")),
                    "device":    str(row.get("device_id", "unknown")),
                    "interface": str(row.get("interface_id", "?")),
                    "mse":       float(batch_scores[flagged.index.get_loc(row.name)]) if row.name in flagged.index else 0.0,
                    "fault":     str(row.get("anomaly_type", "unknown")),
                }
                # Avoid duplicates
                existing = [a["ts"] + a["device"] for a in state["alerts"]]
                if alert["ts"] + alert["device"] not in existing:
                    state["alerts"].insert(0, alert)
                    state["alerts"] = state["alerts"][:20]   # keep last 20

                    # ── Tier 3: Agent (only for first new alert) ──────────────
                    if len(state["alerts"]) <= 5 and state["rca_report"] is None:
                        state["status"] = "investigating"
                        write_state(state)
                        alert_df = flagged[flagged.index == row.name].copy()
                        if len(alert_df) == 0:
                            alert_df = flagged.head(1)
                        run_agent(alert_df, context, all_scores[:start + BATCH], state)
                        state["status"] = "running"

        write_state(state)
        time.sleep(BATCH / (30.0 * speed))   # simulate real-time at given speed

    state["status"] = "done"
    write_state(state)
    print("[pipeline] Replay complete.")


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
