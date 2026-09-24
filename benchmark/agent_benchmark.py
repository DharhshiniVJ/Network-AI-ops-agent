"""
Comprehensive Agent Benchmark
==============================
Tests the Tier 3 LangGraph RCA Agent against 25 curated alerts:
  - 4 alerts per fault type × 5 fault types = 20 fault alerts
  - 5 normal (deliberately clean) alerts = false-positive rejection test

Measures:
  - Verdict accuracy per fault type
  - False-positive elimination rate
  - Tool efficiency (avg calls per correct verdict)
  - Confidence calibration

Usage:
    export GOOGLE_API_KEY=your_key
    python3 benchmark/agent_benchmark.py

Output:
    results/agent_benchmark.json
    results/agent_benchmark_report.md
"""

import os, sys, json, re, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
import joblib

from agent.tools.diagnostic_tools import ALL_TOOLS, inject_network_state
from agent.react_agent import run_rca_investigation
from agent.prompts import RCA_SYSTEM_PROMPT
from agent.run_agent import get_llm, build_network_state
from network.topology import NetworkTopology

# ── Config ────────────────────────────────────────────────────────────────────

FAULT_TYPES   = ['link_failure', 'congestion', 'mtu_mismatch',
                 'interface_flap', 'routing_loop']
ALERTS_PER_FAULT = 4
FALSE_POSITIVE_COUNT = 5

VERDICT_MAP = {
    'link_failure'   : ['LINK_FAILURE'],
    'congestion'     : ['CONGESTION'],
    'mtu_mismatch'   : ['MTU_MISMATCH'],
    'interface_flap' : ['INTERFACE_FLAP'],
    'routing_loop'   : ['ROUTING_LOOP'],
    'none'           : ['FALSE_POSITIVE', 'NORMAL', 'NO_FAULT'],
}


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_everything():
    print("[*] Loading telemetry, models, topology...")
    df = pd.read_parquet('data/snmp_telemetry.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['device_id', 'interface_id', 'timestamp']).reset_index(drop=True)

    rf   = joblib.load('data/models/random_forest.pkl')
    lstm = torch.load('data/models/lstm_autoencoder.pth',
                      map_location='cpu', weights_only=False)
    lstm.device = torch.device('cpu')
    lstm.model.to('cpu')

    print("[*] Running Tier 1+2 detection (this takes ~4 min)...")
    rf_preds, _         = rf.predict(df)
    lstm_preds, scores  = lstm.predict(df)

    df['rf_pred']    = rf_preds
    df['lstm_pred']  = lstm_preds
    df['mse_score']  = scores

    topo = NetworkTopology()
    topo.build_hierarchical_wan()

    syslogs = []
    syslog_candidates = (
        sorted(Path("data").glob("syslog*.jsonl")) +
        sorted(Path("data").glob("syslog*.txt"))
    )
    if syslog_candidates:
        with open(syslog_candidates[-1]) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    syslogs.append(
                        f"[{obj.get('timestamp','')}] {obj.get('host','')} "
                        f"{obj.get('facility','')} {obj.get('message','')}"
                    )
                except Exception:
                    syslogs.append(line)

    build_network_state(df, syslogs, topo, scores, df)
    print(f"[✓] Loaded {len(df):,} records | {len(syslogs):,} syslogs | "
          f"RF:{int(rf_preds.sum())} | LSTM:{int(lstm_preds.sum())} flags")
    return df


# ── Alert Sampler ─────────────────────────────────────────────────────────────

def sample_alerts(df: pd.DataFrame) -> list:
    """
    Sample curated alerts with known ground truth.
    For each fault type: pick 4 records with highest MSE where both
    RF and LSTM agreed AND the true label matches.
    For false positives: pick 5 normal records with lowest MSE.
    """
    alerts = []

    # Fault alerts
    for fault_type in FAULT_TYPES:
        candidates = df[
            (df['anomaly_type'] == fault_type) &
            (df['rf_pred'] == 1) &
            (df['lstm_pred'] == 1)
        ].copy()

        if len(candidates) == 0:
            # Fallback: any record with this fault type
            candidates = df[df['anomaly_type'] == fault_type].copy()

        # Pick top-N by MSE, deduplicate by (device, interface)
        best = (
            candidates.sort_values('mse_score', ascending=False)
                      .drop_duplicates(['device_id', 'interface_id'])
                      .head(ALERTS_PER_FAULT)
        )

        for _, row in best.iterrows():
            alerts.append({
                'device_id'    : row['device_id'],
                'interface_id' : row['interface_id'],
                'anomaly_type' : fault_type,
                'mse_score'    : f"{row['mse_score']:.6f}",
                'tier'         : '1+2',
                'timestamp'    : str(row['timestamp']),
                'ground_truth' : fault_type,
            })

    # False-positive alerts (clean normal records)
    normal = df[
        (df['anomaly_type'] == 'none') &
        (df['rf_pred'] == 0) &
        (df['lstm_pred'] == 1)   # LSTM false positive
    ].copy()

    if len(normal) == 0:
        normal = df[df['anomaly_type'] == 'none'].copy()

    fp_picks = (
        normal.sort_values('mse_score', ascending=False)
              .drop_duplicates(['device_id', 'interface_id'])
              .head(FALSE_POSITIVE_COUNT)
    )
    for _, row in fp_picks.iterrows():
        alerts.append({
            'device_id'    : row['device_id'],
            'interface_id' : row['interface_id'],
            'anomaly_type' : 'none',
            'mse_score'    : f"{row['mse_score']:.6f}",
            'tier'         : 2,
            'timestamp'    : str(row['timestamp']),
            'ground_truth' : 'none',
        })

    print(f"[✓] Sampled {len(alerts)} benchmark alerts "
          f"({len(alerts)-FALSE_POSITIVE_COUNT} faults + "
          f"{FALSE_POSITIVE_COUNT} false-positive tests)")
    return alerts


# ── Verdict Parser ────────────────────────────────────────────────────────────

def parse_verdict(report: str) -> str:
    """Extract the Classification tag from the RCA report."""
    # Look for "- **Classification**: LINK_FAILURE" style
    patterns = [
        r'\*\*Classification\*\*[:\s]+([A-Z_]+)',
        r'Classification[:\s]+([A-Z_]+)',
        r'Fault Classification[:\s]+([A-Z_]+)',
    ]
    for pat in patterns:
        m = re.search(pat, report, re.IGNORECASE)
        if m:
            return m.group(1).upper().strip()

    # Fallback: look for known fault keywords in the report
    report_upper = report.upper()
    keyword_map = {
        'LINK_FAILURE'   : ['LINK_FAILURE', 'LINK FAILURE', 'LINK DOWN'],
        'CONGESTION'     : ['CONGESTION', 'HIGH TRAFFIC', 'BANDWIDTH'],
        'MTU_MISMATCH'   : ['MTU_MISMATCH', 'MTU MISMATCH', 'FRAGMENTATION'],
        'INTERFACE_FLAP' : ['INTERFACE_FLAP', 'INTERFACE FLAP', 'FLAPPING'],
        'ROUTING_LOOP'   : ['ROUTING_LOOP', 'ROUTING LOOP', 'TTL EXPIRED'],
        'FALSE_POSITIVE' : ['FALSE_POSITIVE', 'FALSE POSITIVE', 'NO FAULT',
                            'NO ANOMALY', 'HEALTHY', 'NORMAL'],
    }
    for verdict, keywords in keyword_map.items():
        if any(kw in report_upper for kw in keywords):
            return verdict

    return 'UNKNOWN'


def is_correct(verdict: str, ground_truth: str) -> bool:
    """Check if the agent's verdict matches ground truth."""
    expected = VERDICT_MAP.get(ground_truth, [])
    return verdict in expected or any(e in verdict for e in expected)


# ── Run Benchmark ─────────────────────────────────────────────────────────────

def run_benchmark():
    os.makedirs('results', exist_ok=True)
    df      = load_everything()
    alerts  = sample_alerts(df)
    llm     = get_llm()

    results = []
    correct_by_type = {ft: [] for ft in FAULT_TYPES + ['none']}

    print(f"\n{'═'*60}")
    print(f"  AGENT BENCHMARK — {len(alerts)} alerts")
    print(f"{'═'*60}\n")

    for i, alert in enumerate(alerts, 1):
        gt = alert['ground_truth']
        label = alert['anomaly_type'] if gt != 'none' else 'FALSE_POSITIVE_TEST'
        print(f"[{i:02d}/{len(alerts)}] {alert['device_id']}/{alert['interface_id']}"
              f" | GT={gt} | MSE={alert['mse_score']}")

        t0 = time.time()
        try:
            result = run_rca_investigation(
                alert         = alert,
                tools         = ALL_TOOLS,
                llm           = llm,
                system_prompt = RCA_SYSTEM_PROMPT,
                verbose       = False,
            )
            elapsed     = round(time.time() - t0, 1)
            verdict     = parse_verdict(result['final_report'])
            correct     = is_correct(verdict, gt)
            tool_calls  = len(result['trajectory'])

            status = '✅' if correct else '❌'
            print(f"         Verdict={verdict} {status} | "
                  f"Tools={tool_calls} | {elapsed}s")

            correct_by_type[gt].append(correct)
            results.append({
                **alert,
                'verdict'       : verdict,
                'correct'       : correct,
                'tool_calls'    : tool_calls,
                'elapsed_sec'   : elapsed,
                'status'        : result['status'],
                'trajectory'    : result['trajectory'],
                'final_report'  : result['final_report'],
            })

        except Exception as e:
            print(f"         ERROR: {e}")
            results.append({**alert, 'verdict': 'ERROR', 'correct': False,
                            'tool_calls': 0, 'elapsed_sec': 0,
                            'error': str(e), 'final_report': ''})
            correct_by_type[gt].append(False)

    # ── Save JSON ─────────────────────────────────────────────────
    out_json = Path('results/agent_benchmark.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # ── Generate Report ───────────────────────────────────────────
    generate_report(results, correct_by_type)
    print(f"\n[✓] Saved to {out_json}")


# ── Report Generator ──────────────────────────────────────────────────────────

def generate_report(results: list, correct_by_type: dict):
    total      = len(results)
    total_corr = sum(r['correct'] for r in results)
    accuracy   = total_corr / total * 100

    fault_results = [r for r in results if r['ground_truth'] != 'none']
    fp_results    = [r for r in results if r['ground_truth'] == 'none']

    fp_eliminated = sum(r['correct'] for r in fp_results)
    fp_rate       = fp_eliminated / len(fp_results) * 100 if fp_results else 0

    avg_tools = (
        np.mean([r['tool_calls'] for r in results if r['correct']])
        if any(r['correct'] for r in results) else 0
    )

    print(f"\n{'═'*60}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'═'*60}")
    print(f"  Overall Accuracy        : {total_corr}/{total} = {accuracy:.1f}%")
    print(f"  False Positive Elim.    : {fp_eliminated}/{len(fp_results)} = {fp_rate:.1f}%")
    print(f"  Avg Tool Calls (correct): {avg_tools:.1f}")
    print(f"\n  Per-Fault Accuracy:")
    for ft in FAULT_TYPES:
        lst = correct_by_type[ft]
        if lst:
            acc = sum(lst) / len(lst) * 100
            print(f"    {ft:<20}: {sum(lst)}/{len(lst)} = {acc:.0f}%")
    print(f"{'═'*60}")

    md = f"""# Agent Benchmark Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Overall Performance

| Metric | Value |
|:---|:---:|
| Total Alerts Tested | {total} |
| Correct Verdicts | {total_corr} / {total} |
| **Overall Accuracy** | **{accuracy:.1f}%** |
| False Positive Elimination | {fp_eliminated} / {len(fp_results)} = {fp_rate:.1f}% |
| Avg Tool Calls (correct) | {avg_tools:.1f} |

## Per-Fault-Type Accuracy

| Fault Type | Correct | Total | Accuracy |
|:---|:---:|:---:|:---:|
"""
    for ft in FAULT_TYPES:
        lst = correct_by_type[ft]
        if lst:
            acc = sum(lst) / len(lst) * 100
            md += f"| `{ft}` | {sum(lst)} | {len(lst)} | {acc:.0f}% |\n"

    md += f"| **False Positive (normal)** | {fp_eliminated} | {len(fp_results)} | {fp_rate:.0f}% |\n"

    md += "\n## Per-Alert Results\n\n"
    md += "| # | Device | Interface | Ground Truth | Verdict | ✓/✗ | Tools | Time |\n"
    md += "|:--|:--|:--|:--|:--|:--:|:--:|:--:|\n"
    for i, r in enumerate(results, 1):
        status = '✅' if r['correct'] else '❌'
        md += (f"| {i} | {r['device_id']} | {r['interface_id']} | "
               f"`{r['ground_truth']}` | `{r.get('verdict','ERR')}` | "
               f"{status} | {r.get('tool_calls',0)} | {r.get('elapsed_sec',0)}s |\n")

    Path('results/agent_benchmark_report.md').write_text(md)
    print("[✓] Report saved to results/agent_benchmark_report.md")


if __name__ == '__main__':
    run_benchmark()
