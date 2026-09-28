import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from detection.evaluator import DetectionEvaluator

# Load the python dataset
snmp_path = "data/snmp_telemetry.parquet"
df = pd.read_parquet(snmp_path)

if 'is_anomaly' not in df.columns:
    df['is_anomaly'] = 0
if 'timestamp' in df.columns:
    df = df.sort_values('timestamp')

# Split data (same logic as detection.tiered_triage)
n = len(df)
train_df = df.iloc[:int(0.6*n)]
val_df = df.iloc[int(0.6*n):int(0.8*n)]
test_df = df.iloc[int(0.8*n):]

evaluator = DetectionEvaluator()
results = evaluator.run_full_evaluation(train_df, val_df, test_df)

print("\n==== FINAL METRICS ON PYTHON DATASET ====")
for model, mets in results.items():
    print(f"\n{model}:")
    for k, v in mets.items():
        if k in ('roc_auc', 'pr_auc'):
            print(f"  {k}: {v:.4f}")
        elif k == 'classification_report':
            anomaly = v.get('1', {})
            weighted = v.get('weighted avg', {})
            print(f"  Precision:  {anomaly.get('precision', 0):.4f}")
            print(f"  Recall:     {anomaly.get('recall', 0):.4f}")
            print(f"  F1-Score:   {anomaly.get('f1-score', 0):.4f}")
            print(f"  W-Avg F1:   {weighted.get('f1-score', 0):.4f}")

