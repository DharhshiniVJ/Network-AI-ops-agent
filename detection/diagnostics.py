#!/usr/bin/env python3
"""
Comprehensive LSTM-AE Diagnostic Script
- Grid search over percentile thresholds on Validation Set
- Confusion matrices at each threshold
- Reconstruction error distribution analysis
- Data leakage check for Random Forest
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    confusion_matrix, average_precision_score, roc_auc_score
)
from detection.lstm_autoencoder import LSTMAutoencoderDetector
from detection.feature_engineering import NetworkFeaturePipeline

# ----------------------------------------------------------------
# 1. Load Data with 60/20/20 Split
# ----------------------------------------------------------------
print("=" * 60)
print("LSTM-AE DIAGNOSTIC REPORT")
print("=" * 60)

df = pd.read_parquet("data/snmp_telemetry.parquet")
if 'timestamp' in df.columns:
    df = df.sort_values('timestamp')

train_idx = int(len(df) * 0.6)
val_idx = int(len(df) * 0.8)

train_df = df.iloc[:train_idx].copy()
val_df = df.iloc[train_idx:val_idx].copy()
test_df = df.iloc[val_idx:].copy()

print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

anomaly_rate_train = train_df['is_anomaly'].mean()
anomaly_rate_val = val_df['is_anomaly'].mean()
anomaly_rate_test = test_df['is_anomaly'].mean()
print(f"\nAnomaly rates -> Train: {anomaly_rate_train:.4f}  Val: {anomaly_rate_val:.4f}  Test: {anomaly_rate_test:.4f}")

# ----------------------------------------------------------------
# 2. Train LSTM-AE (without auto-threshold)
# ----------------------------------------------------------------
print("\n--- Training Deep LSTM-Autoencoder ---")
detector = LSTMAutoencoderDetector(seq_len=10, epochs=50)
# Train WITHOUT val_df so it uses the blind 98th percentile
detector.train(train_df)

# ----------------------------------------------------------------
# 3. Get raw reconstruction errors for Val and Test
# ----------------------------------------------------------------
print("\n--- Computing reconstruction errors ---")

# Val errors
X_robust_val, y_val = detector.pipeline.transform(val_df)
X_scaled_val = detector.dl_scaler.transform(X_robust_val)
X_seq_val = detector._create_sequences(X_scaled_val, val_df)

detector.model.eval()
with torch.no_grad():
    X_tensor_val = torch.FloatTensor(X_seq_val).to(detector.device)
    preds_val = detector.model(X_tensor_val)
    val_errors = torch.mean(torch.pow(X_tensor_val - preds_val, 2), dim=[1, 2]).cpu().numpy()

# Test errors
X_robust_test, y_test = detector.pipeline.transform(test_df)
X_scaled_test = detector.dl_scaler.transform(X_robust_test)
X_seq_test = detector._create_sequences(X_scaled_test, test_df)

with torch.no_grad():
    X_tensor_test = torch.FloatTensor(X_seq_test).to(detector.device)
    preds_test = detector.model(X_tensor_test)
    test_errors = torch.mean(torch.pow(X_tensor_test - preds_test, 2), dim=[1, 2]).cpu().numpy()

# ----------------------------------------------------------------
# 4. Reconstruction Error Distribution Analysis
# ----------------------------------------------------------------
print("\n--- Reconstruction Error Distribution ---")
normal_errors_val = val_errors[y_val == 0]
anomaly_errors_val = val_errors[y_val == 1]

print(f"Normal  errors -> Mean: {normal_errors_val.mean():.6f}  Std: {normal_errors_val.std():.6f}  "
      f"Median: {np.median(normal_errors_val):.6f}  Max: {normal_errors_val.max():.6f}")
print(f"Anomaly errors -> Mean: {anomaly_errors_val.mean():.6f}  Std: {anomaly_errors_val.std():.6f}  "
      f"Median: {np.median(anomaly_errors_val):.6f}  Max: {anomaly_errors_val.max():.6f}")

overlap_threshold = np.percentile(normal_errors_val, 95)
anomalies_above = (anomaly_errors_val > overlap_threshold).sum()
print(f"\nAnomaly errors above Normal 95th pct ({overlap_threshold:.6f}): "
      f"{anomalies_above}/{len(anomaly_errors_val)} ({anomalies_above/len(anomaly_errors_val)*100:.1f}%)")

# Plot distributions
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram
axes[0].hist(normal_errors_val, bins=100, alpha=0.6, label=f'Normal (n={len(normal_errors_val)})', color='blue', density=True)
axes[0].hist(anomaly_errors_val, bins=100, alpha=0.6, label=f'Anomaly (n={len(anomaly_errors_val)})', color='red', density=True)
axes[0].set_xlabel('Reconstruction Error (MSE)')
axes[0].set_ylabel('Density')
axes[0].set_title('Reconstruction Error Distribution')
axes[0].legend()

# Log-scale for better visibility
axes[1].hist(normal_errors_val, bins=100, alpha=0.6, label='Normal', color='blue', density=True)
axes[1].hist(anomaly_errors_val, bins=100, alpha=0.6, label='Anomaly', color='red', density=True)
axes[1].set_xlabel('Reconstruction Error (MSE)')
axes[1].set_ylabel('Density (log scale)')
axes[1].set_title('Reconstruction Error Distribution (Log Scale)')
axes[1].set_yscale('log')
axes[1].legend()

plt.tight_layout()
os.makedirs("results/diagnostics", exist_ok=True)
plt.savefig("results/diagnostics/reconstruction_error_distribution.png", dpi=150)
print("Saved: results/diagnostics/reconstruction_error_distribution.png")

# ----------------------------------------------------------------
# 5. Grid Search over Thresholds on Validation Set
# ----------------------------------------------------------------
print("\n--- Threshold Grid Search on Validation Set ---")
print(f"{'Percentile':>10} | {'Threshold':>12} | {'Precision':>10} | {'Recall':>8} | {'F1':>8} | {'FPR':>8} | {'TP':>6} | {'FP':>6} | {'TN':>6} | {'FN':>6}")
print("-" * 110)

percentiles = np.arange(80, 100, 0.5)
best_f1 = 0
best_threshold = None
best_percentile = None
results_list = []

for pct in percentiles:
    threshold = np.percentile(val_errors, pct)
    preds = (val_errors > threshold).astype(int)
    
    p = precision_score(y_val, preds, zero_division=0)
    r = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    results_list.append({'pct': pct, 'threshold': threshold, 'precision': p, 'recall': r, 'f1': f1, 'fpr': fpr})
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_percentile = pct
    
    print(f"{pct:>10.1f} | {threshold:>12.6f} | {p:>10.4f} | {r:>8.4f} | {f1:>8.4f} | {fpr:>8.4f} | {tp:>6} | {fp:>6} | {tn:>6} | {fn:>6}")

print(f"\n★ BEST Threshold on Validation: percentile={best_percentile}, threshold={best_threshold:.6f}, F1={best_f1:.4f}")

# ----------------------------------------------------------------
# 6. Evaluate Best Threshold on Blind Test Set
# ----------------------------------------------------------------
print("\n--- Evaluating Best Threshold on BLIND Test Set ---")
test_preds = (test_errors > best_threshold).astype(int)

test_p = precision_score(y_test, test_preds, zero_division=0)
test_r = recall_score(y_test, test_preds, zero_division=0)
test_f1 = f1_score(y_test, test_preds, zero_division=0)
test_roc = roc_auc_score(y_test, test_errors)
test_pr_auc = average_precision_score(y_test, test_errors)

tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()

print(f"ROC-AUC:    {test_roc:.4f}")
print(f"PR-AUC:     {test_pr_auc:.4f}")
print(f"Precision:  {test_p:.4f}")
print(f"Recall:     {test_r:.4f}")
print(f"F1-Score:   {test_f1:.4f}")
print(f"\nConfusion Matrix:")
print(f"                 Predicted")
print(f"              Normal  Anomaly")
print(f"Actual Normal  {tn:>6}   {fp:>6}")
print(f"Actual Anomaly {fn:>6}   {tp:>6}")

# ----------------------------------------------------------------
# 7. Plot Precision-Recall-F1 vs Threshold
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
res_df = pd.DataFrame(results_list)
ax.plot(res_df['pct'], res_df['precision'], 'b-o', markersize=3, label='Precision')
ax.plot(res_df['pct'], res_df['recall'], 'r-o', markersize=3, label='Recall')
ax.plot(res_df['pct'], res_df['f1'], 'g-o', markersize=3, label='F1-Score')
ax.axvline(x=best_percentile, color='black', linestyle='--', label=f'Best F1 @ {best_percentile}th pct')
ax.set_xlabel('Threshold Percentile')
ax.set_ylabel('Score')
ax.set_title('Precision / Recall / F1 vs Threshold Percentile')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/diagnostics/threshold_sweep.png", dpi=150)
print("\nSaved: results/diagnostics/threshold_sweep.png")

# ----------------------------------------------------------------
# 8. Random Forest Data Leakage Check
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("RANDOM FOREST DATA LEAKAGE CHECK")
print("=" * 60)

# Check if train and test share the same time periods
if 'timestamp' in train_df.columns:
    train_max = train_df['timestamp'].max()
    val_min = val_df['timestamp'].min()
    test_min = test_df['timestamp'].min()
    print(f"Train max timestamp: {train_max}")
    print(f"Val   min timestamp: {val_min}")
    print(f"Test  min timestamp: {test_min}")
    if train_max < val_min < test_min:
        print("✅ No temporal leakage: Train < Val < Test (strictly ordered)")
    else:
        print("⚠️  POTENTIAL TEMPORAL LEAKAGE: timestamps overlap!")

# Check device overlap
train_devices = set(train_df['device_id'].unique())
test_devices = set(test_df['device_id'].unique())
shared_devices = train_devices & test_devices
print(f"\nDevices in train: {len(train_devices)}, test: {len(test_devices)}, shared: {len(shared_devices)}")
if shared_devices:
    print(f"Shared devices: {shared_devices}")
    print("NOTE: Same devices appear in train and test. This is expected for time-series splits,")
    print("      but correlated sequential samples from the same device can inflate RF accuracy.")

# Check if RF features include any leaky columns
print(f"\nFeature columns used by pipeline:")
pipeline = NetworkFeaturePipeline()
X_check, _ = pipeline.fit_transform(train_df.head(100))
print(f"  Feature count: {X_check.shape[1]}")
print(f"  Columns in raw data: {list(train_df.columns)}")

# Check anomaly_type distribution
print(f"\nAnomaly type distribution in Train:")
print(train_df[train_df['is_anomaly']==1]['anomaly_type'].value_counts().to_string())
print(f"\nAnomaly type distribution in Test:")
print(test_df[test_df['is_anomaly']==1]['anomaly_type'].value_counts().to_string())

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
