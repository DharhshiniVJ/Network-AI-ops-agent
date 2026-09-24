#!/usr/bin/env python3
"""
Train & Evaluate ML Models on Real Mininet Kernel Telemetry
=============================================================
Run this AFTER mininet_topo.py has generated mininet_telemetry.csv.

Usage:
    python3 train_on_mininet.py

This script:
  1. Loads the labeled Mininet dataset
  2. Splits 60/20/20 (Train/Val/Test) — stratified by anomaly type
  3. Trains Random Forest (Tier 1) and LSTM-Autoencoder (Tier 2)
  4. Reports full benchmark metrics on the held-out Test set
  5. Saves the new Mininet-trained models to mininet_models/
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import torch

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.feature_engineering import NetworkFeaturePipeline
from detection.lstm_autoencoder import LSTMAutoencoderDetector
from detection.baseline_models import BaselineDetector
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE


# ─── 1. Load Data ────────────────────────────────────────────────────────────

def load_and_split(csv_path='mininet_telemetry.csv'):
    print(f"\n[*] Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['device_id', 'interface_id', 'timestamp']).reset_index(drop=True)

    total  = len(df)
    anoms  = df['is_anomaly'].sum()
    print(f"    Total records   : {total:,}")
    print(f"    Anomaly records : {anoms:,} ({anoms/total*100:.1f}%)")
    print(f"\n    Records per fault type:")
    for atype, cnt in df.groupby('anomaly_type')['is_anomaly'].count().items():
        print(f"      {atype:<22}: {cnt:,}")

    # ── Stratified Split ─────────────────────────────────────────
    # Normal traffic: chronological 60/20/20
    normal_df = df[df['is_anomaly'] == 0].copy()
    n = len(normal_df)
    train_normal = normal_df.iloc[:int(n * 0.60)]
    val_normal   = normal_df.iloc[int(n * 0.60):int(n * 0.80)]
    test_normal  = normal_df.iloc[int(n * 0.80):]

    # Each anomaly type: stratified 60/20/20
    train_parts = [train_normal]
    val_parts   = [val_normal]
    test_parts  = [test_normal]

    for atype in df[df['is_anomaly'] == 1]['anomaly_type'].unique():
        adf = df[df['anomaly_type'] == atype].copy()
        m   = len(adf)
        train_parts.append(adf.iloc[:int(m * 0.60)])
        val_parts.append(adf.iloc[int(m * 0.60):int(m * 0.80)])
        test_parts.append(adf.iloc[int(m * 0.80):])

    train_df = pd.concat(train_parts).sort_values('timestamp').reset_index(drop=True)
    val_df   = pd.concat(val_parts).sort_values('timestamp').reset_index(drop=True)
    test_df  = pd.concat(test_parts).sort_values('timestamp').reset_index(drop=True)

    print(f"\n    Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"    Train anomalies: {train_df['is_anomaly'].sum():,}")
    print(f"    Val   anomalies: {val_df['is_anomaly'].sum():,}")
    print(f"    Test  anomalies: {test_df['is_anomaly'].sum():,}")

    return train_df, val_df, test_df


# ─── 2. Train Random Forest ───────────────────────────────────────────────────

def train_random_forest(train_df, val_df, test_df):
    print("\n" + "=" * 55)
    print("  TIER 1: Training Random Forest on Mininet Data")
    print("=" * 55)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import RobustScaler

    pipeline = NetworkFeaturePipeline()
    X_train, y_train = pipeline.fit_transform(train_df)
    X_val,   y_val   = pipeline.transform(val_df)
    X_test,  y_test  = pipeline.transform(test_df)

    # SMOTE to balance classes
    if y_train.sum() > 0:
        sm = SMOTE(random_state=42, k_neighbors=min(5, int(y_train.sum()) - 1))
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"  SMOTE applied → {len(X_train):,} training samples")

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float('nan')
    pr  = average_precision_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float('nan')

    print(f"\n  ROC-AUC : {roc:.4f}")
    print(f"  PR-AUC  : {pr:.4f}")
    print(f"\n{classification_report(y_test, y_pred, digits=4, zero_division=0, target_names=['Normal', 'Anomaly'])}")

    # Wrap in BaselineDetector interface so test_mininet_ml.py can use it
    detector = BaselineDetector.__new__(BaselineDetector)
    detector.pipeline = pipeline
    detector.model    = clf
    detector.name     = 'RandomForest_Mininet'

    def predict_fn(df_in):
        X, _ = pipeline.transform(df_in)
        return clf.predict(X), clf.predict_proba(X)[:, 1]

    detector.predict = predict_fn

    return detector, {'roc_auc': roc, 'pr_auc': pr}


# ─── 3. Train LSTM-Autoencoder ───────────────────────────────────────────────

def train_lstm(train_df, val_df, test_df):
    print("\n" + "=" * 55)
    print("  TIER 2: Training LSTM-Autoencoder on Mininet Data")
    print("=" * 55)

    lstm = LSTMAutoencoderDetector(epochs=50, batch_size=256)
    lstm.train(train_df, val_df)

    # Evaluate
    preds, scores = lstm.predict(test_df)
    y_test = test_df['is_anomaly'].values

    roc = roc_auc_score(y_test, scores) if len(np.unique(y_test)) > 1 else float('nan')
    pr  = average_precision_score(y_test, scores) if len(np.unique(y_test)) > 1 else float('nan')

    print(f"\n  ROC-AUC : {roc:.4f}")
    print(f"  PR-AUC  : {pr:.4f}")
    print(f"\n{classification_report(y_test, preds, digits=4, zero_division=0, target_names=['Normal', 'Anomaly'])}")

    return lstm, {'roc_auc': roc, 'pr_auc': pr}


# ─── 4. Save Models ───────────────────────────────────────────────────────────

def save_models(rf_detector, lstm_detector, out_dir='mininet_models'):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(rf_detector,   os.path.join(out_dir, 'random_forest.pkl'))
    torch.save(lstm_detector,  os.path.join(out_dir, 'lstm_autoencoder.pth'))
    print(f"\n[✓] Mininet-trained models saved to ./{out_dir}/")
    print("    → random_forest.pkl")
    print("    → lstm_autoencoder.pth")


# ─── 5. Final Summary ─────────────────────────────────────────────────────────

def print_summary(rf_metrics, lstm_metrics):
    print("\n" + "=" * 55)
    print("  FINAL BENCHMARK RESULTS (Mininet-Trained)")
    print("=" * 55)
    print(f"  {'Model':<30} {'ROC-AUC':>8} {'PR-AUC':>8}")
    print(f"  {'-'*46}")
    print(f"  {'Random Forest (Tier 1)':<30} {rf_metrics['roc_auc']:>8.4f} {rf_metrics['pr_auc']:>8.4f}")
    print(f"  {'LSTM-Autoencoder (Tier 2)':<30} {lstm_metrics['roc_auc']:>8.4f} {lstm_metrics['pr_auc']:>8.4f}")
    print("=" * 55)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    csv = 'mininet_telemetry.csv'
    if not os.path.exists(csv):
        print(f"ERROR: {csv} not found. Run mininet_topo.py first.")
        sys.exit(1)

    train_df, val_df, test_df = load_and_split(csv)
    rf_det,   rf_m            = train_random_forest(train_df, val_df, test_df)
    lstm_det, lstm_m          = train_lstm(train_df, val_df, test_df)

    save_models(rf_det, lstm_det)
    print_summary(rf_m, lstm_m)
