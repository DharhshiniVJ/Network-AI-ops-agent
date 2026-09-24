#!/usr/bin/env python3
"""
Fine-Tune Pre-Trained Models on Real Mininet Kernel Telemetry
==============================================================
Transfer Learning / Domain Adaptation with 3 key fixes:

  FIX 1 (LSTM Precision): Threshold grid-search starts at 93rd percentile
         (not 80th) to prevent over-flagging on small Mininet dataset.
         Broken features (flat cpu_utilization=0.5) are zeroed out before
         scaling so they don't poison the reconstruction error.

  FIX 2 (RF Recall): Prediction threshold lowered from 0.5 to 0.35 so
         subtle faults (mtu_mismatch, latency) that score 0.3–0.45 in
         probability are correctly caught instead of silently discarded.

  FIX 3 (Training quality): LSTM fine-tunes for 50 epochs with cosine LR
         decay. RF uses balanced_subsample weighting and 400 estimators.

Usage (run from the emulation/ directory):
    python3 finetune_on_mininet.py

Output:
    mininet_models/lstm_autoencoder.pth   — fine-tuned LSTM
    mininet_models/random_forest.pkl      — domain-adapted RF
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score as sk_f1

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.lstm_autoencoder import LSTMAutoencoderDetector
from detection.baseline_models import BaselineDetector
from detection.feature_engineering import NetworkFeaturePipeline

SIMULATED_RF_PATH   = '../data/models/random_forest.pkl'
SIMULATED_LSTM_PATH = '../data/models/lstm_autoencoder.pth'
MININET_CSV         = 'mininet_telemetry.csv'
OUT_DIR             = 'mininet_models'

# FIX 1: Features known to be unreliable in Mininet kernel telemetry
# cpu_utilization = flat 0.5 dummy → zero it out before LSTM scaling
BROKEN_FEATURES = ['cpu_utilization']


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_and_split():
    print(f"\n[*] Loading Mininet dataset: {MININET_CSV}")
    df = pd.read_csv(MININET_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['device_id', 'interface_id', 'timestamp']).reset_index(drop=True)

    total = len(df)
    anoms = int(df['is_anomaly'].sum())
    print(f"    Total records   : {total:,}")
    print(f"    Anomaly records : {anoms:,} ({anoms/total*100:.1f}%)")
    print(f"    Fault types     : {sorted(df[df['is_anomaly']==1]['anomaly_type'].unique().tolist())}")

    # Stratified 60/20/20
    normal_df = df[df['is_anomaly'] == 0].copy()
    n = len(normal_df)
    train_parts = [normal_df.iloc[:int(n * 0.60)]]
    val_parts   = [normal_df.iloc[int(n * 0.60):int(n * 0.80)]]
    test_parts  = [normal_df.iloc[int(n * 0.80):]]

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


def zero_broken_features(df: pd.DataFrame) -> pd.DataFrame:
    """FIX 1: Zero out features that are unreliable dummy values in Mininet."""
    df = df.copy()
    for feat in BROKEN_FEATURES:
        if feat in df.columns:
            df[feat] = 0.0
    return df


# ─── LSTM Fine-Tuning ─────────────────────────────────────────────────────────

def finetune_lstm(train_df, val_df, test_df):
    print("\n" + "=" * 60)
    print("  TIER 2: Fine-Tuning LSTM-Autoencoder")
    print("=" * 60)

    lstm = torch.load(SIMULATED_LSTM_PATH, map_location='cpu', weights_only=False)
    lstm.device = torch.device('cpu')
    if hasattr(lstm, 'model'):
        lstm.model.to('cpu')
    print(f"    Original threshold: {lstm.optimal_threshold:.6f}")

    # FIX 1a: Zero out broken/dummy features before any scaling
    # CRITICAL: We must use a CONTIGUOUS block of normal traffic to prevent
    # breaking the time-series sequence. If we just filter (is_anomaly == 0),
    # we create massive artificial jumps bridging pre- and post-anomaly data.
    # The new mininet_topo.py has a 15-minute warmup (900 seconds). We use
    # the first 800 contiguous records of the train_df for pure normal baseline.
    clean_train = zero_broken_features(train_df.iloc[:800])
    print(f"    Zeroed broken features: {BROKEN_FEATURES}")
    print(f"    Fine-tuning on {len(clean_train):,} clean normal Mininet samples")

    X_robust, _ = lstm.pipeline.transform(clean_train)

    # Re-fit MinMax scaler on Mininet baseline distribution
    lstm.dl_scaler.fit(X_robust)
    X_scaled = lstm.dl_scaler.transform(X_robust)
    X_seq    = lstm._create_sequences(X_scaled, clean_train)

    # FIX 3: 50 epochs with cosine LR annealing instead of flat 20 epochs
    FINETUNE_EPOCHS = 50
    FINETUNE_LR     = 0.0001

    optimizer   = torch.optim.Adam(lstm.model.parameters(), lr=FINETUNE_LR)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
                      optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-6)
    dataset     = TensorDataset(torch.FloatTensor(X_seq))
    dataloader  = DataLoader(dataset, batch_size=lstm.batch_size, shuffle=True)

    lstm.model.train()
    print(f"\n    Fine-tuning {FINETUNE_EPOCHS} epochs (cosine LR decay)...")
    for epoch in range(FINETUNE_EPOCHS):
        total_loss = 0
        for batch_x, in dataloader:
            batch_x = batch_x.to('cpu')
            optimizer.zero_grad()
            out  = lstm.model(batch_x)
            loss = nn.MSELoss()(out, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}/{FINETUNE_EPOCHS}, "
                  f"Loss: {total_loss/len(dataloader):.6f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

    # FIX 1b: Threshold calibration — start from 95th percentile
    # Because we now use a purely contiguous 15-minute normal baseline, the
    # reconstruction error variance is much lower. We can tighten the net.
    print("\n    Re-calibrating threshold (searching 95th–99.9th percentile)...")
    lstm.model.eval()

    val_clean = zero_broken_features(val_df)
    X_val_robust, y_val = lstm.pipeline.transform(val_clean)
    X_val_scaled        = lstm.dl_scaler.transform(X_val_robust)
    X_val_seq           = lstm._create_sequences(X_val_scaled, val_clean)
    mse_val             = lstm._batched_predict_mse(X_val_seq)

    if len(np.unique(y_val)) > 1:
        best_f1, best_thresh = 0, np.percentile(mse_val, 97)
        for pct in np.arange(95, 99.9, 0.2):   # FIX: start from 95th, step 0.2
            t     = np.percentile(mse_val, pct)
            preds = (mse_val > t).astype(int)
            f1    = sk_f1(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t
        lstm.optimal_threshold = best_thresh
        print(f"    New threshold: {lstm.optimal_threshold:.6f}  (Val F1={best_f1:.4f})")
    else:
        lstm.optimal_threshold = float(np.percentile(mse_val, 97))
        print(f"    New threshold (97th pct): {lstm.optimal_threshold:.6f}")

    # Evaluate
    test_clean = zero_broken_features(test_df)
    lstm_preds, lstm_scores = lstm.predict(test_clean)
    y_test = test_df['is_anomaly'].values

    roc = roc_auc_score(y_test, lstm_scores)           if len(np.unique(y_test)) > 1 else float('nan')
    pr  = average_precision_score(y_test, lstm_scores) if len(np.unique(y_test)) > 1 else float('nan')

    print(f"\n  ROC-AUC : {roc:.4f}")
    print(f"  PR-AUC  : {pr:.4f}")
    print(classification_report(y_test, lstm_preds, digits=4, zero_division=0,
                                target_names=['Normal', 'Anomaly']))
    return lstm, {'roc_auc': roc, 'pr_auc': pr}


# ─── Random Forest Domain Adaptation ─────────────────────────────────────────

def adapt_random_forest(train_df, val_df, test_df):
    print("\n" + "=" * 60)
    print("  TIER 1: Domain-Adapting Random Forest")
    print("=" * 60)

    rf = joblib.load(SIMULATED_RF_PATH)

    # Zero broken features for RF pipeline too
    train_clean = zero_broken_features(train_df)
    test_clean  = zero_broken_features(test_df)

    new_pipeline = NetworkFeaturePipeline()
    X_train, y_train = new_pipeline.fit_transform(train_clean)
    X_test,  y_test  = new_pipeline.transform(test_clean)

    print(f"    Pipeline re-fitted on {len(train_clean):,} Mininet samples")

    # SMOTE balancing
    from imblearn.over_sampling import SMOTE
    if y_train.sum() > 5:
        k = min(5, int(y_train.sum()) - 1)
        sm = SMOTE(random_state=42, k_neighbors=k)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"    SMOTE applied → {len(X_train):,} training samples")

    # FIX 3: 400 estimators, balanced_subsample for per-tree rebalancing
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        class_weight='balanced_subsample',   # per-tree balance (better than global)
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print("    [✓] RF retrained with balanced_subsample (400 trees)")

    # FIX 2: Use threshold 0.35 instead of 0.5 to catch subtle faults
    # RF default predict() uses 0.5. We override with tuned threshold on val set.
    val_clean = zero_broken_features(val_df)
    X_val, y_val = new_pipeline.transform(val_clean)
    val_proba    = clf.predict_proba(X_val)[:, 1]

    best_f1, best_thresh = 0, 0.35
    for t in np.arange(0.20, 0.55, 0.05):
        preds = (val_proba >= t).astype(int)
        f1    = sk_f1(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    print(f"    Optimal RF threshold: {best_thresh:.2f}  (Val F1={best_f1:.4f})")

    # Evaluate with tuned threshold
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_thresh).astype(int)

    roc = roc_auc_score(y_test, y_prob)           if len(np.unique(y_test)) > 1 else float('nan')
    pr  = average_precision_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float('nan')

    print(f"\n  ROC-AUC : {roc:.4f}")
    print(f"  PR-AUC  : {pr:.4f}")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0,
                                target_names=['Normal', 'Anomaly']))

    # Store tuned threshold on detector object
    rf.pipeline      = new_pipeline
    rf.model         = clf
    rf.clf_threshold = best_thresh

    def predict_fn(df_in):
        df_clean   = zero_broken_features(df_in)
        X, _       = new_pipeline.transform(df_clean)
        proba      = clf.predict_proba(X)[:, 1]
        preds      = (proba >= best_thresh).astype(int)
        return preds, proba

    rf.predict = predict_fn
    return rf, {'roc_auc': roc, 'pr_auc': pr}


# ─── Save & Summary ───────────────────────────────────────────────────────────

def save_and_summarize(rf, lstm, rf_m, lstm_m):
    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(rf,  os.path.join(OUT_DIR, 'random_forest.pkl'))
    torch.save(lstm, os.path.join(OUT_DIR, 'lstm_autoencoder.pth'))

    print("\n" + "=" * 60)
    print("  FINAL BENCHMARK — Fine-Tuned on Mininet Real Data")
    print("=" * 60)
    print(f"  {'Model':<38} {'ROC-AUC':>8} {'PR-AUC':>8}")
    print(f"  {'-'*54}")
    print(f"  {'Random Forest (Tier 1)':<38} {rf_m['roc_auc']:>8.4f} {rf_m['pr_auc']:>8.4f}")
    print(f"  {'LSTM-Autoencoder (Tier 2)':<38} {lstm_m['roc_auc']:>8.4f} {lstm_m['pr_auc']:>8.4f}")
    print("=" * 60)
    print(f"\n[✓] Models saved to ./{OUT_DIR}/")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    for p in [SIMULATED_RF_PATH, SIMULATED_LSTM_PATH, MININET_CSV]:
        if not os.path.exists(p):
            print(f"ERROR: Required file not found: {p}")
            sys.exit(1)

    train_df, val_df, test_df = load_and_split()
    lstm, lstm_m               = finetune_lstm(train_df, val_df, test_df)
    rf,   rf_m                 = adapt_random_forest(train_df, val_df, test_df)
    save_and_summarize(rf, lstm, rf_m, lstm_m)
