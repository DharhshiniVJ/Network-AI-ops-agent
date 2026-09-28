import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Any
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from detection.feature_engineering import NetworkFeaturePipeline

try:
    import config
except ImportError:
    class config:
        RANDOM_SEED = 42

# ---------------------------------------------------------
# Temporal Attention Layer
# Computes a weighted sum across all time steps, forcing
# the model to focus on the most anomalous timestep.
# ---------------------------------------------------------
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden_dim)
        scores = self.attn(lstm_out)           # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1) # (batch, seq_len, 1)
        context = (lstm_out * weights).sum(dim=1)  # (batch, hidden_dim)
        return context


# ---------------------------------------------------------
# PyTorch LSTM Autoencoder with Temporal Attention
# ---------------------------------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        # Encoder: input_dim -> 128 -> 64
        self.enc_lstm1 = nn.LSTM(input_dim, 128, batch_first=True, bidirectional=False)
        self.enc_lstm2 = nn.LSTM(128, 64, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.2)

        # Temporal Attention on encoder outputs
        self.attention = TemporalAttention(hidden_dim=64)

        # Bottleneck: 64 -> 32
        self.enc_dense = nn.Linear(64, 32)

        # Expansion: 32 -> 64
        self.dec_dense = nn.Linear(32, 64)

        # Decoder: 64 -> 128 -> input_dim
        self.dec_lstm1 = nn.LSTM(64, 128, batch_first=True)
        self.dec_lstm2 = nn.LSTM(128, input_dim, batch_first=True)

    def forward(self, x):
        # Encode
        x1, _ = self.enc_lstm1(x)
        x1 = self.dropout(x1)
        x2, _ = self.enc_lstm2(x1)  # (batch, seq_len, 64)

        # Attention-weighted context vector instead of last hidden only
        context = self.attention(x2)  # (batch, 64)

        # Bottleneck
        latent = torch.relu(self.enc_dense(context))  # (batch, 32)

        # Decode — repeat latent across seq_len
        dec_in = torch.relu(self.dec_dense(latent))           # (batch, 64)
        dec_in = dec_in.unsqueeze(1).repeat(1, x.size(1), 1)  # (batch, seq_len, 64)

        x3, _ = self.dec_lstm1(dec_in)
        x3 = self.dropout(x3)
        dec_out, _ = self.dec_lstm2(x3)

        return dec_out


# ---------------------------------------------------------
# Detector Class
# ---------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler

class LSTMAutoencoderDetector:
    def __init__(self, seq_len: int = 10, epochs: int = 60, batch_size: int = 128):
        self.pipeline     = NetworkFeaturePipeline()
        self.seq_len      = seq_len
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.device       = torch.device(
            "cuda" if torch.cuda.is_available() else
            ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.model             = None
        self.optimal_threshold = None
        self.feature_weights   = None

        self.dl_scaler = MinMaxScaler(feature_range=(0, 1))
        torch.manual_seed(getattr(config, 'RANDOM_SEED', 42))

    # ------------------------------------------------------------------
    # Feature Weights for Weighted MSE
    # ------------------------------------------------------------------
    def _build_feature_weights(self) -> np.ndarray:
        feature_names = self.pipeline.get_feature_names()
        weights = np.ones(len(feature_names))

        high_signal = {
            'is_link_down':       6.0,
            'is_dropping_packets':5.0,
            'is_high_latency':    4.0,
            'cpu_utilization':    3.0,
            'packet_loss_rate':   3.0,
            'error_rate':         4.0,
            'burst_ratio':        3.0,
            'zscore_in_octets':   2.5,
            'zscore_out_octets':  2.5,
            'zscore_out_discards':2.5,
            'discard_intensity':  2.0,
        }

        for i, name in enumerate(feature_names):
            if name in high_signal:
                weights[i] = high_signal[name]

        weights = weights * len(weights) / weights.sum()
        return weights

    # ------------------------------------------------------------------
    # Weighted MSE
    # ------------------------------------------------------------------
    def _weighted_mse(self, x: torch.Tensor, x_hat: torch.Tensor) -> np.ndarray:
        weights_tensor = torch.FloatTensor(self.feature_weights).to(self.device)
        sq_err         = torch.pow(x - x_hat, 2)          # (batch, seq, feat)
        weighted       = sq_err * weights_tensor           # broadcast
        mse            = torch.mean(weighted, dim=[1, 2])  # (batch,)
        return mse.cpu().numpy()

    # ------------------------------------------------------------------
    # Per-device sequence creation
    # ------------------------------------------------------------------
    def _create_sequences(self, X: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """
        Build sliding-window sequences.
        Resets the window at every device/interface boundary so that
        sequences never span multiple network interfaces.
        """
        num_samples, num_features = X.shape
        X_seq = np.zeros((num_samples, self.seq_len, num_features))

        if 'device_id' in df.columns and 'interface_id' in df.columns:
            group_key = (df['device_id'].astype(str) + '_' +
                         df['interface_id'].astype(str)).values
        else:
            group_key = np.zeros(num_samples, dtype=str)

        for i in range(num_samples):
            # Walk back only while still in the same device/interface
            start_idx = i
            for j in range(i - 1, max(i - self.seq_len, -1) - 1, -1):
                if group_key[j] != group_key[i]:
                    break
                start_idx = j

            seq = X[start_idx: i + 1]
            if len(seq) < self.seq_len:
                pad_len = self.seq_len - len(seq)
                pad = np.repeat(seq[0:1], pad_len, axis=0)
                seq = np.vstack([pad, seq])
            X_seq[i] = seq

        return X_seq

    # ------------------------------------------------------------------
    # Batched inference
    # ------------------------------------------------------------------
    def _batched_predict_mse(self, X_seq: np.ndarray) -> np.ndarray:
        dataset    = TensorDataset(torch.FloatTensor(X_seq))
        dataloader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False)
        all_mse    = []

        self.model.eval()
        with torch.no_grad():
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)
                preds   = self.model(batch_x)
                mse     = self._weighted_mse(batch_x, preds)
                all_mse.append(mse)

        return np.concatenate(all_mse)

    # ------------------------------------------------------------------
    # Score Smoothing — reduces FP spikes from transient normal bursts
    # ------------------------------------------------------------------
    def _smooth_scores(self, scores: np.ndarray, window: int = 3) -> np.ndarray:
        """Apply a rolling median to raw MSE scores."""
        s = pd.Series(scores)
        smoothed = s.rolling(window, min_periods=1, center=True).median().values
        return smoothed.astype(np.float32)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None) -> None:
        print(f"Training LSTM-Autoencoder on {self.device}...")

        # Train on CLEAN normal data only
        clean_train_df = train_df[train_df['is_anomaly'] == 0].copy()
        dirty_count    = len(train_df) - len(clean_train_df)
        print(f"  Filtered {dirty_count} anomaly rows. Training on "
              f"{len(clean_train_df):,} clean normal samples.")

        # Feature Engineering
        X_robust, _  = self.pipeline.fit_transform(clean_train_df)
        X_scaled      = self.dl_scaler.fit_transform(X_robust)

        # Feature weights
        self.feature_weights = self._build_feature_weights()
        print(f"  Feature weights built for {len(self.feature_weights)} features.")

        # Sequences
        X_seq     = self._create_sequences(X_scaled, clean_train_df)
        input_dim = X_seq.shape[2]

        # Model
        self.model = LSTMAutoencoder(input_dim=input_dim).to(self.device)
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        criterion  = nn.MSELoss()

        # Validation sequences for early stopping
        val_loader = None
        if val_df is not None:
            clean_val_df = val_df[val_df['is_anomaly'] == 0].copy()
            if len(clean_val_df) > 0:
                X_val_r, _  = self.pipeline.transform(clean_val_df)
                X_val_s      = self.dl_scaler.transform(X_val_r)
                val_arr      = self._create_sequences(X_val_s, clean_val_df)
                val_dataset  = TensorDataset(torch.FloatTensor(val_arr))
                val_loader   = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        dataset    = TensorDataset(torch.FloatTensor(X_seq))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss  = float('inf')
        patience_count = 0
        early_stop_patience = 10

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss   = criterion(output, batch_x)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(dataloader)

            # Validation loss for scheduler + early stopping
            if val_loader is not None:
                self.model.eval()
                val_total = 0.0
                with torch.no_grad():
                    for (vbatch,) in val_loader:
                        vbatch   = vbatch.to(self.device)
                        vout     = self.model(vbatch)
                        val_total += criterion(vout, vbatch).item()
                val_loss = val_total / len(val_loader)
                self.model.train()

                scheduler.step(val_loss)

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss  = val_loss
                    patience_count = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_count += 1

                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1:3d}/{self.epochs}  "
                          f"train_loss={avg_train_loss:.6f}  "
                          f"val_loss={val_loss:.6f}  "
                          f"lr={optimizer.param_groups[0]['lr']:.6f}")

                if patience_count >= early_stop_patience:
                    print(f"  Early stopping at epoch {epoch+1} "
                          f"(val_loss stagnant for {early_stop_patience} epochs).")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1:3d}/{self.epochs}  "
                          f"train_loss={avg_train_loss:.6f}")

        # Restore best weights if we tracked them
        if val_loader is not None and 'best_state' in locals():
            self.model.load_state_dict(best_state)
            print("  Restored best model weights.")

        # ------------------------------------------------------------------
        # Threshold Calibration — fine-grained sweep on val set
        # ------------------------------------------------------------------
        self.model.eval()
        with torch.no_grad():
            if val_df is not None:
                print("  Calibrating threshold on validation set...")
                X_robust_val, y_val = self.pipeline.transform(val_df)
                X_scaled_val        = self.dl_scaler.transform(X_robust_val)
                X_seq_val           = self._create_sequences(X_scaled_val, val_df)

                raw_mse  = self._batched_predict_mse(X_seq_val)
                mse_val  = self._smooth_scores(raw_mse)

                if len(np.unique(y_val)) > 1:
                    from sklearn.metrics import f1_score as sklearn_f1

                    best_f1        = 0.0
                    best_threshold = np.percentile(mse_val, 98)

                    # Fine-grained sweep: 149 candidates instead of 39
                    for pct in np.arange(85.0, 99.9, 0.1):
                        t     = np.percentile(mse_val, pct)
                        preds = (mse_val > t).astype(int)
                        f1    = sklearn_f1(y_val, preds, zero_division=0)
                        if f1 > best_f1:
                            best_f1        = f1
                            best_threshold = t

                    self.optimal_threshold = best_threshold
                    print(f"  Best threshold at {best_f1:.4f} F1: {self.optimal_threshold:.6f}")
                else:
                    self.optimal_threshold = np.percentile(mse_val, 98)
            else:
                raw_mse                = self._batched_predict_mse(X_seq)
                self.optimal_threshold = np.percentile(self._smooth_scores(raw_mse), 98)
                print(f"  Blind threshold (98th pct): {self.optimal_threshold:.6f}")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_robust, _ = self.pipeline.transform(test_df)
        X_scaled    = self.dl_scaler.transform(X_robust)
        X_seq       = self._create_sequences(X_scaled, test_df)

        raw_mse = self._batched_predict_mse(X_seq)
        mse     = self._smooth_scores(raw_mse)

        binary_predictions = (mse >= self.optimal_threshold).astype(int)
        return binary_predictions, mse

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        _, y_true        = self.pipeline.transform(test_df)
        binary_preds, scores = self.predict(test_df)

        roc_auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else np.nan
        pr_auc  = average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else np.nan

        cls_report = classification_report(
            y_true, binary_preds, output_dict=True, zero_division=0
        )

        return {
            "roc_auc":              roc_auc,
            "pr_auc":               pr_auc,
            "classification_report": cls_report,
            "predictions":          binary_preds,
            "scores":               scores,
        }
