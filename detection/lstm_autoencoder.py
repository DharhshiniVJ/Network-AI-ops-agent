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
# PyTorch Deep LSTM Autoencoder Architecture
# ---------------------------------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder: 128 -> 64
        self.enc_lstm1 = nn.LSTM(input_dim, 128, batch_first=True)
        self.enc_lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        
        # Latent Bottleneck: Dense 32
        self.enc_dense = nn.Linear(64, 32)
        
        # Decoder Expansion: Dense 64
        self.dec_dense = nn.Linear(32, 64)
        
        # Decoder: 128 -> input_dim
        self.dec_lstm1 = nn.LSTM(64, 128, batch_first=True)
        self.dec_lstm2 = nn.LSTM(128, input_dim, batch_first=True)
        
    def forward(self, x):
        # Encode
        x1, _ = self.enc_lstm1(x)
        x1 = self.dropout(x1)
        x2, (hidden, _) = self.enc_lstm2(x1)
        
        # We take the output of the last time step for the bottleneck
        last_hidden = hidden[-1] 
        latent = torch.relu(self.enc_dense(last_hidden))
        
        # Decode
        dec_in = torch.relu(self.dec_dense(latent))
        # Repeat the decoded vector seq_len times to feed into decoder LSTM
        dec_in = dec_in.unsqueeze(1).repeat(1, x.size(1), 1)
        
        x3, _ = self.dec_lstm1(dec_in)
        x3 = self.dropout(x3)
        dec_out, _ = self.dec_lstm2(x3)
        
        return dec_out

# ---------------------------------------------------------
# Detector Class (Matches IsolationForestDetector interface)
# ---------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler

class LSTMAutoencoderDetector:
    def __init__(self, seq_len: int = 10, epochs: int = 50, batch_size: int = 128):
        self.pipeline = NetworkFeaturePipeline()
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = None
        self.optimal_threshold = None
        self.feature_weights = None
        
        # We MUST use MinMaxScaler for LSTMs to prevent gradient explosion
        self.dl_scaler = MinMaxScaler(feature_range=(0, 1))
        
        torch.manual_seed(getattr(config, 'RANDOM_SEED', 42))

    def _build_feature_weights(self) -> np.ndarray:
        """
        FIX 3: Per-feature weighted MSE.
        Diagnostic features (link status, discards, CPU) get higher weights
        because they carry the strongest anomaly signal.
        """
        feature_names = self.pipeline.get_feature_names()
        weights = np.ones(len(feature_names))
        
        high_signal_features = {
            'is_link_down': 5.0,
            'is_dropping_packets': 4.0,
            'is_high_latency': 3.0,
            'cpu_utilization': 3.0,
            'packet_loss_rate': 3.0,
            'zscore_in_octets': 2.0,
            'zscore_out_octets': 2.0,
            'zscore_out_discards': 2.0,
            'discard_intensity': 2.0,
        }
        
        for i, name in enumerate(feature_names):
            if name in high_signal_features:
                weights[i] = high_signal_features[name]
        
        # Normalize so weights sum to len(features)
        weights = weights * len(weights) / weights.sum()
        return weights

    def _weighted_mse(self, x: torch.Tensor, x_hat: torch.Tensor) -> np.ndarray:
        """Compute per-sample weighted MSE across features."""
        # x, x_hat shape: (batch, seq_len, features)
        weights_tensor = torch.FloatTensor(self.feature_weights).to(self.device)
        # Weighted squared error per feature, averaged over seq_len
        sq_err = torch.pow(x - x_hat, 2)  # (batch, seq, feat)
        weighted_sq_err = sq_err * weights_tensor  # broadcast over batch and seq
        mse = torch.mean(weighted_sq_err, dim=[1, 2])  # (batch,)
        return mse.cpu().numpy()

    def _create_sequences(self, X: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        num_samples, num_features = X.shape
        X_seq = np.zeros((num_samples, self.seq_len, num_features))
        
        for i in range(num_samples):
            start_idx = max(0, i - self.seq_len + 1)
            seq = X[start_idx : i + 1]
            if len(seq) < self.seq_len:
                pad_len = self.seq_len - len(seq)
                pad = np.repeat(seq[0:1], pad_len, axis=0)
                seq = np.vstack([pad, seq])
            X_seq[i] = seq
            
        return X_seq

    def _batched_predict_mse(self, X_seq: np.ndarray) -> np.ndarray:
        """Run inference in batches to prevent OOM on large datasets."""
        dataset = TensorDataset(torch.FloatTensor(X_seq))
        # Use a slightly larger batch size for inference if possible, but self.batch_size is safe
        dataloader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False)
        all_mse = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_x, in dataloader:
                batch_x = batch_x.to(self.device)
                preds = self.model(batch_x)
                mse = self._weighted_mse(batch_x, preds)
                all_mse.append(mse)
                
        return np.concatenate(all_mse)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None) -> None:
        print(f"Training LSTM-Autoencoder on {self.device}...")
        
        # ----------------------------------------------------------------
        # FIX 2: Train on CLEAN normal data only
        # ----------------------------------------------------------------
        clean_train_df = train_df[train_df['is_anomaly'] == 0].copy()
        dirty_count = len(train_df) - len(clean_train_df)
        print(f"FIX 2: Filtered out {dirty_count} anomaly samples from training. Training on {len(clean_train_df)} clean normal samples only.")
        
        # 1. Feature Engineering (fit scaler on clean data only)
        X_robust, _ = self.pipeline.fit_transform(clean_train_df)
        
        # 2. Strict DL Scaling (0 to 1)
        X_scaled = self.dl_scaler.fit_transform(X_robust)
        
        # 3. Build feature weights for weighted MSE
        self.feature_weights = self._build_feature_weights()
        print(f"FIX 3: Feature weights built for {len(self.feature_weights)} features.")
        
        # 4. Sequence Creation
        X_seq = self._create_sequences(X_scaled, clean_train_df)
        
        # 5. PyTorch Setup (Deep SOTA Architecture)
        input_dim = X_seq.shape[2]
        self.model = LSTMAutoencoder(input_dim=input_dim).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        dataset = TensorDataset(torch.FloatTensor(X_seq), torch.FloatTensor(X_seq))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 6. Training Loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.6f}")
                
        # 7. Threshold Calibration (Using Validation Set to maximize F1-Score)
        self.model.eval()
        with torch.no_grad():
            if val_df is not None:
                print("Calibrating threshold on Validation Set (with feature weights)...")
                X_robust_val, y_val = self.pipeline.transform(val_df)
                X_scaled_val = self.dl_scaler.transform(X_robust_val)
                X_seq_val = self._create_sequences(X_scaled_val, val_df)
                
                mse_val = self._batched_predict_mse(X_seq_val)
                
                # Grid search over percentile thresholds for max F1
                if len(np.unique(y_val)) > 1:
                    from sklearn.metrics import f1_score as sklearn_f1
                    
                    best_f1 = 0
                    best_threshold = np.percentile(mse_val, 98)
                    
                    for pct in np.arange(80, 99.5, 0.5):
                        t = np.percentile(mse_val, pct)
                        preds = (mse_val > t).astype(int)
                        f1 = sklearn_f1(y_val, preds, zero_division=0)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = t
                    
                    self.optimal_threshold = best_threshold
                    print(f"LSTM-AE SOTA Threshold (Max Val F1={best_f1:.4f}): {self.optimal_threshold:.6f}")
                else:
                    self.optimal_threshold = np.percentile(mse_val, 98)
            else:
                # Fallback if no val_df provided
                mse = self._batched_predict_mse(X_seq)
                self.optimal_threshold = np.percentile(mse, 98)
                print(f"LSTM-AE Blind Statistical Threshold (98th Percentile): {self.optimal_threshold:.6f}")

    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_robust, _ = self.pipeline.transform(test_df)
        X_scaled = self.dl_scaler.transform(X_robust)
        X_seq = self._create_sequences(X_scaled, test_df)
        
        mse = self._batched_predict_mse(X_seq)
            
        binary_predictions = np.where(mse >= self.optimal_threshold, 1, 0)
        return binary_predictions, mse

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        _, y_true = self.pipeline.transform(test_df)
        binary_preds, scores = self.predict(test_df)
        
        roc_auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else np.nan
        pr_auc = average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else np.nan
        
        cls_report = classification_report(y_true, binary_preds, output_dict=True, zero_division=0)
        
        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "classification_report": cls_report,
            "predictions": binary_preds,
            "scores": scores
        }

