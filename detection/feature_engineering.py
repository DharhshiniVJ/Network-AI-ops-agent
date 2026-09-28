import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import RobustScaler

try:
    import config
except ImportError:
    # fallback
    class config:
        ROLLING_WINDOW_SIZE = 12

class NetworkFeaturePipeline:
    def __init__(self):
        self.scaler = RobustScaler()
        self.is_fitted = False
        
        # We explicitly override config here to guarantee smoothing
        self.window_size = 12
        
        self.raw_features = [
            "ifInOctets", "ifOutOctets", "ifInDiscards", "ifOutDiscards", 
            "ifInErrors", "latency_ms", "packet_loss_rate", "cpu_utilization"
        ]
        self.engineered_features = [
            "delta_in_octets", "delta_out_octets",
            "is_link_down", "is_high_latency", "is_dropping_packets",
            "rolling_mean_in", "rolling_std_in", "zscore_in_octets",
            "rolling_mean_out", "rolling_std_out", "zscore_out_octets",
            "rolling_mean_disc", "rolling_std_disc", "zscore_out_discards",
            "in_out_ratio", "discard_intensity",
            "error_rate", "burst_ratio",
        ]
        
    def get_feature_names(self) -> List[str]:
        # POINT 4: Feature Selection
        # We explicitly DROP the raw cumulative counters (ifInOctets, ifOutOctets, ifInDiscards) 
        # from the model input because cumulative counters grow infinitely over time, 
        # ruining the Euclidean distance calculations for the Isolation Forest.
        # We only keep the rates (deltas) and the rolling Z-scores.
        selected_raw = [
            "latency_ms", "packet_loss_rate", "cpu_utilization"
        ]
        return selected_raw + self.engineered_features

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Sort by group and timestamp to ensure correct calculation
        if "timestamp" in df.columns:
            df = df.sort_values(by=["device_id", "interface_id", "timestamp"])
            
            # Cyclical time
            if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                seconds = df["timestamp"].dt.hour * 3600 + df["timestamp"].dt.minute * 60 + df["timestamp"].dt.second
            else:
                timestamps = pd.to_datetime(df["timestamp"])
                seconds = timestamps.dt.hour * 3600 + timestamps.dt.minute * 60 + timestamps.dt.second
                
            df["sin_time"] = np.sin(2 * np.pi * seconds / 86400)
            df["cos_time"] = np.cos(2 * np.pi * seconds / 86400)
        else:
            df["sin_time"] = 0.0
            df["cos_time"] = 0.0

        # Group by
        grouped = df.groupby(["device_id", "interface_id"])
        
        # Temporal diffs
        df["delta_in_octets"] = grouped["ifInOctets"].diff().fillna(0)
        df["delta_out_octets"] = grouped["ifOutOctets"].diff().fillna(0)
        
        # Explicit State Features (Crucial for supervised trees to branch cleanly)
        df["is_link_down"] = (df["ifOperStatus"] == 2).astype(int)
        df["is_high_latency"] = (df["latency_ms"] > 10.0).astype(int)
        df["is_dropping_packets"] = (df["packet_loss_rate"] > 0.0).astype(int)
        
        # Rolling stats
        window = self.window_size
        
        # InOctets Rolling & Z-Score
        df["rolling_mean_in"] = grouped["ifInOctets"].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df["rolling_std_in"] = grouped["ifInOctets"].transform(lambda x: x.rolling(window, min_periods=1).std()).fillna(0)
        df["zscore_in_octets"] = (df["ifInOctets"] - df["rolling_mean_in"]) / (df["rolling_std_in"] + 1e-5)
        
        # OutOctets Rolling & Z-Score
        df["rolling_mean_out"] = grouped["ifOutOctets"].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df["rolling_std_out"] = grouped["ifOutOctets"].transform(lambda x: x.rolling(window, min_periods=1).std()).fillna(0)
        df["zscore_out_octets"] = (df["ifOutOctets"] - df["rolling_mean_out"]) / (df["rolling_std_out"] + 1e-5)
        
        # Discards Rolling & Z-Score
        df["rolling_mean_disc"] = grouped["ifOutDiscards"].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df["rolling_std_disc"] = grouped["ifOutDiscards"].transform(lambda x: x.rolling(window, min_periods=1).std()).fillna(0)
        df["zscore_out_discards"] = (df["ifOutDiscards"] - df["rolling_mean_disc"]) / (df["rolling_std_disc"] + 1e-5)
        
        # Domain ratios
        df["in_out_ratio"]    = (df["ifInOctets"] + 1) / (df["ifOutOctets"] + 1)
        df["discard_intensity"] = (df["ifOutDiscards"] + df["ifInDiscards"]) / ((df["ifInOctets"] / 1000) + 1)

        # error_rate: normalise errors by traffic volume — catches MTU mismatch cleanly
        df["error_rate"] = (df["ifInErrors"] + 1) / (df["ifInOctets"] + 1)

        # burst_ratio: current delta relative to rolling mean — stronger congestion signal
        df["burst_ratio"] = df["delta_in_octets"].abs() / (df["rolling_mean_in"] + 1)
        
        # Fill missing values created by shifts/rolling if any
        df = df.fillna(0)
        
        return df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        engineered_df = self.engineer_features(df)
        X = engineered_df[self.get_feature_names()].values
        
        y = np.zeros(len(df))
        if "is_anomaly" in engineered_df.columns:
            y = engineered_df["is_anomaly"].values
            
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Pipeline has not been fitted yet. Call fit_transform first.")
            
        engineered_df = self.engineer_features(df)
        X = engineered_df[self.get_feature_names()].values
        
        y = np.zeros(len(df))
        if "is_anomaly" in engineered_df.columns:
            y = engineered_df["is_anomaly"].values
            
        X_scaled = self.scaler.transform(X)
        return X_scaled, y
