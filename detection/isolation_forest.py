import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, precision_recall_curve
from detection.feature_engineering import NetworkFeaturePipeline

from sklearn.decomposition import PCA

try:
    import config
except ImportError:
    class config:
        RANDOM_SEED = 42
        ISOLATION_FOREST_ESTIMATORS = 300  
        ISOLATION_FOREST_MAX_SAMPLES = 128 

class IsolationForestDetector:
    def __init__(self):
        self.pipeline = NetworkFeaturePipeline()
        # Add PCA to reduce noise (keep 95% of variance)
        self.pca = PCA(n_components=0.95, random_state=getattr(config, 'RANDOM_SEED', 42))
        
        self.model = IsolationForest(
            n_estimators=getattr(config, 'ISOLATION_FOREST_ESTIMATORS', 300),
            max_samples=getattr(config, 'ISOLATION_FOREST_MAX_SAMPLES', 128),
            contamination="auto",
            random_state=getattr(config, 'RANDOM_SEED', 42),
            n_jobs=-1
        )
        self.optimal_threshold = None
        
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None) -> None:
        X_scaled, y_train = self.pipeline.fit_transform(train_df)
        # Apply PCA to compress noisy features
        X_pca = self.pca.fit_transform(X_scaled)
        
        self.model.fit(X_pca)
        
        # Purely Unsupervised Threshold Calibration
        train_scores = -self.model.decision_function(X_pca)
        self.optimal_threshold = np.percentile(train_scores, 98)
        print(f"Blind Statistical Threshold (98th Percentile) with PCA: {self.optimal_threshold:.4f}")
        
    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled, _ = self.pipeline.transform(test_df)
        X_pca = self.pca.transform(X_scaled)
        
        anomaly_scores = -self.model.decision_function(X_pca)
        
        # Apply the mathematically optimized threshold
        if self.optimal_threshold is not None:
            binary_predictions = np.where(anomaly_scores >= self.optimal_threshold, 1, 0)
        else:
            preds = self.model.predict(X_scaled)
            binary_predictions = np.where(preds == -1, 1, 0)
        
        return binary_predictions, anomaly_scores
        
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
