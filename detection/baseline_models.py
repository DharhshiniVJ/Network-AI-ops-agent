import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    xgb = None
    HAS_XGBOOST = False
from detection.feature_engineering import NetworkFeaturePipeline

try:
    import config
except ImportError:
    class config:
        RANDOM_SEED = 42

class BaselineDetector:
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.pipeline = NetworkFeaturePipeline()
        self.model = None
        self.seed = getattr(config, 'RANDOM_SEED', 42)
        
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None) -> None:
        X_scaled, y = self.pipeline.fit_transform(train_df)
        
        # Apply SMOTE to balance the classes for training
        try:
            from imblearn.over_sampling import SMOTE
            # Only apply SMOTE if there's enough samples of both classes
            if len(np.unique(y)) > 1 and np.sum(y == 1) > 5:
                smote = SMOTE(random_state=self.seed)
                X_train_res, y_train_res = smote.fit_resample(X_scaled, y)
                print(f"SMOTE applied: {X_scaled.shape} -> {X_train_res.shape}")
            else:
                X_train_res, y_train_res = X_scaled, y
        except Exception as e:
            print(f"SMOTE failed: {e}")
            X_train_res, y_train_res = X_scaled, y
        
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=self.seed,
                n_jobs=-1
            )
            self.model.fit(X_train_res, y_train_res)
            
        elif self.model_type == "xgboost":
            if not HAS_XGBOOST:
                raise ImportError(
                    "XGBoost is not available. Install it with: pip install xgboost\n"
                    "On macOS you may also need: brew install libomp"
                )
            # Compute scale_pos_weight
            num_neg = np.sum(y_train_res == 0)
            num_pos = np.sum(y_train_res == 1)
            spw = num_neg / max(1, num_pos)
            
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                scale_pos_weight=spw,
                eval_metric="logloss",
                random_state=self.seed,
                n_jobs=-1
            )
            self.model.fit(X_train_res, y_train_res)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
            
    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled, _ = self.pipeline.transform(test_df)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        return predictions, probabilities
        
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
