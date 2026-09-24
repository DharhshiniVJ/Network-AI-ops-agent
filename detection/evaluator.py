import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from sklearn.metrics import roc_curve, confusion_matrix
import joblib
import os
import torch

from detection.lstm_autoencoder import LSTMAutoencoderDetector
from detection.feature_engineering import NetworkFeaturePipeline
from detection.baseline_models import BaselineDetector

class DetectionEvaluator:
    def __init__(self):
        self.y_true = None
        self.results = None
        self.rf_model_path = "data/models/random_forest.pkl"
        # We won't save the PyTorch model to joblib, we'll save its state dict
        self.lstm_model_path = "data/models/lstm_autoencoder.pth"

    def run_full_evaluation(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        results = {}
        
        # Ensure models dir exists
        os.makedirs("data/models", exist_ok=True)
        
        # 1. LSTM Autoencoder
        print("\n--- Training Tier 2: Deep Learning LSTM-Autoencoder ---")
        lstm_detector = LSTMAutoencoderDetector(seq_len=10, epochs=50)
        lstm_detector.train(train_df, val_df)
        results["LSTMAutoencoder"] = lstm_detector.evaluate(test_df)
        
        # Save PyTorch Model
        torch.save(lstm_detector, self.lstm_model_path)
        
        # Store true labels for plotting (assume same for all)
        _, y_true = lstm_detector.pipeline.transform(test_df)
        self.y_true = y_true
        
        # 2. Random Forest
        rf_detector = BaselineDetector(model_type="random_forest")
        rf_detector.train(train_df, val_df)
        results["RandomForest"] = rf_detector.evaluate(test_df)
        joblib.dump(rf_detector, "data/models/random_forest.pkl")
        
        # 3. XGBoost
        try:
            xgb_detector = BaselineDetector(model_type="xgboost")
            xgb_detector.train(train_df)
            results["XGBoost"] = xgb_detector.evaluate(test_df)
        except ImportError:
            print("Skipping XGBoost (not installed or missing dependencies)")
            
        self.results = results
        return results

    def generate_comparison_table(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        data = []
        for model_name, res in results.items():
            report = res.get("classification_report", {})
            # anomaly class is '1' or '1.0' in classification report
            class_1_metrics = report.get("1", report.get("1.0", {}))
            
            data.append({
                "Model": model_name,
                "ROC-AUC": res.get("roc_auc", np.nan),
                "PR-AUC": res.get("pr_auc", np.nan),
                "Precision": class_1_metrics.get("precision", np.nan),
                "Recall": class_1_metrics.get("recall", np.nan),
                "F1-Score": class_1_metrics.get("f1-score", np.nan)
            })
            
        df_comp = pd.DataFrame(data)
        return df_comp

    def plot_roc_curves(self, results: Dict[str, Dict[str, Any]], save_path: str = None) -> None:
        if self.y_true is None:
            print("y_true not found, run run_full_evaluation first.")
            return
            
        plt.figure(figsize=(8, 6))
        for model_name, res in results.items():
            scores = res["scores"]
            fpr, tpr, _ = roc_curve(self.y_true, scores)
            auc_val = res.get("roc_auc", 0.0)
            if np.isnan(auc_val):
                auc_val = 0.0
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.3f})")
            
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_confusion_matrices(self, results: Dict[str, Dict[str, Any]], save_path: str = None) -> None:
        if self.y_true is None:
            print("y_true not found, run run_full_evaluation first.")
            return
            
        num_models = len(results)
        fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4))
        
        if num_models == 1:
            axes = [axes]
            
        for ax, (model_name, res) in zip(axes, results.items()):
            preds = res["predictions"]
            cm = confusion_matrix(self.y_true, preds)
            
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
            ax.set_title(f"{model_name}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def print_summary(self, results: Dict[str, Dict[str, Any]]) -> None:
        comp_df = self.generate_comparison_table(results)
        print("\n" + "="*50)
        print("MODEL COMPARISON SUMMARY")
        print("="*50)
        print(comp_df.to_string(index=False))
        print("="*50 + "\n")
