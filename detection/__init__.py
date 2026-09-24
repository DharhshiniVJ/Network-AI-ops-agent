"""
ML-based anomaly detection package.

Provides feature engineering, Isolation Forest detection,
and supervised ML baselines for network telemetry analysis.
"""
from detection.feature_engineering import NetworkFeaturePipeline
from detection.isolation_forest import IsolationForestDetector

# Lazy imports for modules with optional heavy dependencies (XGBoost needs libomp)
try:
    from detection.baseline_models import BaselineDetector
except Exception:
    BaselineDetector = None

try:
    from detection.evaluator import DetectionEvaluator
except Exception:
    DetectionEvaluator = None

__all__ = ["NetworkFeaturePipeline", "IsolationForestDetector", "BaselineDetector", "DetectionEvaluator"]
