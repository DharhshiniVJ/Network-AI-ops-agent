"""
Global configuration for the Agentic AI Network RCA Framework.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
TOPOLOGY_DIR = DATA_DIR / "topologies"
TELEMETRY_DIR = DATA_DIR / "telemetry"
RESULTS_DIR = DATA_DIR / "results"

# Create data directories
for d in [TOPOLOGY_DIR, TELEMETRY_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL_MAP = {
    "gemini": "gemini-1.5-flash",
    "openai": "gpt-4o-mini",
    "ollama": "qwen2.5:7b-instruct",
}
LLM_MODEL = LLM_MODEL_MAP.get(LLM_PROVIDER, "gemini-2.0-flash")

# Simulation Parameters
RANDOM_SEED = 42
TELEMETRY_INTERVAL_SEC = 15  # Polling interval in seconds
SIMULATION_DURATION_SEC = 7200  # 2 hours

# Anomaly Detection Parameters
ANOMALY_CONTAMINATION = 0.035  # Tuned closer to actual fault injection rate to improve precision
ISOLATION_FOREST_ESTIMATORS = 200
ISOLATION_FOREST_MAX_SAMPLES = 256

# Agent Parameters
MAX_AGENT_STEPS = 8  # Max diagnostic tool calls per investigation

# Feature Engineering
ROLLING_WINDOW_SIZE = 4  # Rolling window for statistical features
