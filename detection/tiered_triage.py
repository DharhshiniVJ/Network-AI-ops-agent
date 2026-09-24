import pandas as pd
import joblib
import os
import json
import logging
from typing import Dict, Any

from agent.react_agent import ReActAgent
from agent.llm_provider import LLMProviderFactory
from agent.prompts import RCA_SYSTEM_PROMPT, ALERT_TEMPLATE
from agent.tools.registry import TOOL_REGISTRY, TOOLS_SCHEMA
from detection.feature_engineering import NetworkFeaturePipeline
from detection.lstm_autoencoder import LSTMAutoencoderDetector

logger = logging.getLogger(__name__)

class TieredTriagePipeline:
    def __init__(self, llm_provider: str = "gemini"):
        self.rf_model_path = "data/models/random_forest.pkl"
        self.lstm_model_path = "data/models/lstm_autoencoder.pth"
        self.llm_provider = llm_provider
        self.feature_pipeline = NetworkFeaturePipeline()

    def process_telemetry_window(self, df: pd.DataFrame, network_state: Dict) -> Dict[str, Any]:
        """
        Process a window of telemetry data through the 3-Tier Pipeline:
        Tier 1: Supervised Random Forest (Known Faults)
        Tier 2: Deep Learning LSTM-Autoencoder (Zero-Day Faults)
        Tier 3: Agentic AI (Investigate Zero-Days)
        """
        result = {
            "tier_1_status": "Passed",
            "tier_2_status": "Passed",
            "tier_3_status": "Skipped",
            "final_rca": None,
            "anomaly_detected": False
        }

        if not os.path.exists(self.rf_model_path) or not os.path.exists(self.lstm_model_path):
            raise FileNotFoundError("ML Models not found. Run 'python3 main.py detect' first.")

        # Load Models
        import joblib
        import torch
        rf_model = joblib.load(self.rf_model_path)
        if_model = torch.load(self.lstm_model_path)

        # ---------------------------------------------------------
        # TIER 1: Supervised ML (The Known Fault Filter)
        # ---------------------------------------------------------
        # Using Random Forest to quickly identify standard, known issues.
        rf_preds, rf_scores = rf_model.predict(df)
        
        # We assume if the RF is highly confident (>0.85) it's a known fault
        high_conf_anomalies = [i for i, (p, s) in enumerate(zip(rf_preds, rf_scores)) if p == 1 and s > 0.85]
        
        agent_context = ""
        device, interface = "Unknown", "Unknown"
        
        if high_conf_anomalies:
            result["anomaly_detected"] = True
            result["tier_1_status"] = "Known Fault Detected (Passed to Tier 3 for RCA)"
            
            # Identify the specific device/interface causing the alert
            idx = high_conf_anomalies[0]
            device = df.iloc[idx]['device_id']
            interface = df.iloc[idx]['interface_id']
            
            agent_context = f"Tier 1 (Random Forest) detected a known fault on {device} interface {interface} with {rf_scores[idx]:.2f} confidence. Please investigate WHY this is happening and provide remediation."
            result["tier_2_status"] = "Skipped"
            
        else:
            # ---------------------------------------------------------
            # TIER 2: Unsupervised ML (The Zero-Day Detector)
            # ---------------------------------------------------------
            # If the Supervised model missed it, it might be a novel Zero-Day fault.
            if_preds, if_scores = if_model.predict(df)
            
            anomalies = [i for i, p in enumerate(if_preds) if p == 1]
            if not anomalies:
                result["tier_2_status"] = "No Anomalies Detected"
                result["final_rca"] = "System Normal."
                return result
                
            result["anomaly_detected"] = True
            result["tier_2_status"] = "Zero-Day Anomaly Detected (Passed to Tier 3 for RCA)"
            
            idx = anomalies[0]
            device = df.iloc[idx]['device_id']
            interface = df.iloc[idx]['interface_id']
            
        if high_conf_anomalies:
            alert_message = f"Supervised Alert: Tier 1 detected a Known Fault on {device} interface {interface} (Confidence: {rf_scores[idx]:.2f}). Please run diagnostics to find the root cause of this congestion/failure and recommend remediation."
        else:
            alert_message = f"Unsupervised Tripwire Alert: Tier 2 detected a Zero-Day Anomaly on {device} interface {interface} (Score: {if_scores[idx]:.2f}). Signature does not match known faults. Please thoroughly investigate the routing and logs to determine the unknown root cause."
        # ---------------------------------------------------------
        # TIER 3: Agentic AI (The Investigator)
        # ---------------------------------------------------------
        # The LLM Agent is awoken to investigate the Zero-Day anomaly using tools.
        result["tier_3_status"] = "Agent Deployed"
        
        # Inject network state into tools
        from agent.tools import inject_network_state_to_tools
        inject_network_state_to_tools(network_state)
        
        client, model_name = LLMProviderFactory.get_client(self.llm_provider)
        agent = ReActAgent(
            llm_client=client,
            model=model_name,
            system_prompt=RCA_SYSTEM_PROMPT,
            tools_registry=TOOL_REGISTRY,
            tools_schema=TOOLS_SCHEMA,
            verbose=False
        )
        
        llm_alert = ALERT_TEMPLATE.format(alert_message=alert_message)
        agent_result = agent.run(llm_alert)
        
        result["final_rca"] = agent_result
        return result
