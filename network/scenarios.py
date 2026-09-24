from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Any

from network.anomaly_injector import AnomalyInjector
from network.topology import NetworkTopology

class FailureScenario:
    def __init__(self, scenario_id: str, name: str, description: str, topology_type: str, 
                 anomaly_configs: List[Dict[str, Any]], ground_truth_fault_class: str,
                 ground_truth_entity: str, ground_truth_interface: str, ground_truth_peer: str,
                 ground_truth_remediation_keywords: List[str], optimal_tool_sequence: List[str],
                 difficulty: str, alert_message: str):
        self.id = scenario_id
        self.name = name
        self.description = description
        self.topology_type = topology_type
        self.anomaly_configs = anomaly_configs
        self.ground_truth = {
            "fault_class": ground_truth_fault_class,
            "entity": ground_truth_entity,
            "interface": ground_truth_interface,
            "peer": ground_truth_peer,
            "remediation_keywords": ground_truth_remediation_keywords
        }
        self.optimal_tool_sequence = optimal_tool_sequence
        self.difficulty = difficulty
        self.alert_message = alert_message

def build_all_scenarios(spine_leaf_topo: NetworkTopology, wan_topo: NetworkTopology, base_time: datetime) -> List[FailureScenario]:
    scenarios = []
    sl_injector = AnomalyInjector(spine_leaf_topo)
    wan_injector = AnomalyInjector(wan_topo)
    
    cfg_01 = sl_injector.create_link_failure(("Spine1", "Leaf1"), base_time, 30.0)
    scenarios.append(FailureScenario(
        scenario_id="SCEN-01", name="Spine-Leaf Link Failure", description="Link failure",
        topology_type="spine_leaf", anomaly_configs=[cfg_01], ground_truth_fault_class="LINK_FAILURE",
        ground_truth_entity="Spine1", ground_truth_interface="HundredGigE1/0/1", ground_truth_peer="Leaf1",
        ground_truth_remediation_keywords=[], optimal_tool_sequence=[], difficulty="easy", alert_message="Alert"
    ))
    
    cfg_02 = wan_injector.create_link_failure(("R1", "R2"), base_time, 30.0)
    scenarios.append(FailureScenario(
        scenario_id="SCEN-02", name="WAN Core Link Failure", description="Link failure",
        topology_type="hierarchical_wan", anomaly_configs=[cfg_02], ground_truth_fault_class="LINK_FAILURE",
        ground_truth_entity="R1", ground_truth_interface="TenGigE0/0/0", ground_truth_peer="R2",
        ground_truth_remediation_keywords=[], optimal_tool_sequence=[], difficulty="easy", alert_message="Alert"
    ))
    
    cfg_03 = sl_injector.create_congestion(("Spine2", "Leaf2"), base_time, 45.0)
    scenarios.append(FailureScenario(
        scenario_id="SCEN-03", name="Leaf Uplink Congestion", description="Congestion",
        topology_type="spine_leaf", anomaly_configs=[cfg_03], ground_truth_fault_class="CONGESTION",
        ground_truth_entity="Leaf2", ground_truth_interface="HundredGigE1/0/2", ground_truth_peer="Spine2",
        ground_truth_remediation_keywords=[], optimal_tool_sequence=[], difficulty="medium", alert_message="Alert: High output drops on Leaf2"
    ))
    
    return scenarios
