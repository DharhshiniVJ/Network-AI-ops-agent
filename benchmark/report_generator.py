import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

class ReportGenerator:
    def generate_detection_report(self, detection_results: Dict, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        
        # Save comparison table as CSV
        if "comparison_df" in detection_results:
            df = detection_results["comparison_df"]
            df.to_csv(os.path.join(save_dir, "detection_comparison.csv"), index=False)
            
        # Plot curves and matrices
        if "evaluator" in detection_results and "results" in detection_results:
            evaluator = detection_results["evaluator"]
            results = detection_results["results"]
            
            roc_path = os.path.join(save_dir, "roc_curves.png")
            evaluator.plot_roc_curves(results, save_path=roc_path)
            
            cm_path = os.path.join(save_dir, "confusion_matrices.png")
            evaluator.plot_confusion_matrices(results, save_path=cm_path)
            
        print(f"Detection report saved to {save_dir}")

    def generate_agent_report(self, agent_results: Dict, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        
        individual = agent_results.get("individual_results", [])
        aggregate = agent_results.get("aggregate_results", {})
        
        # Save aggregate metrics
        with open(os.path.join(save_dir, "agent_aggregate_metrics.json"), "w") as f:
            json.dump(aggregate, f, indent=4)
            
        # Create per-scenario results table
        rows = []
        for res in individual:
            eval_res = res.get("evaluation", {})
            rows.append({
                "Scenario ID": res.get("scenario"),
                "Accuracy": eval_res.get("accuracy", 0),
                "Steps": eval_res.get("steps_taken", 0),
                "Root Cause Found": eval_res.get("root_cause_found", False)
            })
            
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(save_dir, "agent_per_scenario.csv"), index=False)
            
            # Save step distribution histogram
            plt.figure(figsize=(8, 5))
            sns.histplot(df["Steps"], bins=10, kde=True)
            plt.title("Distribution of Steps Taken by Agent")
            plt.xlabel("Steps")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(save_dir, "agent_steps_distribution.png"))
            plt.close()
            
        print(f"Agent report saved to {save_dir}")

    def generate_full_report(self, detection_results: Dict, agent_results: Dict, save_dir: str):
        self.generate_detection_report(detection_results, os.path.join(save_dir, "detection"))
        self.generate_agent_report(agent_results, os.path.join(save_dir, "agent"))
        
        # Optionally generate combined LaTeX table
        latex = self.create_latex_tables(agent_results)
        with open(os.path.join(save_dir, "tables.tex"), "w") as f:
            f.write(latex)
            
        print(f"Full report generated at {save_dir}")

    def create_latex_tables(self, results: Dict) -> str:
        aggregate = results.get("aggregate_results", {})
        latex = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|l|c|}\n\\hline\n"
        latex += "Metric & Value \\\\\n\\hline\n"
        latex += f"Total Scenarios & {aggregate.get('total_scenarios', 0)} \\\\\n"
        latex += f"Average Accuracy & {aggregate.get('average_accuracy', 0):.2f} \\\\\n"
        latex += f"Average Steps & {aggregate.get('average_steps', 0):.2f} \\\\\n"
        latex += "\\hline\n\\end{tabular}\n\\caption{Agent Evaluation Results}\n\\label{tab:agent_results}\n\\end{table}\n"
        return latex
