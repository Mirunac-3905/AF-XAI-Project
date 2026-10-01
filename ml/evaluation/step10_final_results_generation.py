"""
STEP 10: Final Results and Figures Generation

This script generates the final results and figures for the AF-XAI project.
It uses existing saved metrics and artifacts - no model retraining occurs.

Authoritative experimental result: Step 4 held-out federated evaluation.
Dataset: 20,000 total records (19,440 normal, 560 faults)
Held-out evaluation: 4,449 records (3,889 normal, 560 faults)
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pathlib import Path

# Set style for better-looking figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Paths
PROJECT_ROOT = Path("D:/FINAL-YEAR-PROJECT")
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"

# Ensure directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics():
    """Load all existing metrics from JSON files."""
    with open(METRICS_DIR / "client_1_metrics.json", 'r') as f:
        client_1 = json.load(f)
    with open(METRICS_DIR / "client_2_metrics.json", 'r') as f:
        client_2 = json.load(f)
    with open(METRICS_DIR / "client_3_metrics.json", 'r') as f:
        client_3 = json.load(f)
    with open(METRICS_DIR / "adaptive_global_metrics.json", 'r') as f:
        adaptive = json.load(f)
    with open(METRICS_DIR / "centralized_metrics.json", 'r') as f:
        centralized = json.load(f)
    with open(REPORTS_DIR / "aggregation_weights.json", 'r') as f:
        aggregation_weights = json.load(f)
    
    return {
        "client_1": client_1,
        "client_2": client_2,
        "client_3": client_3,
        "adaptive": adaptive,
        "centralized": centralized,
        "aggregation_weights": aggregation_weights
    }


def task1_final_metrics_table(metrics):
    """TASK 1: Create final metrics table in CSV, JSON, and TXT formats."""
    
    # Create metrics dictionary
    metrics_data = {
        "Client 1": {
            "Accuracy": metrics["client_1"]["accuracy"],
            "Precision": metrics["client_1"]["precision"],
            "Recall": metrics["client_1"]["recall"],
            "F1": metrics["client_1"]["f1_score"],
            "TPR": metrics["client_1"]["tpr"],
            "FPR": metrics["client_1"]["fpr"],
            "ROC-AUC": metrics["client_1"]["roc_auc"]
        },
        "Client 2": {
            "Accuracy": metrics["client_2"]["accuracy"],
            "Precision": metrics["client_2"]["precision"],
            "Recall": metrics["client_2"]["recall"],
            "F1": metrics["client_2"]["f1_score"],
            "TPR": metrics["client_2"]["tpr"],
            "FPR": metrics["client_2"]["fpr"],
            "ROC-AUC": metrics["client_2"]["roc_auc"]
        },
        "Client 3": {
            "Accuracy": metrics["client_3"]["accuracy"],
            "Precision": metrics["client_3"]["precision"],
            "Recall": metrics["client_3"]["recall"],
            "F1": metrics["client_3"]["f1_score"],
            "TPR": metrics["client_3"]["tpr"],
            "FPR": metrics["client_3"]["fpr"],
            "ROC-AUC": metrics["client_3"]["roc_auc"]
        },
        "Adaptive Federated OCSVM": {
            "Accuracy": metrics["adaptive"]["accuracy"],
            "Precision": metrics["adaptive"]["precision"],
            "Recall": metrics["adaptive"]["recall"],
            "F1": metrics["adaptive"]["f1_score"],
            "TPR": metrics["adaptive"]["tpr"],
            "FPR": metrics["adaptive"]["fpr"],
            "ROC-AUC": metrics["adaptive"]["roc_auc"]
        },
        "Centralized OCSVM": {
            "Accuracy": metrics["centralized"]["accuracy"],
            "Precision": metrics["centralized"]["precision"],
            "Recall": metrics["centralized"]["recall"],
            "F1": metrics["centralized"]["f1_score"],
            "TPR": metrics["centralized"]["tpr"],
            "FPR": metrics["centralized"]["fpr"],
            "ROC-AUC": metrics["centralized"]["roc_auc"]
        }
    }
    
    # Save as CSV
    df = pd.DataFrame.from_dict(metrics_data, orient='index')
    df.to_csv(REPORTS_DIR / "final_metrics.csv", float_format='%.6f')
    print("[OK] Created final_metrics.csv")
    
    # Save as JSON
    with open(REPORTS_DIR / "final_metrics.json", 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print("[OK] Created final_metrics.json")
    
    # Save as TXT
    with open(REPORTS_DIR / "final_metrics.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FINAL METRICS TABLE\n")
        f.write("=" * 80 + "\n\n")
        f.write(df.to_string(float_format='%.6f'))
        f.write("\n\n" + "=" * 80 + "\n")
    print("[OK] Created final_metrics.txt")


def task2_confusion_matrix(metrics):
    """TASK 2: Generate adaptive confusion matrix figure."""
    
    cm = np.array([[3654, 235], [226, 334]])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fault'],
                yticklabels=['Normal', 'Fault'],
                cbar_kws={'label': 'Count'})
    plt.title('Adaptive Federated OCSVM Confusion Matrix\n(Held-out Evaluation: 4,449 samples)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "adaptive_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Created adaptive_confusion_matrix.png")


def task3_roc_curve(metrics):
    """TASK 3: Generate adaptive vs centralized ROC curve."""
    
    # Load prediction data
    adaptive_df = pd.read_csv(METRICS_DIR / "adaptive_global_predictions.csv")
    centralized_df = pd.read_csv(METRICS_DIR / "centralized_predictions.csv")
    
    # Use anomaly_score (higher = more anomalous/fault)
    y_true = adaptive_df['fault_indicator'].values
    adaptive_scores = adaptive_df['anomaly_score'].values
    centralized_scores = centralized_df['anomaly_score'].values
    
    # Calculate ROC curves
    fpr_adaptive, tpr_adaptive, _ = roc_curve(y_true, adaptive_scores)
    fpr_centralized, tpr_centralized, _ = roc_curve(y_true, centralized_scores)
    
    roc_auc_adaptive = metrics["adaptive"]["roc_auc"]
    roc_auc_centralized = metrics["centralized"]["roc_auc"]
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_adaptive, tpr_adaptive, linewidth=2, 
             label=f'Adaptive Federated OCSVM (AUC = {roc_auc_adaptive:.4f})')
    plt.plot(fpr_centralized, tpr_centralized, linewidth=2, 
             label=f'Centralized OCSVM (AUC = {roc_auc_centralized:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve: Adaptive Federated vs Centralized OCSVM\n(Held-out Evaluation)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "adaptive_vs_centralized_roc.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Created adaptive_vs_centralized_roc.png")


def task4_model_comparison(metrics):
    """TASK 4: Generate model comparison metrics figure."""
    
    models = ['Client 1', 'Client 2', 'Client 3', 'Adaptive Federated', 'Centralized']
    
    accuracy = [
        metrics["client_1"]["accuracy"],
        metrics["client_2"]["accuracy"],
        metrics["client_3"]["accuracy"],
        metrics["adaptive"]["accuracy"],
        metrics["centralized"]["accuracy"]
    ]
    
    precision = [
        metrics["client_1"]["precision"],
        metrics["client_2"]["precision"],
        metrics["client_3"]["precision"],
        metrics["adaptive"]["precision"],
        metrics["centralized"]["precision"]
    ]
    
    recall = [
        metrics["client_1"]["recall"],
        metrics["client_2"]["recall"],
        metrics["client_3"]["recall"],
        metrics["adaptive"]["recall"],
        metrics["centralized"]["recall"]
    ]
    
    f1 = [
        metrics["client_1"]["f1_score"],
        metrics["client_2"]["f1_score"],
        metrics["client_3"]["f1_score"],
        metrics["adaptive"]["f1_score"],
        metrics["centralized"]["f1_score"]
    ]
    
    roc_auc = [
        metrics["client_1"]["roc_auc"],
        metrics["client_2"]["roc_auc"],
        metrics["client_3"]["roc_auc"],
        metrics["adaptive"]["roc_auc"],
        metrics["centralized"]["roc_auc"]
    ]
    
    # Create separate subplots for each metric
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Comparison Metrics\n(Held-out Evaluation)', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    
    # Accuracy
    bars = axes[0, 0].bar(models, accuracy, color=colors, alpha=0.8)
    axes[0, 0].set_title('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Score', fontsize=11)
    axes[0, 0].set_ylim([0.85, 0.91])
    axes[0, 0].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, accuracy):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Precision
    bars = axes[0, 1].bar(models, precision, color=colors, alpha=0.8)
    axes[0, 1].set_title('Precision', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Score', fontsize=11)
    axes[0, 1].set_ylim([0.55, 0.65])
    axes[0, 1].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, precision):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Recall
    bars = axes[0, 2].bar(models, recall, color=colors, alpha=0.8)
    axes[0, 2].set_title('Recall', fontsize=12, fontweight='bold')
    axes[0, 2].set_ylabel('Score', fontsize=11)
    axes[0, 2].set_ylim([0.55, 0.65])
    axes[0, 2].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, recall):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # F1-score
    bars = axes[1, 0].bar(models, f1, color=colors, alpha=0.8)
    axes[1, 0].set_title('F1-Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Score', fontsize=11)
    axes[1, 0].set_ylim([0.55, 0.65])
    axes[1, 0].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, f1):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # ROC-AUC
    bars = axes[1, 1].bar(models, roc_auc, color=colors, alpha=0.8)
    axes[1, 1].set_title('ROC-AUC', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Score', fontsize=11)
    axes[1, 1].set_ylim([0.88, 0.92])
    axes[1, 1].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, roc_auc):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Remove empty subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Created model_comparison_metrics.png")


def task5_aggregation_weights(metrics):
    """TASK 5: Generate adaptive aggregation weights figure."""
    
    clients = ['Client 1', 'Client 2', 'Client 3']
    weights = [
        metrics["adaptive"]["aggregation_weights"]["client_1"],
        metrics["adaptive"]["aggregation_weights"]["client_2"],
        metrics["adaptive"]["aggregation_weights"]["client_3"]
    ]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(clients, weights, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
    plt.title('Quality-Aware Adaptive Aggregation Weights', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Aggregation Weight', fontsize=12)
    plt.xlabel('Client', fontsize=12)
    plt.ylim([0.32, 0.34])
    
    for bar, weight in zip(bars, weights):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{weight:.6f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "adaptive_aggregation_weights.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Created adaptive_aggregation_weights.png")


def task6_shap_global_importance():
    """TASK 6: Copy/verify SHAP global feature importance figure."""
    
    # Check if the existing figure exists
    existing_shap = FIGURES_DIR / "xai_global_feature_importance.png"
    target_shap = FIGURES_DIR / "shap_global_feature_importance.png"
    
    if existing_shap.exists():
        # Copy to the target name
        import shutil
        shutil.copy2(existing_shap, target_shap)
        print(f"[OK] Copied existing SHAP figure to shap_global_feature_importance.png")
    else:
        print(f"[!] Warning: Existing SHAP figure not found at {existing_shap}")
        print("  Generating new figure from XAI report...")
        
        # Load XAI report and generate figure
        with open(REPORTS_DIR / "xai_report.json", 'r') as f:
            xai_report = json.load(f)
        
        importance = xai_report["global_feature_importance"]
        features = [
            "voltage", "stability_index", "frequency", "renewable_output",
            "temperature", "reactive_power", "active_power", "humidity",
            "power_loss", "load_demand"
        ]
        
        values = [importance[f] for f in features]
        
        plt.figure(figsize=(12, 6))
        bars = plt.barh(features, values, color='steelblue', alpha=0.8)
        plt.title('Mean Absolute SHAP Contribution\n(Adaptive Federated OCSVM)', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Mean |SHAP Value|', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        
        for bar, val in zip(bars, values):
            plt.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(target_shap, dpi=300, bbox_inches='tight')
        plt.close()
        print("[OK] Generated shap_global_feature_importance.png")


def task7_sample_explanations():
    """TASK 7: Verify sample explanation figures exist."""
    
    required_figures = [
        "xai_correct_fault_explanation.png",
        "xai_normal_explanation.png",
        "xai_missed_fault_explanation.png",
        "xai_false_alarm_explanation.png"
    ]
    
    all_exist = True
    for fig_name in required_figures:
        fig_path = FIGURES_DIR / fig_name
        if fig_path.exists():
            print(f"[OK] Found {fig_name}")
        else:
            print(f"[!] Warning: {fig_name} not found")
            all_exist = False
    
    return all_exist


def task8_results_summary(metrics):
    """TASK 8: Create final results summary report."""
    
    with open(REPORTS_DIR / "final_results_summary.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("AF-XAI PROJECT: FINAL RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # 1. Dataset
        f.write("1. DATASET\n")
        f.write("-" * 80 + "\n")
        f.write("Total records: 20,000\n")
        f.write("Normal samples: 19,440 (97.2%)\n")
        f.write("Fault samples: 560 (2.8%)\n")
        f.write("Held-out evaluation: 4,449 records (3,889 normal, 560 faults)\n\n")
        
        # 2. Federated client configuration
        f.write("2. FEDERATED CLIENT CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write("Client 1: 5,011 training samples, 1,431 test samples\n")
        f.write("Client 2: 5,508 training samples, 1,587 test samples\n")
        f.write("Client 3: 5,032 training samples, 1,431 test samples\n\n")
        
        # 3. Held-out evaluation protocol
        f.write("3. HELD-OUT EVALUATION PROTOCOL\n")
        f.write("-" * 80 + "\n")
        f.write("Each client test row is scaled with that client's frozen Step 2 scaler,\n")
        f.write("then scored by the aggregated RBF decision function. Combined metrics\n")
        f.write("concatenate those per-client predictions.\n\n")
        
        # 4. Local model results
        f.write("4. LOCAL MODEL RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write("Client 1:\n")
        f.write(f"  Accuracy: {metrics['client_1']['accuracy']:.6f}\n")
        f.write(f"  Precision: {metrics['client_1']['precision']:.6f}\n")
        f.write(f"  Recall: {metrics['client_1']['recall']:.6f}\n")
        f.write(f"  F1: {metrics['client_1']['f1_score']:.6f}\n")
        f.write(f"  ROC-AUC: {metrics['client_1']['roc_auc']:.6f}\n\n")
        
        f.write("Client 2:\n")
        f.write(f"  Accuracy: {metrics['client_2']['accuracy']:.6f}\n")
        f.write(f"  Precision: {metrics['client_2']['precision']:.6f}\n")
        f.write(f"  Recall: {metrics['client_2']['recall']:.6f}\n")
        f.write(f"  F1: {metrics['client_2']['f1_score']:.6f}\n")
        f.write(f"  ROC-AUC: {metrics['client_2']['roc_auc']:.6f}\n\n")
        
        f.write("Client 3:\n")
        f.write(f"  Accuracy: {metrics['client_3']['accuracy']:.6f}\n")
        f.write(f"  Precision: {metrics['client_3']['precision']:.6f}\n")
        f.write(f"  Recall: {metrics['client_3']['recall']:.6f}\n")
        f.write(f"  F1: {metrics['client_3']['f1_score']:.6f}\n")
        f.write(f"  ROC-AUC: {metrics['client_3']['roc_auc']:.6f}\n\n")
        
        # 5. Adaptive federated results
        f.write("5. ADAPTIVE FEDERATED RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy: {metrics['adaptive']['accuracy']:.6f}\n")
        f.write(f"Precision: {metrics['adaptive']['precision']:.6f}\n")
        f.write(f"Recall: {metrics['adaptive']['recall']:.6f}\n")
        f.write(f"F1: {metrics['adaptive']['f1_score']:.6f}\n")
        f.write(f"FPR: {metrics['adaptive']['fpr']:.6f}\n")
        f.write(f"ROC-AUC: {metrics['adaptive']['roc_auc']:.6f}\n")
        f.write(f"Confusion Matrix: TN={metrics['adaptive']['confusion_matrix']['tn']}, ")
        f.write(f"FP={metrics['adaptive']['confusion_matrix']['fp']}, ")
        f.write(f"FN={metrics['adaptive']['confusion_matrix']['fn']}, ")
        f.write(f"TP={metrics['adaptive']['confusion_matrix']['tp']}\n")
        f.write(f"Combined support vectors: {metrics['adaptive']['n_support_vectors']}\n\n")
        
        # 6. Centralized baseline
        f.write("6. CENTRALIZED BASELINE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy: {metrics['centralized']['accuracy']:.6f}\n")
        f.write(f"Precision: {metrics['centralized']['precision']:.6f}\n")
        f.write(f"Recall: {metrics['centralized']['recall']:.6f}\n")
        f.write(f"F1: {metrics['centralized']['f1_score']:.6f}\n")
        f.write(f"FPR: {metrics['centralized']['fpr']:.6f}\n")
        f.write(f"ROC-AUC: {metrics['centralized']['roc_auc']:.6f}\n")
        f.write(f"Confusion Matrix: TN={metrics['centralized']['confusion_matrix']['tn']}, ")
        f.write(f"FP={metrics['centralized']['confusion_matrix']['fp']}, ")
        f.write(f"FN={metrics['centralized']['confusion_matrix']['fn']}, ")
        f.write(f"TP={metrics['centralized']['confusion_matrix']['tp']}\n\n")
        
        # 7. Aggregation weights
        f.write("7. ADAPTIVE AGGREGATION WEIGHTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Client 1: {metrics['adaptive']['aggregation_weights']['client_1']:.6f}\n")
        f.write(f"Client 2: {metrics['adaptive']['aggregation_weights']['client_2']:.6f}\n")
        f.write(f"Client 3: {metrics['adaptive']['aggregation_weights']['client_3']:.6f}\n")
        f.write("Quality weight formula: quality_score = 0.5 * F1 + 0.5 * ROC_AUC;\n")
        f.write("weighted_quality = quality_score * data_completeness;\n")
        f.write("aggregation_weight_i = weighted_quality_i / sum(weighted_quality)\n\n")
        
        # 8. SHAP findings
        f.write("8. SHAP FINDINGS\n")
        f.write("-" * 80 + "\n")
        f.write("Feature ranking (by mean absolute SHAP contribution):\n")
        f.write("1. voltage\n")
        f.write("2. stability_index\n")
        f.write("3. frequency\n")
        f.write("4. renewable_output\n")
        f.write("5. temperature\n")
        f.write("6. reactive_power\n")
        f.write("7. active_power\n")
        f.write("8. humidity\n")
        f.write("9. power_loss\n")
        f.write("10. load_demand\n")
        f.write("Note: SHAP values explain contribution to the model's decision,\n")
        f.write("not physical causality.\n\n")
        
        # 9. False-positive/missed-fault examples
        f.write("9. FALSE-POSITIVE / MISSED-FAULT EXAMPLES\n")
        f.write("-" * 80 + "\n")
        f.write("Sample explanations are available in results/figures/:\n")
        f.write("- xai_correct_fault_explanation.png\n")
        f.write("- xai_normal_explanation.png\n")
        f.write("- xai_missed_fault_explanation.png\n")
        f.write("- xai_false_alarm_explanation.png\n\n")
        
        # 10. Limitations
        f.write("10. LIMITATIONS\n")
        f.write("-" * 80 + "\n")
        f.write("The adaptive federated model achieved a recall of 59.64% and\n")
        f.write("F1-score of 59.17%, while the centralized baseline achieved\n")
        f.write("57.50% recall and 58.87% F1-score. The adaptive model shows\n")
        f.write("higher recall but slightly lower precision compared to the\n")
        f.write("centralized baseline. The ROC-AUC values are comparable\n")
        f.write("(0.8990 vs 0.8975). Performance differences are modest\n")
        f.write("and may vary with different data distributions or\n")
        f.write("hyperparameter configurations.\n\n")
        
        f.write("IMPORTANT: The 20,000-record API validation is a separate\n")
        f.write("full-dataset validation and is NOT the primary held-out\n")
        f.write("performance evaluation reported here.\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 80 + "\n")
    
    print("[OK] Created final_results_summary.txt")


def task9_reproducibility(metrics):
    """TASK 9: Record reproducibility information."""
    
    reproducibility_info = {
        "model_filename": "models/global/adaptive_aggregated_ocsvm.joblib",
        "scaler_convention": "Each client uses its own scaler fitted on that client's training data",
        "feature_order": [
            "voltage", "frequency", "active_power", "reactive_power",
            "load_demand", "renewable_output", "temperature", "humidity",
            "power_loss", "stability_index"
        ],
        "prediction_threshold": "0 (decision_function < 0 indicates anomaly/fault)",
        "random_state": 42,
        "evaluation_sample_count": 4449,
        "held_out_normal_samples": 3889,
        "held_out_fault_samples": 560,
        "combined_support_vectors": metrics["adaptive"]["n_support_vectors"],
        "gamma": metrics["adaptive"]["gamma"],
        "global_intercept": metrics["adaptive"]["global_intercept"]
    }
    
    with open(REPORTS_DIR / "reproducibility_info.json", 'w') as f:
        json.dump(reproducibility_info, f, indent=2)
    print("[OK] Created reproducibility_info.json")


def main():
    """Main execution function."""
    print("=" * 80)
    print("STEP 10: FINAL RESULTS AND FIGURES GENERATION")
    print("=" * 80)
    print()
    
    # Load existing metrics
    print("Loading existing metrics...")
    metrics = load_metrics()
    print("[OK] Loaded all metrics")
    print()
    
    # Execute tasks
    print("TASK 1: Final Metrics Table")
    task1_final_metrics_table(metrics)
    print()
    
    print("TASK 2: Confusion Matrix")
    task2_confusion_matrix(metrics)
    print()
    
    print("TASK 3: ROC Curve")
    task3_roc_curve(metrics)
    print()
    
    print("TASK 4: Model Comparison")
    task4_model_comparison(metrics)
    print()
    
    print("TASK 5: Aggregation Weights")
    task5_aggregation_weights(metrics)
    print()
    
    print("TASK 6: SHAP Global Importance")
    task6_shap_global_importance()
    print()
    
    print("TASK 7: Sample Explanations")
    task7_sample_explanations()
    print()
    
    print("TASK 8: Results Summary")
    task8_results_summary(metrics)
    print()
    
    print("TASK 9: Reproducibility")
    task9_reproducibility(metrics)
    print()
    
    print("=" * 80)
    print("STEP 10 COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
