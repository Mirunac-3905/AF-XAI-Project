"""
AF-XAI Step 9: Evaluation Consistency Audit

This script audits the consistency between:
- Step 4: Adaptive aggregation evaluation on held-out test sets (4,449 rows)
- Step 8: End-to-end validation on full dataset (20,000 rows)

Key differences to verify:
1. Row indices used in evaluation
2. Feature order
3. Scaler convention (per-client vs uniform)
4. Model parameters
5. Decision function and prediction threshold
6. Label mapping
7. Metric calculations

No models are retrained. No artifacts are modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from ml.aggregation.adaptive_aggregation import AdaptiveAggregatedOCSVM

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Paths
RAW_DATASET_PATH = PROJECT_ROOT / "dataset" / "raw" / "power_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "global" / "adaptive_aggregated_ocsvm.joblib"
SCALER_1_PATH = PROJECT_ROOT / "models" / "local" / "client_1_scaler.joblib"
SCALER_2_PATH = PROJECT_ROOT / "models" / "local" / "client_2_scaler.joblib"
SCALER_3_PATH = PROJECT_ROOT / "models" / "local" / "client_3_scaler.joblib"

# Step 4 artifacts
STEP4_METRICS_PATH = PROJECT_ROOT / "results" / "metrics" / "adaptive_global_metrics.json"
STEP4_PREDICTIONS_PATH = PROJECT_ROOT / "results" / "metrics" / "adaptive_global_predictions.csv"

# Step 8 artifacts
STEP8_JSON_PATH = PROJECT_ROOT / "results" / "reports" / "end_to_end_validation.json"
STEP8_CSV_PATH = PROJECT_ROOT / "results" / "reports" / "end_to_end_validation.csv"

# Client test data (Step 4 held-out sets)
CLIENT_1_TEST_PATH = PROJECT_ROOT / "dataset" / "processed" / "client_1" / "test.csv"
CLIENT_2_TEST_PATH = PROJECT_ROOT / "dataset" / "processed" / "client_2" / "test.csv"
CLIENT_3_TEST_PATH = PROJECT_ROOT / "dataset" / "processed" / "client_3" / "test.csv"

# Output
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ML_FEATURES = [
    "voltage",
    "frequency",
    "active_power",
    "reactive_power",
    "load_demand",
    "renewable_output",
    "temperature",
    "humidity",
    "power_loss",
    "stability_index",
]

CLIENT_IDS = ["client_1", "client_2", "client_3"]


def load_step4_artifacts() -> tuple[dict, pd.DataFrame]:
    """Load Step 4 metrics and predictions."""
    with open(STEP4_METRICS_PATH, 'r') as f:
        step4_metrics = json.load(f)
    
    step4_predictions = pd.read_csv(STEP4_PREDICTIONS_PATH)
    return step4_metrics, step4_predictions


def load_step8_artifacts() -> tuple[dict, pd.DataFrame]:
    """Load Step 8 validation results."""
    with open(STEP8_JSON_PATH, 'r') as f:
        step8_json = json.load(f)
    
    step8_csv = pd.read_csv(STEP8_CSV_PATH)
    return step8_json, step8_csv


def load_model_and_scalers() -> tuple[AdaptiveAggregatedOCSVM, dict]:
    """Load the adaptive model and all client scalers."""
    model = joblib.load(MODEL_PATH)
    scalers = {
        "client_1": joblib.load(SCALER_1_PATH),
        "client_2": joblib.load(SCALER_2_PATH),
        "client_3": joblib.load(SCALER_3_PATH),
    }
    return model, scalers


def load_client_test_data() -> dict[str, pd.DataFrame]:
    """Load the held-out test sets from each client."""
    return {
        "client_1": pd.read_csv(CLIENT_1_TEST_PATH),
        "client_2": pd.read_csv(CLIENT_2_TEST_PATH),
        "client_3": pd.read_csv(CLIENT_3_TEST_PATH),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, anomaly_scores: np.ndarray) -> dict:
    """Compute classification metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, anomaly_scores))
    except ValueError:
        roc_auc = None

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "tpr": tpr,
        "fpr": fpr,
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "labels": ["Normal", "Fault"],
            "matrix": cm.tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def recompute_step4_predictions(
    model: AdaptiveAggregatedOCSVM,
    scalers: dict,
    client_test_data: dict,
) -> pd.DataFrame:
    """
    Recompute predictions on Step 4 test rows using per-client scaling.
    
    This matches the Step 4 evaluation convention where each client's test data
    is scaled with that client's own scaler.
    """
    all_predictions = []
    
    for client_id in CLIENT_IDS:
        test_df = client_test_data[client_id].copy()
        scaler = scalers[client_id]
        
        # Scale features with this client's scaler
        X_scaled = scaler.transform(test_df[ML_FEATURES])
        
        # Get predictions
        decision_scores = model.decision_function(X_scaled)
        predicted_faults = model.predict(X_scaled)
        anomaly_scores = -decision_scores
        
        predictions = pd.DataFrame({
            "timestamp": test_df["timestamp"],
            "node_id": test_df["node_id"],
            "region_id": test_df["region_id"],
            "fault_indicator": test_df["fault_indicator"],
            "predicted_fault": predicted_faults,
            "decision_score": decision_scores,
            "anomaly_score": anomaly_scores,
            "client_id": client_id,
        })
        
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions, axis=0, ignore_index=True)


def compute_step4_metrics_recomputed(predictions: pd.DataFrame) -> dict:
    """Compute metrics from recomputed Step 4 predictions."""
    y_true = predictions["fault_indicator"].values
    y_pred = predictions["predicted_fault"].values
    anomaly_scores = predictions["anomaly_score"].values
    
    metrics = compute_metrics(y_true, y_pred, anomaly_scores)
    metrics["test_samples"] = int(len(predictions))
    metrics["test_normal_samples"] = int((y_true == 0).sum())
    metrics["test_fault_samples"] = int((y_true == 1).sum())
    
    return metrics


def compute_step8_metrics_recomputed(
    model: AdaptiveAggregatedOCSVM,
    scaler,
    raw_dataset: pd.DataFrame,
) -> dict:
    """
    Recompute predictions on all 20,000 rows using client_1 scaler.
    
    This matches the Step 8 validation convention where all data is scaled
    with client_1's scaler (matching the API convention).
    """
    # Scale all features with client_1 scaler
    X_scaled = scaler.transform(raw_dataset[ML_FEATURES])
    
    # Get predictions
    decision_scores = model.decision_function(X_scaled)
    predicted_faults = model.predict(X_scaled)
    anomaly_scores = -decision_scores
    
    y_true = raw_dataset["fault_indicator"].values
    
    metrics = compute_metrics(y_true, predicted_faults, anomaly_scores)
    metrics["test_samples"] = int(len(raw_dataset))
    metrics["test_normal_samples"] = int((y_true == 0).sum())
    metrics["test_fault_samples"] = int((y_true == 1).sum())
    
    return metrics


def compare_metrics(saved: dict, recomputed: dict, label: str) -> dict:
    """Compare saved and recomputed metrics."""
    comparison = {}
    
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "tpr", "fpr"]
    
    for key in metric_keys:
        saved_val = saved.get(key)
        recomputed_val = recomputed.get(key)
        
        if saved_val is not None and recomputed_val is not None:
            diff = abs(saved_val - recomputed_val)
            match = diff < 1e-6
            comparison[key] = {
                "saved": saved_val,
                "recomputed": recomputed_val,
                "difference": diff,
                "match": match,
            }
    
    # Compare confusion matrix
    saved_cm = saved.get("confusion_matrix", {})
    recomputed_cm = recomputed.get("confusion_matrix", {})
    
    cm_match = (
        saved_cm.get("tn") == recomputed_cm.get("tn") and
        saved_cm.get("fp") == recomputed_cm.get("fp") and
        saved_cm.get("fn") == recomputed_cm.get("fn") and
        saved_cm.get("tp") == recomputed_cm.get("tp")
    )
    
    comparison["confusion_matrix"] = {
        "saved": saved_cm,
        "recomputed": recomputed_cm,
        "match": cm_match,
    }
    
    return comparison


def save_audit_report(
    step4_saved: dict,
    step4_recomputed: dict,
    step4_comparison: dict,
    step8_saved: dict,
    step8_recomputed: dict,
    step8_comparison: dict,
    output_json: Path,
    output_txt: Path,
) -> None:
    """Save the audit report to JSON and text files."""
    
    report = {
        "audit_summary": {
            "purpose": "Verify consistency between Step 4 (held-out test evaluation) and Step 8 (full dataset validation)",
            "key_difference": "Step 4 uses per-client scaling; Step 8 uses uniform client_1 scaling (API convention)",
        },
        "step4_evaluation": {
            "description": "Held-out test sets from 3 clients (4,449 rows total)",
            "scaling_convention": "Per-client scaling (each client's test data scaled with that client's own scaler)",
            "saved_metrics": step4_saved,
            "recomputed_metrics": step4_recomputed,
            "comparison": step4_comparison,
        },
        "step8_validation": {
            "description": "Full dataset (20,000 rows including training data)",
            "scaling_convention": "Uniform client_1 scaling (matches API prediction_service.py convention)",
            "saved_metrics": step8_saved,
            "recomputed_metrics": step8_recomputed,
            "comparison": step8_comparison,
        },
        "conclusions": {
            "step4_consistency": "PASS" if step4_comparison["confusion_matrix"]["match"] else "DISCREPANCY",
            "step8_consistency": "PASS" if step8_comparison["confusion_matrix"]["match"] else "DISCREPANCY",
        },
    }
    
    # Save JSON
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    
    # Save text
    lines = []
    lines.append("=" * 80)
    lines.append("AF-XAI Evaluation Consistency Audit Report")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append("AUDIT PURPOSE")
    lines.append("-" * 40)
    lines.append("Verify consistency between:")
    lines.append("  - Step 4: Adaptive aggregation evaluation on held-out test sets")
    lines.append("  - Step 8: End-to-end validation on full dataset")
    lines.append("")
    
    lines.append("KEY DIFFERENCE")
    lines.append("-" * 40)
    lines.append("  Step 4: Per-client scaling (each client's test data scaled with that client's own scaler)")
    lines.append("  Step 8: Uniform client_1 scaling (matches API prediction_service.py convention)")
    lines.append("")
    
    lines.append("STEP 4: HELD-OUT TEST EVALUATION")
    lines.append("-" * 40)
    lines.append(f"  Description: Held-out test sets from 3 clients")
    lines.append(f"  Total rows: {step4_saved.get('test_samples', 'N/A')}")
    lines.append(f"  Normal samples: {step4_saved.get('test_normal_samples', 'N/A')}")
    lines.append(f"  Fault samples: {step4_saved.get('test_fault_samples', 'N/A')}")
    lines.append(f"  Scaling: Per-client scaling")
    lines.append("")
    
    lines.append("  Saved Metrics (Step 4):")
    lines.append(f"    Accuracy: {step4_saved.get('accuracy', 'N/A'):.6f}")
    lines.append(f"    Precision: {step4_saved.get('precision', 'N/A'):.6f}")
    lines.append(f"    Recall: {step4_saved.get('recall', 'N/A'):.6f}")
    lines.append(f"    F1-score: {step4_saved.get('f1_score', 'N/A'):.6f}")
    lines.append(f"    TPR: {step4_saved.get('tpr', 'N/A'):.6f}")
    lines.append(f"    FPR: {step4_saved.get('fpr', 'N/A'):.6f}")
    lines.append(f"    ROC-AUC: {step4_saved.get('roc_auc', 'N/A'):.6f}")
    lines.append("")
    
    lines.append("  Confusion Matrix (Saved):")
    cm_saved = step4_saved.get("confusion_matrix", {})
    lines.append(f"                Predicted")
    lines.append(f"                Normal  Fault")
    lines.append(f"  Actual Normal   {cm_saved.get('tn', 'N/A'):>5}  {cm_saved.get('fp', 'N/A'):>5}")
    lines.append(f"         Fault    {cm_saved.get('fn', 'N/A'):>5}  {cm_saved.get('tp', 'N/A'):>5}")
    lines.append("")
    
    lines.append("  Recomputed Metrics (Audit):")
    lines.append(f"    Accuracy: {step4_recomputed.get('accuracy', 'N/A'):.6f}")
    lines.append(f"    Precision: {step4_recomputed.get('precision', 'N/A'):.6f}")
    lines.append(f"    Recall: {step4_recomputed.get('recall', 'N/A'):.6f}")
    lines.append(f"    F1-score: {step4_recomputed.get('f1_score', 'N/A'):.6f}")
    lines.append(f"    TPR: {step4_recomputed.get('tpr', 'N/A'):.6f}")
    lines.append(f"    FPR: {step4_recomputed.get('fpr', 'N/A'):.6f}")
    lines.append(f"    ROC-AUC: {step4_recomputed.get('roc_auc', 'N/A'):.6f}")
    lines.append("")
    
    lines.append("  Confusion Matrix (Recomputed):")
    cm_recomputed = step4_recomputed.get("confusion_matrix", {})
    lines.append(f"                Predicted")
    lines.append(f"                Normal  Fault")
    lines.append(f"  Actual Normal   {cm_recomputed.get('tn', 'N/A'):>5}  {cm_recomputed.get('fp', 'N/A'):>5}")
    lines.append(f"         Fault    {cm_recomputed.get('fn', 'N/A'):>5}  {cm_recomputed.get('tp', 'N/A'):>5}")
    lines.append("")
    
    lines.append("  Comparison:")
    lines.append(f"    Confusion Matrix Match: {step4_comparison['confusion_matrix']['match']}")
    for key, comp in step4_comparison.items():
        if key != "confusion_matrix":
            lines.append(f"    {key}: {comp['saved']:.6f} vs {comp['recomputed']:.6f} (diff: {comp['difference']:.6f}) - {'MATCH' if comp['match'] else 'MISMATCH'}")
    lines.append("")
    
    lines.append("STEP 8: FULL DATASET VALIDATION")
    lines.append("-" * 40)
    lines.append(f"  Description: Full dataset (includes training data)")
    lines.append(f"  Total rows: {step8_saved.get('total_records', 'N/A')}")
    lines.append(f"  Normal samples: {step8_saved.get('normal_samples', 'N/A')}")
    lines.append(f"  Fault samples: {step8_saved.get('fault_samples', 'N/A')}")
    lines.append(f"  Scaling: Uniform client_1 scaling (API convention)")
    lines.append("")
    
    lines.append("  Saved Metrics (Step 8):")
    lines.append(f"    Accuracy: {step8_saved.get('accuracy', 'N/A'):.6f}")
    lines.append(f"    Precision: {step8_saved.get('precision', 'N/A'):.6f}")
    lines.append(f"    Recall: {step8_saved.get('recall', 'N/A'):.6f}")
    lines.append(f"    F1-score: {step8_saved.get('f1_score', 'N/A'):.6f}")
    lines.append(f"    TPR: {step8_saved.get('tpr', 'N/A'):.6f}")
    lines.append(f"  FPR: {step8_saved.get('fpr', 'N/A'):.6f}")
    lines.append(f"    ROC-AUC: {step8_saved.get('roc_auc', 'N/A'):.6f}")
    lines.append("")
    
    lines.append("  Confusion Matrix (Saved):")
    cm8_saved = step8_saved.get("confusion_matrix", {})
    lines.append(f"                Predicted")
    lines.append(f"                Normal  Fault")
    lines.append(f"  Actual Normal   {cm8_saved.get('tn', 'N/A'):>5}  {cm8_saved.get('fp', 'N/A'):>5}")
    lines.append(f"         Fault    {cm8_saved.get('fn', 'N/A'):>5}  {cm8_saved.get('tp', 'N/A'):>5}")
    lines.append("")
    
    lines.append("  Recomputed Metrics (Audit):")
    lines.append(f"    Accuracy: {step8_recomputed.get('accuracy', 'N/A'):.6f}")
    lines.append(f"    Precision: {step8_recomputed.get('precision', 'N/A'):.6f}")
    lines.append(f"    Recall: {step8_recomputed.get('recall', 'N/A'):.6f}")
    lines.append(f"    F1-score: {step8_recomputed.get('f1_score', 'N/A'):.6f}")
    lines.append(f"    TPR: {step8_recomputed.get('tpr', 'N/A'):.6f}")
    lines.append(f"    FPR: {step8_recomputed.get('fpr', 'N/A'):.6f}")
    lines.append(f"    ROC-AUC: {step8_recomputed.get('roc_auc', 'N/A'):.6f}")
    lines.append("")
    
    lines.append("  Confusion Matrix (Recomputed):")
    cm8_recomputed = step8_recomputed.get("confusion_matrix", {})
    lines.append(f"                Predicted")
    lines.append(f"                Normal  Fault")
    lines.append(f"  Actual Normal   {cm8_recomputed.get('tn', 'N/A'):>5}  {cm8_recomputed.get('fp', 'N/A'):>5}")
    lines.append(f"         Fault    {cm8_recomputed.get('fn', 'N/A'):>5}  {cm8_recomputed.get('tp', 'N/A'):>5}")
    lines.append("")
    
    lines.append("  Comparison:")
    lines.append(f"    Confusion Matrix Match: {step8_comparison['confusion_matrix']['match']}")
    for key, comp in step8_comparison.items():
        if key != "confusion_matrix":
            lines.append(f"    {key}: {comp['saved']:.6f} vs {comp['recomputed']:.6f} (diff: {comp['difference']:.6f}) - {'MATCH' if comp['match'] else 'MISMATCH'}")
    lines.append("")
    
    lines.append("CONCLUSIONS")
    lines.append("-" * 40)
    lines.append(f"  Step 4 Consistency: {report['conclusions']['step4_consistency']}")
    lines.append(f"  Step 8 Consistency: {report['conclusions']['step8_consistency']}")
    lines.append("")
    
    lines.append("IMPORTANT NOTES")
    lines.append("-" * 40)
    lines.append("  - Step 4 evaluates on held-out test sets ONLY (no training data)")
    lines.append("  - Step 8 evaluates on the FULL dataset (includes training data)")
    lines.append("  - These are different evaluation scopes and should NOT have identical metrics")
    lines.append("  - The key audit is whether Step 4 recomputed metrics match saved Step 4 metrics")
    lines.append("  - Step 8 uses a different scaling convention (API: client_1 scaler for all)")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("End of Audit Report")
    lines.append("=" * 80)
    
    output_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("AF-XAI Evaluation Consistency Audit (Step 9)")
    print("=" * 80)
    print()
    
    # Load artifacts
    print("Loading artifacts...")
    step4_metrics, step4_predictions = load_step4_artifacts()
    step8_json, step8_csv = load_step8_artifacts()
    model, scalers = load_model_and_scalers()
    client_test_data = load_client_test_data()
    raw_dataset = pd.read_csv(RAW_DATASET_PATH)
    print("  All artifacts loaded")
    print()
    
    # Verify model parameters
    print("Model parameters:")
    print(f"  Support vectors: {model.support_vectors_.shape[0]}")
    print(f"  Gamma: {model.gamma}")
    print(f"  Intercept: {model.intercept_[0]:.6f}")
    print()
    
    # Recompute Step 4 predictions with per-client scaling
    print("Recomputing Step 4 predictions (per-client scaling)...")
    step4_recomputed_preds = recompute_step4_predictions(model, scalers, client_test_data)
    step4_recomputed_metrics = compute_step4_metrics_recomputed(step4_recomputed_preds)
    print(f"  Test rows: {step4_recomputed_metrics['test_samples']}")
    print(f"  Normal: {step4_recomputed_metrics['test_normal_samples']}, Fault: {step4_recomputed_metrics['test_fault_samples']}")
    print()
    
    # Compare Step 4 metrics
    print("Comparing Step 4 metrics...")
    step4_comparison = compare_metrics(step4_metrics, step4_recomputed_metrics, "Step 4")
    print(f"  Confusion matrix match: {step4_comparison['confusion_matrix']['match']}")
    for key, comp in step4_comparison.items():
        if key != "confusion_matrix":
            status = "MATCH" if comp['match'] else "MISMATCH"
            print(f"  {key}: {status} (diff: {comp['difference']:.6f})")
    print()
    
    # Recompute Step 8 metrics with client_1 scaling
    print("Recomputing Step 8 metrics (uniform client_1 scaling)...")
    step8_recomputed_metrics = compute_step8_metrics_recomputed(model, scalers["client_1"], raw_dataset)
    print(f"  Total rows: {step8_recomputed_metrics['test_samples']}")
    print(f"  Normal: {step8_recomputed_metrics['test_normal_samples']}, Fault: {step8_recomputed_metrics['test_fault_samples']}")
    print()
    
    # Compare Step 8 metrics
    print("Comparing Step 8 metrics...")
    step8_comparison = compare_metrics(step8_json["metrics"], step8_recomputed_metrics, "Step 8")
    print(f"  Confusion matrix match: {step8_comparison['confusion_matrix']['match']}")
    for key, comp in step8_comparison.items():
        if key != "confusion_matrix":
            status = "MATCH" if comp['match'] else "MISMATCH"
            print(f"  {key}: {status} (diff: {comp['difference']:.6f})")
    print()
    
    # Save audit report
    print("Saving audit report...")
    json_path = REPORTS_DIR / "evaluation_consistency_audit.json"
    txt_path = REPORTS_DIR / "evaluation_consistency_audit.txt"
    
    save_audit_report(
        step4_metrics,
        step4_recomputed_metrics,
        step4_comparison,
        step8_json["metrics"],
        step8_recomputed_metrics,
        step8_comparison,
        json_path,
        txt_path,
    )
    print(f"  JSON: {json_path}")
    print(f"  Text: {txt_path}")
    print()
    
    # Print summary
    print("=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)
    print()
    
    print("STEP 4: HELD-OUT TEST EVALUATION (4,449 rows)")
    print("-" * 80)
    print(f"  Scaling: Per-client (each client's own scaler)")
    print(f"  Test rows: {step4_metrics['test_samples']}")
    print(f"  Normal: {step4_metrics['test_normal_samples']}, Fault: {step4_metrics['test_fault_samples']}")
    print()
    print("  Confusion Matrix (Saved):")
    cm4 = step4_metrics["confusion_matrix"]
    print(f"                Predicted")
    print(f"                Normal  Fault")
    print(f"  Actual Normal   {cm4['tn']:>5}  {cm4['fp']:>5}")
    print(f"         Fault    {cm4['fn']:>5}  {cm4['tp']:>5}")
    print()
    print("  Confusion Matrix (Recomputed):")
    cm4r = step4_recomputed_metrics["confusion_matrix"]
    print(f"                Predicted")
    print(f"                Normal  Fault")
    print(f"  Actual Normal   {cm4r['tn']:>5}  {cm4r['fp']:>5}")
    print(f"         Fault    {cm4r['fn']:>5}  {cm4r['tp']:>5}")
    print()
    print(f"  Consistency: {'PASS' if step4_comparison['confusion_matrix']['match'] else 'DISCREPANCY'}")
    print()
    
    print("STEP 8: FULL DATASET VALIDATION (20,000 rows)")
    print("-" * 80)
    print(f"  Scaling: Uniform client_1 (API convention)")
    print(f"  Total rows: {step8_json['dataset_info']['total_records']}")
    print(f"  Normal: {step8_json['dataset_info']['normal_samples']}, Fault: {step8_json['dataset_info']['fault_samples']}")
    print(f"  NOTE: Includes training data (not a held-out test evaluation)")
    print()
    print("  Confusion Matrix (Saved):")
    cm8 = step8_json["metrics"]["confusion_matrix"]
    print(f"                Predicted")
    print(f"                Normal  Fault")
    print(f"  Actual Normal   {cm8['tn']:>5}  {cm8['fp']:>5}")
    print(f"         Fault    {cm8['fn']:>5}  {cm8['tp']:>5}")
    print()
    print("  Confusion Matrix (Recomputed):")
    cm8r = step8_recomputed_metrics["confusion_matrix"]
    print(f"                Predicted")
    print(f"                Normal  Fault")
    print(f"  Actual Normal   {cm8r['tn']:>5}  {cm8r['fp']:>5}")
    print(f"         Fault    {cm8r['fn']:>5}  {cm8r['tp']:>5}")
    print()
    print(f"  Consistency: {'PASS' if step8_comparison['confusion_matrix']['match'] else 'DISCREPANCY'}")
    print()
    
    print("KEY FINDINGS")
    print("-" * 80)
    print("  - Step 4 and Step 8 use DIFFERENT row sets:")
    print("    * Step 4: 4,449 held-out test rows (no training data)")
    print("    * Step 8: 20,000 total rows (includes training data)")
    print("  - Step 4 and Step 8 use DIFFERENT scaling conventions:")
    print("    * Step 4: Per-client scaling (each client's own scaler)")
    print("    * Step 8: Uniform client_1 scaling (API convention)")
    print("  - Therefore, metrics SHOULD differ between Step 4 and Step 8")
    print("  - The audit verifies that recomputed metrics match saved metrics")
    print("    for each evaluation independently")
    print()
    
    if not step4_comparison['confusion_matrix']['match']:
        print("WARNING: Step 4 recomputed metrics do NOT match saved metrics")
        print("This indicates a potential discrepancy in the evaluation pipeline.")
    else:
        print("Step 4: Recomputed metrics match saved metrics (PASS)")
    
    if not step8_comparison['confusion_matrix']['match']:
        print("WARNING: Step 8 recomputed metrics do NOT match saved metrics")
        print("This indicates a potential discrepancy in the validation pipeline.")
    else:
        print("Step 8: Recomputed metrics match saved metrics (PASS)")
    
    print()
    print("Audit complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
