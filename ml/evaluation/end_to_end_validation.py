"""
AF-XAI Step 8: End-to-end validation of the adaptive aggregated OCSVM.

This script validates the complete pipeline using:
- Real dataset: dataset/raw/power_dataset.csv
- Real trained model: models/global/adaptive_aggregated_ocsvm.joblib
- Same preprocessing/scaling convention as Step 4 (client_1 scaler)

No new models are trained. No fake data is created.
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

# Import AdaptiveAggregatedOCSVM for joblib unpickling
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from ml.aggregation.adaptive_aggregation import AdaptiveAggregatedOCSVM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATASET_PATH = PROJECT_ROOT / "dataset" / "raw" / "power_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "global" / "adaptive_aggregated_ocsvm.joblib"
SCALER_PATH = PROJECT_ROOT / "models" / "local" / "client_1_scaler.joblib"

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

RANDOM_STATE = 42


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the raw dataset and return with original row index as a column."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    df = df.reset_index()  # Keep original index as a column named 'index'
    return df


def load_model_and_scaler() -> tuple[AdaptiveAggregatedOCSVM, object]:
    """Load the adaptive aggregated model and client_1 scaler."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def prepare_features(df: pd.DataFrame, scaler) -> np.ndarray:
    """
    Extract and scale features using the same convention as Step 4.

    Uses client_1 scaler consistently (matching prediction_service.py convention).
    """
    X = df[ML_FEATURES].values
    X_scaled = scaler.transform(X)
    return X_scaled


def run_predictions(
    df: pd.DataFrame,
    model: AdaptiveAggregatedOCSVM,
    scaler,
) -> pd.DataFrame:
    """
    Run predictions on the entire dataset.

    Returns DataFrame with original metadata and prediction results.
    """
    X_scaled = prepare_features(df, scaler)
    decision_scores = model.decision_function(X_scaled)
    predicted_faults = model.predict(X_scaled)
    anomaly_scores = -decision_scores

    results = df.copy()
    results["decision_score"] = decision_scores
    results["predicted_fault"] = predicted_faults
    results["anomaly_score"] = anomaly_scores
    return results


def select_normal_sample(results: pd.DataFrame) -> dict:
    """
    CASE 1: Select a real normal sample (fault_indicator == 0).

    Choose the first valid sample after sorting by row index.
    """
    normal_samples = results[results["fault_indicator"] == 0].copy()
    if len(normal_samples) == 0:
        raise ValueError("No normal samples found in dataset")

    # Sort by index and take first
    normal_samples = normal_samples.sort_values("index").reset_index(drop=True)
    sample = normal_samples.iloc[0]

    return {
        "case": "REAL_NORMAL_SAMPLE",
        "row_index": int(sample["index"]),
        "node_id": int(sample["node_id"]),
        "timestamp": sample["timestamp"],
        "actual_label": int(sample["fault_indicator"]),
        "predicted_label": int(sample["predicted_fault"]),
        "decision_score": float(sample["decision_score"]),
        "anomaly_score": float(sample["anomaly_score"]),
        "status": "CORRECT" if sample["predicted_fault"] == 0 else "FALSE_POSITIVE",
    }


def select_fault_sample(results: pd.DataFrame) -> dict:
    """
    CASE 2: Select a real fault sample (fault_indicator == 1).

    Choose the first valid sample after sorting by row index.
    """
    fault_samples = results[results["fault_indicator"] == 1].copy()
    if len(fault_samples) == 0:
        raise ValueError("No fault samples found in dataset")

    # Sort by index and take first
    fault_samples = fault_samples.sort_values("index").reset_index(drop=True)
    sample = fault_samples.iloc[0]

    return {
        "case": "REAL_FAULT_SAMPLE",
        "row_index": int(sample["index"]),
        "node_id": int(sample["node_id"]),
        "timestamp": sample["timestamp"],
        "actual_label": int(sample["fault_indicator"]),
        "predicted_label": int(sample["predicted_fault"]),
        "decision_score": float(sample["decision_score"]),
        "anomaly_score": float(sample["anomaly_score"]),
        "status": "CORRECT" if sample["predicted_fault"] == 1 else "MISSED_FAULT",
    }


def find_false_alarm(results: pd.DataFrame) -> dict:
    """
    CASE 3: Find a false alarm (actual == 0, predicted == 1).

    Return the first example if found, otherwise report none.
    """
    false_alarms = results[
        (results["fault_indicator"] == 0) & (results["predicted_fault"] == 1)
    ].copy()

    if len(false_alarms) == 0:
        return {
            "case": "FALSE_ALARM",
            "row_index": None,
            "node_id": None,
            "timestamp": None,
            "actual_label": None,
            "predicted_label": None,
            "decision_score": None,
            "anomaly_score": None,
            "status": "No false alarm found in the evaluated set",
        }

    # Sort by index and take first
    false_alarms = false_alarms.sort_values("index").reset_index(drop=True)
    sample = false_alarms.iloc[0]

    return {
        "case": "FALSE_ALARM",
        "row_index": int(sample["index"]),
        "node_id": int(sample["node_id"]),
        "timestamp": sample["timestamp"],
        "actual_label": int(sample["fault_indicator"]),
        "predicted_label": int(sample["predicted_fault"]),
        "decision_score": float(sample["decision_score"]),
        "anomaly_score": float(sample["anomaly_score"]),
        "status": "FALSE_POSITIVE",
    }


def find_missed_fault(results: pd.DataFrame) -> dict:
    """
    CASE 4: Find a missed fault (actual == 1, predicted == 0).

    Return the first example if found, otherwise report none.
    """
    missed_faults = results[
        (results["fault_indicator"] == 1) & (results["predicted_fault"] == 0)
    ].copy()

    if len(missed_faults) == 0:
        return {
            "case": "MISSED_FAULT",
            "row_index": None,
            "node_id": None,
            "timestamp": None,
            "actual_label": None,
            "predicted_label": None,
            "decision_score": None,
            "anomaly_score": None,
            "status": "No missed fault found in the evaluated set",
        }

    # Sort by index and take first
    missed_faults = missed_faults.sort_values("index").reset_index(drop=True)
    sample = missed_faults.iloc[0]

    return {
        "case": "MISSED_FAULT",
        "row_index": int(sample["index"]),
        "node_id": int(sample["node_id"]),
        "timestamp": sample["timestamp"],
        "actual_label": int(sample["fault_indicator"]),
        "predicted_label": int(sample["predicted_fault"]),
        "decision_score": float(sample["decision_score"]),
        "anomaly_score": float(sample["anomaly_score"]),
        "status": "MISSED_FAULT",
    }


def compute_metrics(results: pd.DataFrame) -> dict:
    """Compute classification metrics on the full dataset."""
    y_true = results["fault_indicator"].values
    y_pred = results["predicted_fault"].values
    anomaly_scores = results["anomaly_score"].values

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


def save_csv_report(cases: list[dict], results: pd.DataFrame, output_path: Path) -> None:
    """Save the validation cases to CSV."""
    rows = []
    for case in cases:
        if case["row_index"] is not None:
            rows.append(case)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def save_json_report(
    cases: list[dict],
    metrics: dict,
    dataset_info: dict,
    output_path: Path,
) -> None:
    """Save detailed validation report to JSON."""
    report = {
        "dataset_info": dataset_info,
        "selected_cases": cases,
        "metrics": metrics,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def save_text_report(
    cases: list[dict],
    metrics: dict,
    dataset_info: dict,
    output_path: Path,
) -> None:
    """Save human-readable validation report to text file."""
    lines = []
    lines.append("=" * 80)
    lines.append("AF-XAI End-to-End Validation Report")
    lines.append("=" * 80)
    lines.append("")

    # Dataset information
    lines.append("1. Dataset Information")
    lines.append("-" * 40)
    lines.append(f"  Source: {dataset_info['source']}")
    lines.append(f"  Total records: {dataset_info['total_records']:,}")
    lines.append(f"  Normal samples: {dataset_info['normal_samples']:,}")
    lines.append(f"  Fault samples: {dataset_info['fault_samples']:,}")
    lines.append(f"  Features used: {', '.join(ML_FEATURES)}")
    lines.append("")

    # Model information
    lines.append("2. Model Information")
    lines.append("-" * 40)
    lines.append(f"  Model: {dataset_info['model']}")
    lines.append(f"  Scaler: {dataset_info['scaler']}")
    lines.append(f"  Support vectors: {dataset_info['n_support_vectors']:,}")
    lines.append(f"  Gamma: {dataset_info['gamma']:.6f}")
    lines.append("")

    # Selected cases
    lines.append("3. Selected Validation Cases")
    lines.append("-" * 40)
    for case in cases:
        lines.append(f"  {case['case']}:")
        if case["row_index"] is not None:
            lines.append(f"    Row index: {case['row_index']}")
            lines.append(f"    Node ID: {case['node_id']}")
            lines.append(f"    Timestamp: {case['timestamp']}")
            lines.append(f"    Actual label: {case['actual_label']}")
            lines.append(f"    Predicted label: {case['predicted_label']}")
            lines.append(f"    Decision score: {case['decision_score']:.6f}")
            lines.append(f"    Anomaly score: {case['anomaly_score']:.6f}")
            lines.append(f"    Status: {case['status']}")
        else:
            lines.append(f"    {case['status']}")
        lines.append("")

    # Metrics
    lines.append("4. Overall Metrics")
    lines.append("-" * 40)
    lines.append(f"  Accuracy: {metrics['accuracy']:.4f}")
    lines.append(f"  Precision: {metrics['precision']:.4f}")
    lines.append(f"  Recall: {metrics['recall']:.4f}")
    lines.append(f"  F1-score: {metrics['f1_score']:.4f}")
    lines.append(f"  TPR (True Positive Rate): {metrics['tpr']:.4f}")
    lines.append(f"  FPR (False Positive Rate): {metrics['fpr']:.4f}")
    if metrics['roc_auc'] is not None:
        lines.append(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    else:
        lines.append(f"  ROC-AUC: N/A")
    lines.append("")

    # Confusion matrix
    lines.append("5. Confusion Matrix")
    lines.append("-" * 40)
    cm = metrics['confusion_matrix']
    lines.append(f"                Predicted")
    lines.append(f"                Normal  Fault")
    lines.append(f"  Actual Normal   {cm['tn']:>5}  {cm['fp']:>5}")
    lines.append(f"         Fault    {cm['fn']:>5}  {cm['tp']:>5}")
    lines.append("")

    lines.append("=" * 80)
    lines.append("End of Report")
    lines.append("=" * 80)

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("AF-XAI End-to-End Validation (Step 8)")
    print("=" * 80)
    print()

    # Load dataset
    print("Loading dataset...")
    df = load_dataset(RAW_DATASET_PATH)
    print(f"  Loaded {len(df):,} records from {RAW_DATASET_PATH}")
    print()

    # Load model and scaler
    print("Loading model and scaler...")
    model, scaler = load_model_and_scaler()
    print(f"  Model: {MODEL_PATH}")
    print(f"  Scaler: {SCALER_PATH}")
    print(f"  Support vectors: {model.support_vectors_.shape[0]:,}")
    print(f"  Gamma: {model.gamma:.6f}")
    print()

    # Run predictions
    print("Running predictions on dataset...")
    results = run_predictions(df, model, scaler)
    print(f"  Predictions completed for {len(results):,} records")
    print()

    # Select validation cases
    print("Selecting validation cases...")
    normal_sample = select_normal_sample(results)
    fault_sample = select_fault_sample(results)
    false_alarm = find_false_alarm(results)
    missed_fault = find_missed_fault(results)

    cases = [normal_sample, fault_sample, false_alarm, missed_fault]
    for case in cases:
        print(f"  {case['case']}: {case['status']}")
    print()

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(results)
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-score: {metrics['f1_score']:.4f}")
    print(f"  TPR: {metrics['tpr']:.4f}")
    print(f"  FPR: {metrics['fpr']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print()

    # Prepare dataset info
    dataset_info = {
        "source": str(RAW_DATASET_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "total_records": int(len(df)),
        "normal_samples": int((df["fault_indicator"] == 0).sum()),
        "fault_samples": int((df["fault_indicator"] == 1).sum()),
        "model": str(MODEL_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "scaler": str(SCALER_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "n_support_vectors": int(model.support_vectors_.shape[0]),
        "gamma": float(model.gamma),
    }

    # Save reports
    print("Saving reports...")
    json_path = REPORTS_DIR / "end_to_end_validation.json"
    csv_path = REPORTS_DIR / "end_to_end_validation.csv"
    txt_path = REPORTS_DIR / "end_to_end_validation_report.txt"

    save_json_report(cases, metrics, dataset_info, json_path)
    save_csv_report(cases, results, csv_path)
    save_text_report(cases, metrics, dataset_info, txt_path)

    print(f"  JSON report: {json_path}")
    print(f"  CSV report: {csv_path}")
    print(f"  Text report: {txt_path}")
    print()

    # Print summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print()
    print("Selected Cases:")
    for case in cases:
        print(f"\n{case['case']}:")
        if case["row_index"] is not None:
            print(f"  Row index: {case['row_index']}")
            print(f"  Node ID: {case['node_id']}")
            print(f"  Timestamp: {case['timestamp']}")
            print(f"  Actual: {case['actual_label']}, Predicted: {case['predicted_label']}")
            print(f"  Decision score: {case['decision_score']:.6f}")
            print(f"  Anomaly score: {case['anomaly_score']:.6f}")
            print(f"  Status: {case['status']}")
        else:
            print(f"  {case['status']}")

    print()
    print("Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"                Predicted")
    print(f"                Normal  Fault")
    print(f"  Actual Normal   {cm['tn']:>5}  {cm['fp']:>5}")
    print(f"         Fault    {cm['fn']:>5}  {cm['tp']:>5}")

    print()
    print("Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-score: {metrics['f1_score']:.4f}")
    print(f"  TPR: {metrics['tpr']:.4f}")
    print(f"  FPR: {metrics['fpr']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

    print()
    print("Verification:")
    print(f"  Dataset source: {RAW_DATASET_PATH}")
    print(f"  Model source: {MODEL_PATH}")
    print(f"  Scaler source: {SCALER_PATH}")
    print(f"  Total records evaluated: {len(results):,}")
    print(f"  Normal samples: {dataset_info['normal_samples']:,}")
    print(f"  Fault samples: {dataset_info['fault_samples']:,}")
    print()
    print("Validation complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
