"""
AF-XAI Step 5: Explainability for the adaptive aggregated OCSVM using SHAP.

This script generates feature-level explanations for predictions made by the
actual adaptive aggregated RBF OCSVM using SHAP (KernelExplainer).

The function being explained is the anomaly score:
f(X) = -adaptive_model.decision_function(X)

Higher anomaly score indicates stronger fault/anomaly indication.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler

# Import AdaptiveAggregatedOCSVM from the aggregation module
# This is required for joblib to unpickle the saved model
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "aggregation"))
from adaptive_aggregation import AdaptiveAggregatedOCSVM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_GLOBAL_DIR = PROJECT_ROOT / "models" / "global"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"

CLIENT_IDS = ["client_1", "client_2", "client_3"]
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


def validate_required_files() -> list[str]:
    """Validate that all required files exist."""
    issues = []
    
    # Check adaptive model
    model_path = MODELS_GLOBAL_DIR / "adaptive_aggregated_ocsvm.joblib"
    if not model_path.exists():
        issues.append(f"missing: {model_path.relative_to(PROJECT_ROOT).as_posix()}")
    
    # Check predictions file
    predictions_path = METRICS_DIR / "adaptive_global_predictions.csv"
    if not predictions_path.exists():
        issues.append(f"missing: {predictions_path.relative_to(PROJECT_ROOT).as_posix()}")
    
    # Check scaled training data
    for client_id in CLIENT_IDS:
        train_path = PROCESSED_DIR / "scaled" / f"{client_id}_train_scaled.csv"
        if not train_path.exists():
            issues.append(f"missing: {train_path.relative_to(PROJECT_ROOT).as_posix()}")
    
    if issues:
        raise FileNotFoundError("Step 5 validation failed:\n  " + "\n  ".join(issues))
    
    return [f"found all required files for Step 5"]


def load_adaptive_model() -> object:
    """Load the adaptive aggregated OCSVM model."""
    model_path = MODELS_GLOBAL_DIR / "adaptive_aggregated_ocsvm.joblib"
    model = joblib.load(model_path)
    print(f"Loaded adaptive model from: {model_path.relative_to(PROJECT_ROOT)}")
    print(f"Model has {model.support_vectors_.shape[0]} support vectors")
    print(f"Model gamma: {model.gamma}")
    print(f"Model n_features_in_: {model.n_features_in_}")
    return model


def load_predictions() -> pd.DataFrame:
    """Load the adaptive global predictions."""
    predictions_path = METRICS_DIR / "adaptive_global_predictions.csv"
    predictions = pd.read_csv(predictions_path)
    print(f"Loaded predictions from: {predictions_path.relative_to(PROJECT_ROOT)}")
    print(f"Total predictions: {len(predictions)}")
    return predictions


def load_background_data(n_samples: int = 100, random_state: int = 42) -> np.ndarray:
    """
    Load and combine scaled normal training data from all clients.
    Returns a representative subset for SHAP background data.
    """
    print("\nLoading background data from scaled normal training samples...")
    
    background_parts = []
    for client_id in CLIENT_IDS:
        train_path = PROCESSED_DIR / "scaled" / f"{client_id}_train_scaled.csv"
        train_df = pd.read_csv(train_path)
        
        # Filter only normal samples (fault_indicator = 0)
        normal_df = train_df[train_df["fault_indicator"] == 0]
        print(f"  {client_id}: {len(normal_df)} normal training samples")
        
        # Extract only the ML features
        features_df = normal_df[ML_FEATURES]
        background_parts.append(features_df)
    
    # Combine all normal training data
    combined_background = pd.concat(background_parts, axis=0, ignore_index=True)
    print(f"Total normal training samples combined: {len(combined_background)}")
    
    # Sample a representative subset
    if len(combined_background) > n_samples:
        combined_background = combined_background.sample(
            n=n_samples, random_state=random_state
        )
        print(f"Sampled {n_samples} background samples (random_state={random_state})")
    else:
        print(f"Using all {len(combined_background)} samples as background")
    
    return combined_background.to_numpy()


def anomaly_score_function(model: object, X: np.ndarray) -> np.ndarray:
    """
    Wrapper function for SHAP explanation.
    Returns anomaly score = -decision_function(X).
    """
    return -model.decision_function(X)


def select_samples_by_category(predictions: pd.DataFrame) -> dict:
    """
    Select representative samples from each prediction category.
    Returns a dict with category names as keys and sample indices as values.
    """
    categories = {
        "correct_fault": None,  # fault_indicator=1, predicted_fault=1
        "normal": None,  # fault_indicator=0, predicted_fault=0
        "missed_fault": None,  # fault_indicator=1, predicted_fault=0
        "false_alarm": None,  # fault_indicator=0, predicted_fault=1
    }
    
    # Select samples with strongest anomaly scores within each category
    for category, mask_func in [
        ("correct_fault", lambda df: (df["fault_indicator"] == 1) & (df["predicted_fault"] == 1)),
        ("normal", lambda df: (df["fault_indicator"] == 0) & (df["predicted_fault"] == 0)),
        ("missed_fault", lambda df: (df["fault_indicator"] == 1) & (df["predicted_fault"] == 0)),
        ("false_alarm", lambda df: (df["fault_indicator"] == 0) & (df["predicted_fault"] == 1)),
    ]:
        mask = mask_func(predictions)
        if mask.sum() > 0:
            # Select sample with highest absolute anomaly score in this category
            category_df = predictions[mask].copy()
            category_df["abs_anomaly"] = category_df["anomaly_score"].abs()
            selected_idx = category_df["abs_anomaly"].idxmax()
            categories[category] = selected_idx
            print(f"  {category}: found {mask.sum()} samples, selected index {selected_idx}")
        else:
            print(f"  {category}: NO SAMPLES AVAILABLE")
    
    return categories


def get_scaled_features_for_sample(
    predictions: pd.DataFrame,
    sample_idx: int,
    client_id: str
) -> np.ndarray:
    """
    Get the scaled feature values for a specific sample.
    This requires loading the client's test data and extracting the scaled features.
    """
    # Load the client's scaled test data
    test_path = PROCESSED_DIR / "scaled" / f"{client_id}_test_scaled.csv"
    test_df = pd.read_csv(test_path)
    
    # Find the row index in the test data that matches the sample
    sample_data = predictions.iloc[sample_idx]
    
    # Match by timestamp and node_id to find the correct row
    match_mask = (
        (test_df["timestamp"] == sample_data["timestamp"]) &
        (test_df["node_id"] == sample_data["node_id"])
    )
    
    if match_mask.sum() == 0:
        raise ValueError(f"Could not find matching sample in test data for client {client_id}")
    
    test_idx = match_mask.idxmax()
    scaled_features = test_df.loc[test_idx, ML_FEATURES].to_numpy()
    
    return scaled_features


def get_original_features_for_sample(
    predictions: pd.DataFrame,
    sample_idx: int,
    client_id: str
) -> np.ndarray:
    """
    Get the original (unscaled) feature values for a specific sample.
    """
    # Load the client's original test data
    test_path = PROCESSED_DIR / client_id / "test.csv"
    test_df = pd.read_csv(test_path)
    
    # Find the row index in the test data that matches the sample
    sample_data = predictions.iloc[sample_idx]
    
    # Match by timestamp and node_id to find the correct row
    match_mask = (
        (test_df["timestamp"] == sample_data["timestamp"]) &
        (test_df["node_id"] == sample_data["node_id"])
    )
    
    if match_mask.sum() == 0:
        raise ValueError(f"Could not find matching sample in test data for client {client_id}")
    
    test_idx = match_mask.idxmax()
    original_features = test_df.loc[test_idx, ML_FEATURES].to_numpy()
    
    return original_features


def compute_shap_explanations(
    model: object,
    background_data: np.ndarray,
    sample_features: np.ndarray,
    n_background_samples: int = 50
) -> tuple:
    """
    Compute SHAP values for a single sample using KernelExplainer.
    Returns (shap_values, base_value, explainer).
    """
    # Use a subset of background data for faster computation
    if len(background_data) > n_background_samples:
        background_subset = background_data[:n_background_samples]
    else:
        background_subset = background_data
    
    # Define the function to explain
    def predict_fn(X):
        return anomaly_score_function(model, X)
    
    # Create KernelExplainer
    explainer = shap.KernelExplainer(predict_fn, background_subset)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(sample_features.reshape(1, -1))
    base_value = explainer.expected_value if isinstance(explainer.expected_value, float) else explainer.expected_value[0]
    
    return shap_values[0], base_value, explainer


def verify_shap_additivity(
    base_value: float,
    shap_values: np.ndarray,
    anomaly_score: float,
    tolerance: float = 0.1
) -> bool:
    """
    Verify that base_value + sum(SHAP values) ≈ anomaly_score.
    Returns True if within tolerance, False otherwise.
    """
    predicted_score = base_value + shap_values.sum()
    difference = abs(predicted_score - anomaly_score)
    
    if difference > tolerance:
        print(f"    WARNING: SHAP additivity check failed!")
        print(f"      base_value + sum(SHAP) = {predicted_score:.6f}")
        print(f"      actual anomaly_score = {anomaly_score:.6f}")
        print(f"      difference = {difference:.6f}")
        return False
    else:
        print(f"    SHAP additivity check passed (diff={difference:.6f})")
        return True


def create_explanation_csv(
    explanations: list[dict],
    output_path: Path
) -> None:
    """Create a CSV file with all explanation results."""
    # Define column order
    feature_cols = ML_FEATURES
    shap_cols = [f"shap_{feat}" for feat in ML_FEATURES]
    
    # Create DataFrame
    rows = []
    for exp in explanations:
        row = {
            "sample_id": exp["sample_id"],
            "client_id": exp["client_id"],
            "timestamp": exp["timestamp"],
            "node_id": exp["node_id"],
            "region_id": exp["region_id"],
            "fault_indicator": exp["fault_indicator"],
            "predicted_fault": exp["predicted_fault"],
            "decision_score": exp["decision_score"],
            "anomaly_score": exp["anomaly_score"],
            "category": exp["category"],
        }
        
        # Add original feature values
        for feat, val in zip(ML_FEATURES, exp["original_features"]):
            row[feat] = val
        
        # Add SHAP values
        for feat, val in zip(ML_FEATURES, exp["shap_values"]):
            row[f"shap_{feat}"] = val
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved explanations to: {output_path.relative_to(PROJECT_ROOT)}")


def create_feature_importance_figure(
    explanations: list[dict],
    output_path: Path
) -> None:
    """
    Create a horizontal bar chart showing global feature importance
    based on mean absolute SHAP values across all explained samples.
    """
    # Calculate mean absolute SHAP values for each feature
    all_shap_values = np.array([exp["shap_values"] for exp in explanations])
    mean_abs_shap = np.mean(np.abs(all_shap_values), axis=0)
    
    # Sort features by importance
    sorted_indices = np.argsort(mean_abs_shap)[::-1]
    sorted_features = [ML_FEATURES[i] for i in sorted_indices]
    sorted_importance = mean_abs_shap[sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(sorted_features))
    
    ax.barh(y_pos, sorted_importance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('Global Feature Importance (Mean Absolute SHAP)')
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved global feature importance figure to: {output_path.relative_to(PROJECT_ROOT)}")


def create_sample_explanation_figure(
    explanation: dict,
    output_path: Path,
    title: str
) -> None:
    """
    Create a horizontal bar chart showing feature-level explanation for a single sample.
    Features are ordered by absolute SHAP magnitude.
    """
    shap_values = explanation["shap_values"]
    feature_names = ML_FEATURES
    
    # Sort by absolute SHAP value
    sorted_indices = np.argsort(np.abs(shap_values))[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_shap = shap_values[sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(sorted_features))
    
    colors = ['red' if val > 0 else 'blue' for val in sorted_shap]
    ax.barh(y_pos, sorted_shap, align='center', color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('SHAP Value')
    ax.set_title(title)
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Pushes toward fault (positive)'),
        Patch(facecolor='blue', label='Pushes toward normal (negative)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved sample explanation figure to: {output_path.relative_to(PROJECT_ROOT)}")


def create_xai_report(
    explanations: list[dict],
    background_sample_count: int,
    output_path: Path
) -> None:
    """Create a comprehensive JSON report about the XAI analysis."""
    
    # Calculate global feature importance
    all_shap_values = np.array([exp["shap_values"] for exp in explanations])
    mean_abs_shap = np.mean(np.abs(all_shap_values), axis=0)
    
    # Sort features by importance
    sorted_indices = np.argsort(mean_abs_shap)[::-1]
    global_importance = {
        ML_FEATURES[i]: float(mean_abs_shap[i])
        for i in sorted_indices
    }
    
    # Collect sample categories
    categories = {}
    for exp in explanations:
        cat = exp["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(exp["sample_id"])
    
    # Collect base values and anomaly scores
    base_values = [exp["base_value"] for exp in explanations]
    anomaly_scores = [exp["anomaly_score"] for exp in explanations]
    
    # SHAP additivity checks
    additivity_checks = []
    for exp in explanations:
        additivity_checks.append({
            "sample_id": exp["sample_id"],
            "category": exp["category"],
            "base_value": float(exp["base_value"]),
            "sum_shap": float(exp["shap_values"].sum()),
            "anomaly_score": float(exp["anomaly_score"]),
            "difference": float(abs(exp["base_value"] + exp["shap_values"].sum() - exp["anomaly_score"])),
            "passed": exp["additivity_passed"]
        })
    
    # Find top contributors for each sample
    top_contributors = {}
    for exp in explanations:
        shap_values = exp["shap_values"]
        sorted_indices = np.argsort(shap_values)[::-1]  # Most positive first
        sorted_negative = np.argsort(shap_values)  # Most negative first
        
        top_positive = [
            {"feature": ML_FEATURES[i], "shap_value": float(shap_values[i])}
            for i in sorted_indices[:5]
        ]
        top_negative = [
            {"feature": ML_FEATURES[i], "shap_value": float(shap_values[i])}
            for i in sorted_negative[:5]
        ]
        
        top_contributors[exp["sample_id"]] = {
            "category": exp["category"],
            "top_positive_contributors": top_positive,
            "top_negative_contributors": top_negative
        }
    
    report = {
        "explanation_method": "SHAP (KernelExplainer)",
        "model_explained": "AdaptiveAggregatedOCSVM",
        "function_explained": "anomaly_score = -decision_function(X)",
        "feature_names": ML_FEATURES,
        "background_sample_count": background_sample_count,
        "explained_sample_count": len(explanations),
        "sample_categories": categories,
        "base_values": {
            "mean": float(np.mean(base_values)),
            "std": float(np.std(base_values)),
            "min": float(np.min(base_values)),
            "max": float(np.max(base_values))
        },
        "anomaly_scores": {
            "mean": float(np.mean(anomaly_scores)),
            "std": float(np.std(anomaly_scores)),
            "min": float(np.min(anomaly_scores)),
            "max": float(np.max(anomaly_scores))
        },
        "global_feature_importance": global_importance,
        "shap_additivity_checks": additivity_checks,
        "top_contributors_by_sample": top_contributors,
        "notes": [
            "SHAP values explain the anomaly_score function (higher = more anomalous).",
            "Positive SHAP value: feature pushes prediction toward higher anomaly/fault score.",
            "Negative SHAP value: feature pushes prediction toward lower anomaly/normal score.",
            "Global feature importance is based on mean absolute SHAP value across all explained samples.",
            "This is a model-agnostic explanation using KernelExplainer.",
            "The explanation does not imply physical causality, only contribution to the model's decision."
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"Saved XAI report to: {output_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    print("AF-XAI Explainability for Adaptive Aggregated OCSVM (Step 5)")
    print("=" * 60)
    print()
    
    # Validate files
    validation_notes = validate_required_files()
    for note in validation_notes:
        print(f"  {note}")
    print()
    
    # Load model and predictions
    model = load_adaptive_model()
    predictions = load_predictions()
    
    # Load background data
    background_data = load_background_data(n_samples=100, random_state=42)
    print(f"Background data shape: {background_data.shape}")
    print()
    
    # Select samples by category
    print("Selecting samples by category...")
    categories = select_samples_by_category(predictions)
    print()
    
    # Compute explanations for each available category
    explanations = []
    for category, sample_idx in categories.items():
        if sample_idx is None:
            print(f"Skipping {category}: no samples available")
            continue
        
        print(f"\nProcessing {category} (sample index {sample_idx})...")
        
        # Get sample data
        sample_data = predictions.iloc[sample_idx]
        client_id = sample_data["client_id"]
        
        print(f"  Client: {client_id}")
        print(f"  Timestamp: {sample_data['timestamp']}")
        print(f"  Node ID: {sample_data['node_id']}")
        print(f"  Actual fault_indicator: {sample_data['fault_indicator']}")
        print(f"  Predicted fault: {sample_data['predicted_fault']}")
        print(f"  Decision score: {sample_data['decision_score']:.6f}")
        print(f"  Anomaly score: {sample_data['anomaly_score']:.6f}")
        
        # Get scaled features
        scaled_features = get_scaled_features_for_sample(predictions, sample_idx, client_id)
        print(f"  Scaled features shape: {scaled_features.shape}")
        
        # Get original features
        original_features = get_original_features_for_sample(predictions, sample_idx, client_id)
        print(f"  Original features shape: {original_features.shape}")
        
        # Compute SHAP values
        print("  Computing SHAP values...")
        shap_values, base_value, explainer = compute_shap_explanations(
            model, background_data, scaled_features, n_background_samples=50
        )
        
        print(f"  Base value: {base_value:.6f}")
        print(f"  SHAP values shape: {shap_values.shape}")
        print(f"  SHAP values: {shap_values}")
        
        # Verify additivity
        anomaly_score = sample_data['anomaly_score']
        additivity_passed = verify_shap_additivity(base_value, shap_values, anomaly_score)
        
        # Store explanation
        explanation = {
            "sample_id": int(sample_idx),
            "client_id": client_id,
            "timestamp": sample_data["timestamp"],
            "node_id": sample_data["node_id"],
            "region_id": sample_data["region_id"],
            "fault_indicator": int(sample_data["fault_indicator"]),
            "predicted_fault": int(sample_data["predicted_fault"]),
            "decision_score": float(sample_data["decision_score"]),
            "anomaly_score": float(anomaly_score),
            "category": category,
            "original_features": original_features,
            "scaled_features": scaled_features,
            "shap_values": shap_values,
            "base_value": base_value,
            "additivity_passed": additivity_passed
        }
        explanations.append(explanation)
        
        # Create individual explanation figure
        figure_title = f"{category.replace('_', ' ').title()} Explanation"
        figure_path = FIGURES_DIR / f"xai_{category}_explanation.png"
        create_sample_explanation_figure(explanation, figure_path, figure_title)
    
    print()
    print(f"Total explanations generated: {len(explanations)}")
    
    # Create output directories
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save explanations CSV
    csv_path = METRICS_DIR / "xai_sample_explanations.csv"
    create_explanation_csv(explanations, csv_path)
    
    # Create global feature importance figure
    if len(explanations) > 0:
        importance_path = FIGURES_DIR / "xai_global_feature_importance.png"
        create_feature_importance_figure(explanations, importance_path)
    
    # Create XAI report
    report_path = REPORTS_DIR / "xai_report.json"
    create_xai_report(explanations, len(background_data), report_path)
    
    # Print summary
    print()
    print("=" * 60)
    print("=== STEP 5 COMPLETE ===")
    print()
    print("XAI method: SHAP (KernelExplainer)")
    print("Model: AdaptiveAggregatedOCSVM")
    print(f"Background samples: {len(background_data)}")
    print(f"Explained samples: {len(explanations)}")
    print()
    
    for exp in explanations:
        print(f"Sample category: {exp['category']}")
        print(f"  Actual: {exp['fault_indicator']}")
        print(f"  Predicted: {exp['predicted_fault']}")
        print(f"  Anomaly score: {exp['anomaly_score']:.6f}")
        
        # Top 5 contributing features
        shap_values = exp['shap_values']
        sorted_indices = np.argsort(np.abs(shap_values))[::-1][:5]
        
        print("  Top 5 contributing features:")
        for i in sorted_indices:
            print(f"    {ML_FEATURES[i]} -> {shap_values[i]:.6f}")
        print()
    
    # Print global feature importance ranking
    if len(explanations) > 0:
        all_shap_values = np.array([exp["shap_values"] for exp in explanations])
        mean_abs_shap = np.mean(np.abs(all_shap_values), axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1]
        
        print("Global mean absolute SHAP importance ranking:")
        for rank, idx in enumerate(sorted_indices, 1):
            print(f"  {rank}. {ML_FEATURES[idx]}: {mean_abs_shap[idx]:.6f}")
    
    print()
    print("Output files:")
    print(f"  - {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"  - {report_path.relative_to(PROJECT_ROOT)}")
    if len(explanations) > 0:
        global_importance_path = FIGURES_DIR / 'xai_global_feature_importance.png'
        print(f"  - {global_importance_path.relative_to(PROJECT_ROOT)}")
        for exp in explanations:
            fig_path = FIGURES_DIR / f"xai_{exp['category']}_explanation.png"
            print(f"  - {fig_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
