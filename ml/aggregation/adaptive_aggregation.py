"""
AF-XAI Step 4: adaptive quality-weighted aggregation of local RBF One-Class SVMs.

Local sklearn OCSVM artefacts from Step 3 are not modified. Parameter averaging
is not used. The global scorer is an approximate concatenation of quality-weighted
support vectors and dual coefficients.

The quality-weight formula below is an implementation choice for this experimental
prototype. It is not claimed to be a formula specified by the IEEE paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
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
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
MODELS_LOCAL_DIR = PROJECT_ROOT / "models" / "local"
MODELS_GLOBAL_DIR = PROJECT_ROOT / "models" / "global"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"

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
OCSVM_KERNEL = "rbf"
OCSVM_GAMMA = "auto"
OCSVM_NU = 0.05
QUALITY_F1_WEIGHT = 0.5
QUALITY_AUC_WEIGHT = 0.5
GAMMA_RTOL = 1e-6
GAMMA_ATOL = 1e-8


class AdaptiveAggregatedOCSVM:
    """Approximate global RBF One-Class SVM from concatenated local representations."""

    def __init__(
        self,
        support_vectors_: np.ndarray,
        dual_coef_: np.ndarray,
        intercept_: np.ndarray,
        gamma: float,
        n_features_in_: int,
    ) -> None:
        self.support_vectors_ = np.asarray(support_vectors_, dtype=float)
        self.dual_coef_ = np.asarray(dual_coef_, dtype=float)
        if self.dual_coef_.ndim == 1:
            self.dual_coef_ = self.dual_coef_.reshape(1, -1)
        self.intercept_ = np.asarray(intercept_, dtype=float).reshape(1)
        self.gamma = float(gamma)
        self.n_features_in_ = int(n_features_in_)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected X with shape (n_samples, {self.n_features_in_}), got {X.shape}."
            )
        kernel_matrix = rbf_kernel(X, self.support_vectors_, gamma=self.gamma)
        scores = kernel_matrix @ self.dual_coef_.ravel() + float(self.intercept_[0])
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted_fault: 1 if decision_function < 0, else 0."""
        return (self.decision_function(X) < 0).astype(int)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, anomaly_scores: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    tpr = float(tp / (tp + fn)) if (tp + fn) else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "tpr": tpr,
        "fpr": fpr,
        "roc_auc": float(roc_auc_score(y_true, anomaly_scores)),
        "confusion_matrix": {
            "labels": ["Normal", "Fault"],
            "matrix": cm.tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def save_confusion_matrix_figure(cm: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Normal", "Fault"])
    ax.set_yticks([0, 1], labels=["Normal", "Fault"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def data_completeness(train_df: pd.DataFrame) -> float:
    """Fraction of non-missing ML-feature cells in the client's normal training table."""
    feature_block = train_df[ML_FEATURES]
    n_cells = int(feature_block.size)
    if n_cells == 0:
        return 0.0
    n_missing = int(feature_block.isna().sum().sum())
    return float(1.0 - (n_missing / n_cells))


def validate_required_files() -> list[str]:
    issues = []
    for client_id in CLIENT_IDS:
        paths = [
            MODELS_LOCAL_DIR / f"{client_id}_ocsvm.joblib",
            MODELS_LOCAL_DIR / f"{client_id}_scaler.joblib",
            PROCESSED_DIR / client_id / "train_normal.csv",
            PROCESSED_DIR / client_id / "test.csv",
        ]
        for path in paths:
            if not path.exists():
                issues.append(f"missing: {path.relative_to(PROJECT_ROOT).as_posix()}")
    local_metrics = METRICS_DIR / "local_model_metrics.csv"
    if not local_metrics.exists():
        issues.append(f"missing: {local_metrics.relative_to(PROJECT_ROOT).as_posix()}")
    if issues:
        raise FileNotFoundError("Step 4 validation failed:\n  " + "\n  ".join(issues))
    return [f"found all required files for {', '.join(CLIENT_IDS)}"]


def load_local_bundle(client_id: str) -> dict:
    model = joblib.load(MODELS_LOCAL_DIR / f"{client_id}_ocsvm.joblib")
    scaler = joblib.load(MODELS_LOCAL_DIR / f"{client_id}_scaler.joblib")
    train_df = pd.read_csv(PROCESSED_DIR / client_id / "train_normal.csv")
    test_df = pd.read_csv(PROCESSED_DIR / client_id / "test.csv")
    missing_features = [c for c in ML_FEATURES if c not in train_df.columns or c not in test_df.columns]
    if missing_features:
        raise ValueError(f"{client_id} is missing ML features: {missing_features}")
    if "fault_indicator" not in test_df.columns:
        raise ValueError(f"{client_id} test.csv has no fault_indicator ground-truth column.")
    gamma = float(getattr(model, "_gamma", model.gamma if isinstance(model.gamma, (int, float)) else np.nan))
    support_vectors = np.asarray(model.support_vectors_, dtype=float)
    dual_coef = np.asarray(model.dual_coef_, dtype=float)
    intercept = np.asarray(model.intercept_, dtype=float).reshape(-1)
    return {
        "client_id": client_id,
        "model": model,
        "scaler": scaler,
        "train_df": train_df,
        "test_df": test_df,
        "gamma": gamma,
        "support_vectors": support_vectors,
        "dual_coef": dual_coef,
        "intercept": float(intercept[0]),
        "n_support_vectors": int(support_vectors.shape[0]),
        "n_features": int(support_vectors.shape[1]),
        "n_train": int(len(train_df)),
    }


def compute_aggregation_weights(bundles: list[dict], local_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bundle in bundles:
        client_id = bundle["client_id"]
        metric_row = local_metrics.loc[local_metrics["client_id"] == client_id]
        if metric_row.empty:
            raise ValueError(f"No Step 3 metrics found for {client_id}.")
        f1 = float(metric_row.iloc[0]["f1_score"])
        roc_auc = float(metric_row.iloc[0]["roc_auc"])
        completeness = data_completeness(bundle["train_df"])
        quality_score = QUALITY_F1_WEIGHT * f1 + QUALITY_AUC_WEIGHT * roc_auc
        weighted_quality = quality_score * completeness
        rows.append(
            {
                "client_id": client_id,
                "F1": f1,
                "ROC_AUC": roc_auc,
                "data_completeness": completeness,
                "quality_score": quality_score,
                "weighted_quality": weighted_quality,
            }
        )
    table = pd.DataFrame(rows)
    total = float(table["weighted_quality"].sum())
    if total <= 0:
        raise ValueError("Sum of weighted_quality is not positive; cannot normalize aggregation weights.")
    table["aggregation_weight"] = table["weighted_quality"] / total
    return table


def build_global_model(bundles: list[dict], weight_table: pd.DataFrame, gamma: float) -> AdaptiveAggregatedOCSVM:
    sv_parts = []
    dual_parts = []
    intercept_acc = 0.0
    for bundle in bundles:
        client_id = bundle["client_id"]
        weight = float(weight_table.loc[weight_table["client_id"] == client_id, "aggregation_weight"].iloc[0])
        sv_parts.append(bundle["support_vectors"])
        dual = np.asarray(bundle["dual_coef"], dtype=float).ravel() * weight
        dual_parts.append(dual)
        intercept_acc += weight * bundle["intercept"]
    support_vectors = np.vstack(sv_parts)
    dual_coef = np.concatenate(dual_parts).reshape(1, -1)
    intercept = np.array([intercept_acc], dtype=float)
    n_features = int(support_vectors.shape[1])
    return AdaptiveAggregatedOCSVM(
        support_vectors_=support_vectors,
        dual_coef_=dual_coef,
        intercept_=intercept,
        gamma=gamma,
        n_features_in_=n_features,
    )


def evaluate_global_on_client(bundle: dict, global_model: AdaptiveAggregatedOCSVM) -> tuple[pd.DataFrame, dict]:
    test_df = bundle["test_df"].copy()
    y_true = test_df["fault_indicator"].to_numpy()
    if y_true.shape[0] != len(test_df):
        raise ValueError(f"{bundle['client_id']}: ground-truth length does not match test rows.")
    X_scaled = bundle["scaler"].transform(test_df[ML_FEATURES])
    decision_score = global_model.decision_function(X_scaled)
    predicted_fault = global_model.predict(X_scaled)
    anomaly_score = -decision_score
    metrics = compute_classification_metrics(y_true, predicted_fault, anomaly_score)
    predictions = pd.DataFrame(
        {
            "timestamp": test_df["timestamp"],
            "node_id": test_df["node_id"],
            "region_id": test_df["region_id"],
            "fault_indicator": y_true,
            "predicted_fault": predicted_fault,
            "decision_score": decision_score,
            "anomaly_score": anomaly_score,
            "client_id": bundle["client_id"],
        }
    )
    metrics["client_id"] = bundle["client_id"]
    metrics["test_samples"] = int(len(test_df))
    metrics["test_normal_samples"] = int((y_true == 0).sum())
    metrics["test_fault_samples"] = int((y_true == 1).sum())
    return predictions, metrics


def train_centralized_baseline(bundles: list[dict]) -> tuple[OneClassSVM, StandardScaler, pd.DataFrame, dict]:
    train_parts = []
    test_parts = []
    for bundle in bundles:
        train_part = bundle["train_df"].copy()
        test_part = bundle["test_df"].copy()
        train_part["client_id"] = bundle["client_id"]
        test_part["client_id"] = bundle["client_id"]
        train_parts.append(train_part)
        test_parts.append(test_part)
    train_all = pd.concat(train_parts, axis=0, ignore_index=True)
    test_all = pd.concat(test_parts, axis=0, ignore_index=True)
    if int((train_all["fault_indicator"] != 0).sum()) != 0:
        raise ValueError("Centralized training set contains non-normal samples.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_all[ML_FEATURES])
    model = OneClassSVM(kernel=OCSVM_KERNEL, gamma=OCSVM_GAMMA, nu=OCSVM_NU)
    model.fit(X_train)

    X_test = scaler.transform(test_all[ML_FEATURES])
    y_true = test_all["fault_indicator"].to_numpy()
    decision_score = model.decision_function(X_test)
    raw_predict = model.predict(X_test)
    predicted_fault = np.where(raw_predict == -1, 1, 0).astype(int)
    anomaly_score = -decision_score
    metrics = compute_classification_metrics(y_true, predicted_fault, anomaly_score)
    predictions = pd.DataFrame(
        {
            "timestamp": test_all["timestamp"],
            "node_id": test_all["node_id"],
            "region_id": test_all["region_id"],
            "fault_indicator": y_true,
            "predicted_fault": predicted_fault,
            "decision_score": decision_score,
            "anomaly_score": anomaly_score,
            "client_id": test_all["client_id"],
        }
    )
    metrics["train_samples"] = int(len(train_all))
    metrics["test_samples"] = int(len(test_all))
    metrics["test_normal_samples"] = int((y_true == 0).sum())
    metrics["test_fault_samples"] = int((y_true == 1).sum())
    metrics["model"] = "centralized_ocsvm"
    metrics["scaler_fitted_on"] = "combined normal training samples from all clients"
    return model, scaler, predictions, metrics


def print_validation_summary(
    notes: list[str],
    bundles: list[dict],
    gammas: list[float],
    weight_table: pd.DataFrame,
) -> None:
    print("Step 4 validation summary")
    print("-------------------------")
    for note in notes:
        print(f"  {note}")
    print(f"  local models and scalers: {len(bundles)} clients")
    print(f"  feature columns present: {ML_FEATURES}")
    print(f"  support-vector feature dimensions: {[b['n_features'] for b in bundles]}")
    print(f"  support vectors per client: {[b['n_support_vectors'] for b in bundles]}")
    print(f"  resolved RBF gamma values: {gammas}")
    print(f"  aggregation weights: {weight_table['aggregation_weight'].tolist()}")
    print(f"  aggregation weights sum: {float(weight_table['aggregation_weight'].sum()):.12f}")
    for bundle in bundles:
        n_test = len(bundle["test_df"])
        n_labels = int(bundle["test_df"]["fault_indicator"].notna().sum())
        print(f"  {bundle['client_id']} test rows={n_test}, non-null fault_indicator={n_labels}")
    print()


def save_model_comparison_figure(
    local_metrics: pd.DataFrame,
    adaptive_f1: float,
    adaptive_auc: float,
    centralized_f1: float,
    centralized_auc: float,
    out_path: Path,
) -> None:
    labels = [
        "Client 1 local OCSVM",
        "Client 2 local OCSVM",
        "Client 3 local OCSVM",
        "Adaptive aggregated OCSVM",
        "Centralized OCSVM",
    ]
    f1_values = [
        float(local_metrics.loc[local_metrics["client_id"] == "client_1", "f1_score"].iloc[0]),
        float(local_metrics.loc[local_metrics["client_id"] == "client_2", "f1_score"].iloc[0]),
        float(local_metrics.loc[local_metrics["client_id"] == "client_3", "f1_score"].iloc[0]),
        adaptive_f1,
        centralized_f1,
    ]
    auc_values = [
        float(local_metrics.loc[local_metrics["client_id"] == "client_1", "roc_auc"].iloc[0]),
        float(local_metrics.loc[local_metrics["client_id"] == "client_2", "roc_auc"].iloc[0]),
        float(local_metrics.loc[local_metrics["client_id"] == "client_3", "roc_auc"].iloc[0]),
        adaptive_auc,
        centralized_auc,
    ]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, f1_values, width, label="F1")
    ax.bar(x + width / 2, auc_values, width, label="ROC-AUC")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Measured F1 and ROC-AUC comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    print("AF-XAI adaptive quality-weighted aggregation (Step 4)")
    print("Local OCSVM models and Step 2 scalers are loaded only; they are not refit.")
    print()

    file_notes = validate_required_files()
    local_metrics = pd.read_csv(METRICS_DIR / "local_model_metrics.csv")
    bundles = [load_local_bundle(client_id) for client_id in CLIENT_IDS]

    feature_dims = {b["n_features"] for b in bundles}
    if feature_dims != {len(ML_FEATURES)}:
        raise ValueError(
            f"Support-vector feature dimensions are incompatible: {[b['n_features'] for b in bundles]}"
        )

    gammas = [b["gamma"] for b in bundles]
    if not np.allclose(gammas, gammas[0], rtol=GAMMA_RTOL, atol=GAMMA_ATOL):
        raise ValueError(f"Local RBF gamma values are not compatible: {gammas}")
    global_gamma = float(gammas[0])

    weight_table = compute_aggregation_weights(bundles, local_metrics)
    weight_sum = float(weight_table["aggregation_weight"].sum())
    if not np.isclose(weight_sum, 1.0, rtol=1e-9, atol=1e-12):
        raise ValueError(f"Aggregation weights must sum to 1, got {weight_sum}")

    print_validation_summary(file_notes, bundles, gammas, weight_table)

    global_model = build_global_model(bundles, weight_table, global_gamma)

    MODELS_GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    per_client_preds = []
    per_client_metrics = []
    for bundle in bundles:
        preds, metrics = evaluate_global_on_client(bundle, global_model)
        per_client_preds.append(preds)
        per_client_metrics.append(metrics)

    adaptive_pred = pd.concat(per_client_preds, axis=0, ignore_index=True)
    y_true_all = adaptive_pred["fault_indicator"].to_numpy()
    y_pred_all = adaptive_pred["predicted_fault"].to_numpy()
    anomaly_all = adaptive_pred["anomaly_score"].to_numpy()
    adaptive_metrics = compute_classification_metrics(y_true_all, y_pred_all, anomaly_all)
    adaptive_metrics["test_samples"] = int(len(adaptive_pred))
    adaptive_metrics["test_normal_samples"] = int((y_true_all == 0).sum())
    adaptive_metrics["test_fault_samples"] = int((y_true_all == 1).sum())
    adaptive_metrics["model"] = "adaptive_aggregated_ocsvm"
    adaptive_metrics["evaluation"] = (
        "Each client test row is scaled with that client's frozen Step 2 scaler, "
        "then scored by the aggregated RBF decision function. Combined metrics "
        "concatenate those per-client predictions."
    )

    joblib.dump(global_model, MODELS_GLOBAL_DIR / "adaptive_aggregated_ocsvm.joblib")
    adaptive_pred.to_csv(METRICS_DIR / "adaptive_global_predictions.csv", index=False)

    centralized_model, centralized_scaler, centralized_pred, centralized_metrics = train_centralized_baseline(bundles)
    joblib.dump(centralized_model, MODELS_GLOBAL_DIR / "centralized_ocsvm.joblib")
    joblib.dump(centralized_scaler, MODELS_GLOBAL_DIR / "centralized_scaler.joblib")
    centralized_pred.to_csv(METRICS_DIR / "centralized_predictions.csv", index=False)

    adaptive_payload = {
        **adaptive_metrics,
        "n_support_vectors": int(global_model.support_vectors_.shape[0]),
        "gamma": global_model.gamma,
        "global_intercept": float(global_model.intercept_[0]),
        "aggregation_weights": {
            row["client_id"]: float(row["aggregation_weight"])
            for _, row in weight_table.iterrows()
        },
        "files": {
            "model": "models/global/adaptive_aggregated_ocsvm.joblib",
            "predictions": "results/metrics/adaptive_global_predictions.csv",
        },
    }
    (METRICS_DIR / "adaptive_global_metrics.json").write_text(
        json.dumps(adaptive_payload, indent=2), encoding="utf-8"
    )
    (METRICS_DIR / "centralized_metrics.json").write_text(
        json.dumps(centralized_metrics, indent=2), encoding="utf-8"
    )

    client_metrics_rows = []
    for metrics in per_client_metrics:
        client_metrics_rows.append(
            {
                "client_id": metrics["client_id"],
                "test_samples": metrics["test_samples"],
                "test_normal_samples": metrics["test_normal_samples"],
                "test_fault_samples": metrics["test_fault_samples"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "tpr": metrics["tpr"],
                "fpr": metrics["fpr"],
                "roc_auc": metrics["roc_auc"],
            }
        )
    pd.DataFrame(client_metrics_rows).to_csv(METRICS_DIR / "global_client_metrics.csv", index=False)

    weight_records = weight_table.to_dict(orient="records")
    weights_payload = {
        "quality_weight_formula": (
            "quality_score = 0.5 * F1 + 0.5 * ROC_AUC; "
            "weighted_quality = quality_score * data_completeness; "
            "aggregation_weight_i = weighted_quality_i / sum(weighted_quality). "
            "This quality-weight formula is an implementation choice for the experimental "
            "prototype. It is not claimed to be a formula specified by the IEEE paper."
        ),
        "f1_and_roc_auc_source": "results/metrics/local_model_metrics.csv (Step 3 local evaluation)",
        "data_completeness_definition": (
            "1 - (missing ML-feature cells / total ML-feature cells) on each "
            "client's train_normal.csv. No missing values were introduced."
        ),
        "clients": weight_records,
    }
    (REPORTS_DIR / "aggregation_weights.json").write_text(
        json.dumps(weights_payload, indent=2), encoding="utf-8"
    )

    report = {
        "number_of_clients": len(CLIENT_IDS),
        "total_support_vectors": int(global_model.support_vectors_.shape[0]),
        "support_vectors_per_client": {
            b["client_id"]: b["n_support_vectors"] for b in bundles
        },
        "gamma_values": {b["client_id"]: b["gamma"] for b in bundles},
        "resolved_global_gamma": global_gamma,
        "local_intercepts": {b["client_id"]: b["intercept"] for b in bundles},
        "global_intercept": float(global_model.intercept_[0]),
        "global_intercept_note": (
            "global_intercept is a quality-weighted sum of local intercepts. "
            "This is an approximation of a jointly trained One-Class SVM intercept."
        ),
        "aggregation_weights": {
            row["client_id"]: float(row["aggregation_weight"])
            for _, row in weight_table.iterrows()
        },
        "feature_names": ML_FEATURES,
        "ocsvm_configuration": {
            "kernel": OCSVM_KERNEL,
            "gamma": OCSVM_GAMMA,
            "nu": OCSVM_NU,
            "hyperparameter_tuning": False,
        },
        "aggregation_method": (
            "Approximate RBF SVM aggregation: concatenate local support vectors and "
            "multiply each client's dual coefficients by its aggregation weight. "
            "Support-vector coordinates are not averaged. decision_function(X) = "
            "K_rbf(X, SV) @ weighted_dual_coef + weighted_intercept."
        ),
        "approximation_notes": [
            "This is an approximate RBF SVM aggregation, not an exact federated SVM solution.",
            "Local support vectors live in each client's Step 2 scaled feature space.",
            "Global scoring on a client test set uses that client's frozen scaler.",
            "The centralized OCSVM is a pooled-data baseline, not the federated model.",
            "Quality weights are a prototype implementation choice, not a paper-specified formula.",
        ],
        "sklearn_local_models_modified": False,
        "step2_scalers_refit": False,
    }
    (REPORTS_DIR / "aggregation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    save_confusion_matrix_figure(
        np.array(adaptive_metrics["confusion_matrix"]["matrix"]),
        "Adaptive aggregated OCSVM confusion matrix",
        FIGURES_DIR / "adaptive_global_confusion_matrix.png",
    )
    save_confusion_matrix_figure(
        np.array(centralized_metrics["confusion_matrix"]["matrix"]),
        "Centralized OCSVM confusion matrix",
        FIGURES_DIR / "centralized_confusion_matrix.png",
    )
    save_model_comparison_figure(
        local_metrics,
        adaptive_metrics["f1_score"],
        adaptive_metrics["roc_auc"],
        centralized_metrics["f1_score"],
        centralized_metrics["roc_auc"],
        FIGURES_DIR / "model_comparison.png",
    )

    print("=== STEP 4 COMPLETE ===")
    print()
    print("Local aggregation weights:")
    for _, row in weight_table.iterrows():
        label = row["client_id"].replace("client_", "Client ")
        print(f"{label}: {float(row['aggregation_weight']):.6f}")
    print()
    print("Adaptive Global:")
    print(f"Accuracy: {adaptive_metrics['accuracy']:.6f}")
    print(f"Precision: {adaptive_metrics['precision']:.6f}")
    print(f"Recall: {adaptive_metrics['recall']:.6f}")
    print(f"F1: {adaptive_metrics['f1_score']:.6f}")
    print(f"TPR: {adaptive_metrics['tpr']:.6f}")
    print(f"FPR: {adaptive_metrics['fpr']:.6f}")
    print(f"ROC-AUC: {adaptive_metrics['roc_auc']:.6f}")
    print()
    print("Centralized Baseline:")
    print(f"Accuracy: {centralized_metrics['accuracy']:.6f}")
    print(f"Precision: {centralized_metrics['precision']:.6f}")
    print(f"Recall: {centralized_metrics['recall']:.6f}")
    print(f"F1: {centralized_metrics['f1_score']:.6f}")
    print(f"TPR: {centralized_metrics['tpr']:.6f}")
    print(f"FPR: {centralized_metrics['fpr']:.6f}")
    print(f"ROC-AUC: {centralized_metrics['roc_auc']:.6f}")
    print()
    print("Per-client adaptive-global metrics:")
    for metrics in per_client_metrics:
        print(
            f"  {metrics['client_id']}: "
            f"Accuracy={metrics['accuracy']:.6f}  "
            f"Precision={metrics['precision']:.6f}  "
            f"Recall={metrics['recall']:.6f}  "
            f"F1={metrics['f1_score']:.6f}  "
            f"TPR={metrics['tpr']:.6f}  "
            f"FPR={metrics['fpr']:.6f}  "
            f"ROC-AUC={metrics['roc_auc']:.6f}"
        )


if __name__ == "__main__":
    main()
