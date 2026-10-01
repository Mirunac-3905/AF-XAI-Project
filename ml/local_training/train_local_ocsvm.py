"""
AF-XAI Step 3: independent local One-Class SVM training and evaluation.

Each simulated substation trains only on its own normal samples.
No global model, aggregation, SHAP, LIME, Flask, or React is performed here.
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
from sklearn.svm import OneClassSVM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
MODELS_LOCAL_DIR = PROJECT_ROOT / "models" / "local"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"

CLIENT_IDS = ["client_1", "client_2", "client_3"]

# Identical to Step 2. fault_indicator is evaluation ground truth, never an input.
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
RANDOM_STATE = None  # sklearn OneClassSVM with this kernel does not use a random_state


def verify_training_is_normal(train_df: pd.DataFrame, client_id: str) -> None:
    n_faults = int((train_df["fault_indicator"] != 0).sum())
    if n_faults != 0:
        raise ValueError(
            f"{client_id}: train_normal.csv contains {n_faults} non-normal rows. "
            "OCSVM must be trained only on fault_indicator == 0."
        )


def load_client_tables(client_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = PROCESSED_DIR / client_id / "train_normal.csv"
    test_path = PROCESSED_DIR / client_id / "test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    verify_training_is_normal(train_df, client_id)
    return train_df, test_df


def load_step2_scaler(client_id: str):
    scaler_path = MODELS_LOCAL_DIR / f"{client_id}_scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Step 2 scaler not found for {client_id}: {scaler_path}. Do not refit."
        )
    return joblib.load(scaler_path)


def scale_features(df: pd.DataFrame, scaler) -> np.ndarray:
    """Transform the 10 ML features with the frozen Step 2 scaler."""
    return scaler.transform(df[ML_FEATURES])


def train_local_ocsvm(X_train_scaled: np.ndarray) -> OneClassSVM:
    model = OneClassSVM(kernel=OCSVM_KERNEL, gamma=OCSVM_GAMMA, nu=OCSVM_NU)
    model.fit(X_train_scaled)
    return model


def ocsvm_outputs_to_fault_labels(raw_predict: np.ndarray) -> np.ndarray:
    """Map sklearn OCSVM labels: +1 normal → 0, -1 anomaly → 1 (fault)."""
    predicted_fault = np.where(raw_predict == -1, 1, 0).astype(int)
    return predicted_fault


def anomaly_scores_from_decision(decision_scores: np.ndarray) -> np.ndarray:
    """
    decision_function is larger for inliers (normal).
    Negate so a larger score corresponds to the fault/positive class for ROC-AUC.
    """
    return -decision_scores


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, anomaly_scores: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = float(tp / (tp + fn)) if (tp + fn) else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    roc_auc = float(roc_auc_score(y_true, anomaly_scores))
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
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def save_confusion_matrix_figure(cm: np.ndarray, client_id: str, out_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Normal", "Fault"])
    ax.set_yticks([0, 1], labels=["Normal", "Fault"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{client_id} confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def print_pretrain_stats(client_id: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    test_normal = int((test_df["fault_indicator"] == 0).sum())
    test_fault = int((test_df["fault_indicator"] == 1).sum())
    print(f"Client {client_id}")
    print("--------")
    print(f"Training samples: {len(train_df)}")
    print(f"Fault samples in training: {int((train_df['fault_indicator'] != 0).sum())}")
    print(f"Test samples: {len(test_df)}")
    print(f"Test normal: {test_normal}")
    print(f"Test fault: {test_fault}")
    print()


def print_config_and_metrics(metrics: dict) -> None:
    print("Model configuration:")
    print(f"kernel = {OCSVM_KERNEL}")
    print(f"gamma = {OCSVM_GAMMA}")
    print(f"nu = {OCSVM_NU}")
    print()
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"F1: {metrics['f1_score']:.6f}")
    print(f"TPR: {metrics['tpr']:.6f}")
    print(f"FPR: {metrics['fpr']:.6f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.6f}")
    print()


def run_client(client_id: str) -> dict:
    train_df, test_df = load_client_tables(client_id)
    print_pretrain_stats(client_id, train_df, test_df)

    scaler = load_step2_scaler(client_id)
    X_train = scale_features(train_df, scaler)
    X_test = scale_features(test_df, scaler)

    model = train_local_ocsvm(X_train)
    model_path = MODELS_LOCAL_DIR / f"{client_id}_ocsvm.joblib"
    joblib.dump(model, model_path)

    raw_predict = model.predict(X_test)
    decision_scores = model.decision_function(X_test)
    predicted_fault = ocsvm_outputs_to_fault_labels(raw_predict)
    y_true = test_df["fault_indicator"].to_numpy()
    scores = anomaly_scores_from_decision(decision_scores)
    metrics = compute_metrics(y_true, predicted_fault, scores)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame(
        {
            "timestamp": test_df["timestamp"],
            "node_id": test_df["node_id"],
            "region_id": test_df["region_id"],
            "fault_indicator": y_true,
            "predicted_fault": predicted_fault,
            "decision_score": decision_scores,
        }
    )
    pred_path = METRICS_DIR / f"{client_id}_predictions.csv"
    predictions.to_csv(pred_path, index=False)

    cm = np.array(metrics["confusion_matrix"]["matrix"], dtype=int)
    fig_path = FIGURES_DIR / f"{client_id}_confusion_matrix.png"
    save_confusion_matrix_figure(cm, client_id, fig_path)

    client_metrics = {
        "client_id": client_id,
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "test_normal_samples": int((test_df["fault_indicator"] == 0).sum()),
        "test_fault_samples": int((test_df["fault_indicator"] == 1).sum()),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "tpr": metrics["tpr"],
        "fpr": metrics["fpr"],
        "roc_auc": metrics["roc_auc"],
        "confusion_matrix": metrics["confusion_matrix"],
        "model_path": str(model_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "scaler_path": f"models/local/{client_id}_scaler.joblib",
        "scaler_refitted": False,
        "ocsvm": {
            "kernel": OCSVM_KERNEL,
            "gamma": OCSVM_GAMMA,
            "nu": OCSVM_NU,
        },
        "features": ML_FEATURES,
        "positive_class": "fault_indicator == 1",
        "roc_auc_score_definition": (
            "roc_auc uses -decision_function so larger values indicate the fault class"
        ),
        "decision_score_column": (
            "predictions.csv decision_score is the raw OneClassSVM decision_function "
            "(larger = more normal)"
        ),
        "files": {
            "predictions": str(pred_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "confusion_matrix_figure": str(fig_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
    }
    json_path = METRICS_DIR / f"{client_id}_metrics.json"
    json_path.write_text(json.dumps(client_metrics, indent=2), encoding="utf-8")

    print_config_and_metrics(metrics)
    return client_metrics


def write_combined_outputs(client_rows: list[dict]) -> list[Path]:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(
        [
            {
                "client_id": row["client_id"],
                "train_samples": row["train_samples"],
                "test_samples": row["test_samples"],
                "test_normal_samples": row["test_normal_samples"],
                "test_fault_samples": row["test_fault_samples"],
                "accuracy": row["accuracy"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1_score": row["f1_score"],
                "tpr": row["tpr"],
                "fpr": row["fpr"],
                "roc_auc": row["roc_auc"],
            }
            for row in client_rows
        ]
    )
    csv_path = METRICS_DIR / "local_model_metrics.csv"
    table.to_csv(csv_path, index=False)

    config = {
        "features": ML_FEATURES,
        "kernel": OCSVM_KERNEL,
        "gamma": OCSVM_GAMMA,
        "nu": OCSVM_NU,
        "clients": CLIENT_IDS,
        "training_method": (
            "Independent One-Class SVM per client. Each model is trained only on that "
            "client's normal training samples (fault_indicator == 0). No raw data, "
            "support vectors, or parameters are shared across clients in this step."
        ),
        "scaler_method": (
            "StandardScaler artefacts from Step 2 (models/local/client_X_scaler.joblib). "
            "Scalers are loaded and applied only; they are not refit."
        ),
        "random_state": RANDOM_STATE,
        "hyperparameter_tuning": False,
        "global_model": False,
        "federated_aggregation": False,
    }
    config_path = REPORTS_DIR / "local_ocsvm_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return [csv_path, config_path]


def main() -> None:
    print("AF-XAI local One-Class SVM training (Step 3)")
    print("Scalers are loaded from Step 2 and are not refit.")
    print()

    MODELS_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    client_rows = [run_client(client_id) for client_id in CLIENT_IDS]
    extra = write_combined_outputs(client_rows)

    generated = list(extra)
    for client_id in CLIENT_IDS:
        generated.extend(
            [
                MODELS_LOCAL_DIR / f"{client_id}_ocsvm.joblib",
                METRICS_DIR / f"{client_id}_predictions.csv",
                METRICS_DIR / f"{client_id}_metrics.json",
                FIGURES_DIR / f"{client_id}_confusion_matrix.png",
            ]
        )

    print("Generated files:")
    for path in generated:
        print(f"  {path.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
