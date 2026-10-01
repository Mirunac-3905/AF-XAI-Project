"""
AF-XAI Prediction Service for Adaptive Aggregated OCSVM.

This service loads the Step 4 adaptive aggregated RBF OCSVM model and provides
prediction functionality using the existing trained artifacts.

SCALING CONVENTION:
The Step 4 adaptive model was evaluated by scaling each client's test data with
that client's own Step 2 scaler. Since the API receives samples without client
context, this service uses client_1's scaler consistently for all predictions.
This matches the training convention where the global model was built from
separately scaled client feature spaces.
"""

from __future__ import annotations

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Import AdaptiveAggregatedOCSVM for joblib unpickling
from ml.aggregation.adaptive_aggregation import AdaptiveAggregatedOCSVM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_GLOBAL_DIR = PROJECT_ROOT / "models" / "global"
MODELS_LOCAL_DIR = PROJECT_ROOT / "models" / "local"

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


class PredictionService:
    """Service for loading the adaptive model and making predictions."""

    def __init__(self):
        self.model: Optional[AdaptiveAggregatedOCSVM] = None
        self.scaler = None
        self.model_loaded = False
        self.scaler_loaded = False

    def load_model(self) -> None:
        """Load the adaptive aggregated OCSVM model from Step 4."""
        model_path = MODELS_GLOBAL_DIR / "adaptive_aggregated_ocsvm.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = joblib.load(model_path)
        self.model_loaded = True

    def load_scaler(self) -> None:
        """
        Load a client scaler for input preprocessing.

        SCALING CONVENTION:
        The Step 4 adaptive model was evaluated by scaling each client's test data
        with that client's own Step 2 scaler. Since the API receives samples without
        client context, we use client_1's scaler consistently for all predictions.
        This matches the training convention where the global model was built from
        separately scaled client feature spaces.
        """
        scaler_path = MODELS_LOCAL_DIR / "client_1_scaler.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        self.scaler = joblib.load(scaler_path)
        self.scaler_loaded = True

    def initialize(self) -> None:
        """Load model and scaler on service startup."""
        self.load_model()
        self.load_scaler()

    def validate_input(self, data: Dict[str, Any]) -> tuple[bool, Optional[str], Optional[np.ndarray]]:
        """
        Validate input data and extract feature values.

        Returns:
            (is_valid, error_message, feature_array)
        """
        # Check for missing fields
        missing_features = [f for f in ML_FEATURES if f not in data]
        if missing_features:
            return False, f"Missing required feature: {missing_features[0]}", None

        # Check for extra fields that could create ambiguity
        feature_keys = set(data.keys())
        ml_feature_set = set(ML_FEATURES)
        extra_features = feature_keys - ml_feature_set
        if extra_features:
            return False, f"Extra feature names create ambiguity: {list(extra_features)}", None

        # Extract and validate feature values
        feature_values = []
        for feature in ML_FEATURES:
            value = data[feature]
            if value is None:
                return False, f"Null value for feature: {feature}", None
            try:
                float_value = float(value)
                feature_values.append(float_value)
            except (ValueError, TypeError):
                return False, f"Non-numeric value for feature: {feature}", None

        return True, None, np.array(feature_values).reshape(1, -1)

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for a single sample.

        Args:
            data: Dictionary containing the 10 ML features

        Returns:
            Dictionary with prediction results including:
            - predicted_fault: 0 (normal) or 1 (fault)
            - decision_score: raw decision function output
            - anomaly_score: -decision_score
            - label: "normal" or "fault"
        """
        if not self.model_loaded or not self.scaler_loaded:
            raise RuntimeError("Model or scaler not loaded. Call initialize() first.")

        # Validate input
        is_valid, error_msg, X_input = self.validate_input(data)
        if not is_valid:
            raise ValueError(error_msg)

        # Scale input using the client scaler
        X_scaled = self.scaler.transform(X_input)

        # Get decision score from adaptive model
        decision_score = self.model.decision_function(X_scaled)[0]

        # Compute anomaly score
        anomaly_score = -decision_score

        # Determine prediction
        predicted_fault = 1 if decision_score < 0 else 0
        label = "fault" if predicted_fault == 1 else "normal"

        return {
            "predicted_fault": int(predicted_fault),
            "decision_score": float(decision_score),
            "anomaly_score": float(anomaly_score),
            "label": label,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the loaded model."""
        if not self.model_loaded:
            return {"model_loaded": False}

        return {
            "model_loaded": True,
            "model_type": "AdaptiveAggregatedOCSVM",
            "n_support_vectors": int(self.model.support_vectors_.shape[0]),
            "gamma": float(self.model.gamma),
            "n_features_in": int(self.model.n_features_in_),
            "scaler_convention": "client_1_scaler (consistent with Step 4 evaluation per-client scaling)",
        }


# Global service instance
prediction_service = PredictionService()
