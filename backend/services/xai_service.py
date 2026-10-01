"""
AF-XAI XAI Service for SHAP Explanations.

This service provides SHAP-based feature-level explanations for predictions
made by the adaptive aggregated OCSVM model. It reuses the Step 5 SHAP
implementation methodology.

The function being explained is the anomaly score:
f(X) = -adaptive_model.decision_function(X)

Higher anomaly score indicates stronger fault/anomaly indication.

SHAP VALUE INTERPRETATION:
- Positive SHAP value: pushes anomaly score toward higher fault/anomaly indication
- Negative SHAP value: pushes anomaly score toward lower anomaly/normal indication
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Import AdaptiveAggregatedOCSVM for joblib unpickling
from ml.aggregation.adaptive_aggregation import AdaptiveAggregatedOCSVM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
SCALED_DIR = PROCESSED_DIR / "scaled"

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

BACKGROUND_N_SAMPLES = 100
BACKGROUND_RANDOM_STATE = 42


class XAIService:
    """Service for SHAP-based explanations of model predictions."""

    def __init__(self):
        self.model: Optional[AdaptiveAggregatedOCSVM] = None
        self.background_data: Optional[np.ndarray] = None
        self.explainer: Optional[shap.KernelExplainer] = None
        self.model_loaded = False
        self.background_loaded = False
        self.explainer_initialized = False

    def load_model(self, model: AdaptiveAggregatedOCSVM) -> None:
        """Set the model for explanation (typically shared with prediction service)."""
        self.model = model
        self.model_loaded = True

    def load_background_data(self) -> None:
        """
        Load and combine scaled normal training data from all clients.
        Uses the same deterministic 100-sample background dataset as Step 5.
        """
        background_parts = []
        for client_id in CLIENT_IDS:
            train_path = SCALED_DIR / f"{client_id}_train_scaled.csv"
            if not train_path.exists():
                raise FileNotFoundError(f"Scaled training data not found: {train_path}")

            train_df = pd.read_csv(train_path)

            # Filter only normal samples (fault_indicator = 0)
            normal_df = train_df[train_df["fault_indicator"] == 0]

            # Extract only the ML features
            features_df = normal_df[ML_FEATURES]
            background_parts.append(features_df)

        # Combine all normal training data
        combined_background = pd.concat(background_parts, axis=0, ignore_index=True)

        # Sample a representative subset (deterministic, same as Step 5)
        if len(combined_background) > BACKGROUND_N_SAMPLES:
            combined_background = combined_background.sample(
                n=BACKGROUND_N_SAMPLES, random_state=BACKGROUND_RANDOM_STATE
            )

        self.background_data = combined_background.to_numpy()
        self.background_loaded = True

    def initialize_explainer(self) -> None:
        """
        Initialize the SHAP KernelExplainer with the background data.

        This is done lazily on first request because it can be computationally
        expensive. The explainer is then reused for subsequent requests.
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if not self.background_loaded:
            raise RuntimeError("Background data not loaded. Call load_background_data() first.")

        # Define the function to explain (anomaly score)
        def predict_fn(X):
            return -self.model.decision_function(X)

        # Create KernelExplainer with background data
        self.explainer = shap.KernelExplainer(predict_fn, self.background_data)
        self.explainer_initialized = True

    def explain(
        self,
        X_scaled: np.ndarray,
        X_original: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Generate SHAP explanations for a single sample.

        Args:
            X_scaled: Scaled feature values (1, n_features)
            X_original: Original (unscaled) feature values (1, n_features)

        Returns:
            List of feature-level explanations, each containing:
            - feature: feature name
            - value: original input value
            - shap_value: SHAP contribution value
            - contribution: "fault" or "normal" based on SHAP sign
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Lazy initialization of explainer
        if not self.explainer_initialized:
            self.load_background_data()
            self.initialize_explainer()

        # Compute SHAP values
        shap_values = self.explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Flatten if needed
        if shap_values.ndim > 1:
            shap_values = shap_values.flatten()

        # Extract original values
        original_values = X_original.flatten()

        # Build feature-level explanations
        explanations = []
        for feature, value, shap_val in zip(ML_FEATURES, original_values, shap_values):
            contribution = "fault" if shap_val > 0 else "normal"
            explanations.append({
                "feature": feature,
                "value": float(value),
                "shap_value": float(shap_val),
                "contribution": contribution,
            })

        return explanations

    def get_xai_info(self) -> Dict[str, Any]:
        """Return information about the XAI service."""
        return {
            "method": "SHAP KernelExplainer",
            "function_explained": "anomaly_score = -decision_function(X)",
            "background_samples": BACKGROUND_N_SAMPLES,
            "background_random_state": BACKGROUND_RANDOM_STATE,
            "explainer_initialized": self.explainer_initialized,
        }


# Global service instance
xai_service = XAIService()
