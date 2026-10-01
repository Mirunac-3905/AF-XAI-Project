"""
AF-XAI Prediction Routes.

Flask routes for the prediction API including:
- GET /api/health: Health check endpoint
- POST /api/predict: Prediction endpoint with SHAP explanations
"""

from flask import Blueprint, request, jsonify
import numpy as np
from typing import Dict, Any
import sys
from pathlib import Path

# Add parent directory to path for imports
backend_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_dir))

from services.prediction_service import prediction_service
from services.xai_service import xai_service

prediction_bp = Blueprint('prediction', __name__)


@prediction_bp.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.

    Returns:
        JSON response with service status information.
    """
    model_info = prediction_service.get_model_info()
    xai_info = xai_service.get_xai_info()

    return jsonify({
        "success": True,
        "service": "AF-XAI Prediction API",
        "model": "AdaptiveAggregatedOCSVM",
        "model_loaded": model_info.get("model_loaded", False),
        "xai": "SHAP KernelExplainer",
    }), 200


@prediction_bp.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint with SHAP explanations.

    Input JSON:
    {
      "voltage": 1.01,
      "frequency": 50.0,
      "active_power": 100.0,
      "reactive_power": 20.0,
      "load_demand": 120.0,
      "renewable_output": 30.0,
      "temperature": 30.0,
      "humidity": 60.0,
      "power_loss": 5.0,
      "stability_index": 15.5
    }

    Returns:
        JSON response with prediction and feature-level SHAP explanations.
    """
    try:
        # Validate request has JSON data
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Request must be JSON"
            }), 400

        data = request.get_json()

        # Validate input using prediction service
        is_valid, error_msg, X_input = prediction_service.validate_input(data)
        if not is_valid:
            return jsonify({
                "success": False,
                "error": error_msg
            }), 400

        # Make prediction
        prediction_result = prediction_service.predict(data)

        # Get SHAP explanations
        # We need both scaled and original features for SHAP
        X_scaled = prediction_service.scaler.transform(X_input)
        explanations = xai_service.explain(X_scaled, X_input)

        # Build response
        response = {
            "success": True,
            "prediction": {
                "label": prediction_result["label"],
                "predicted_fault": prediction_result["predicted_fault"],
                "decision_score": prediction_result["decision_score"],
                "anomaly_score": prediction_result["anomaly_score"],
                "explanations": explanations,
            }
        }

        return jsonify(response), 200

    except ValueError as e:
        # Input validation errors
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except RuntimeError as e:
        # Service not initialized errors
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    except Exception as e:
        # Unexpected errors (don't expose stack traces)
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500
