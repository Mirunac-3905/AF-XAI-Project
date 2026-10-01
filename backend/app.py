"""
AF-XAI Flask Application - Step 6: REST API for Adaptive Aggregated OCSVM.

This Flask application provides a REST API for the AF-XAI system, including:
- Model loading (Step 4 adaptive aggregated OCSVM)
- Prediction endpoint with SHAP explanations (Step 5)
- Health check endpoint

The API loads existing trained artifacts and does not retrain or refit any models.
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# IMPORTANT: Import AdaptiveAggregatedOCSVM BEFORE loading the model
# This is required for joblib to unpickle the saved model correctly
from ml.aggregation.adaptive_aggregation import AdaptiveAggregatedOCSVM

# Add backend directory to path for imports
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from flask import Flask
from flask_cors import CORS

from routes.prediction_routes import prediction_bp
from services.prediction_service import prediction_service
from services.xai_service import xai_service


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Enable CORS for frontend integration
    CORS(app)

    # Register blueprints
    app.register_blueprint(prediction_bp, url_prefix='/api')

    # Initialize services on startup
    try:
        print("Initializing AF-XAI Prediction Service...")
        prediction_service.initialize()
        print(f"  Model loaded: AdaptiveAggregatedOCSVM")
        print(f"  Support vectors: {prediction_service.model.support_vectors_.shape[0]}")
        print(f"  Gamma: {prediction_service.model.gamma}")
        print(f"  Scaler: client_1_scaler (consistent with Step 4 per-client scaling)")

        # Load model into XAI service (shared with prediction service)
        xai_service.load_model(prediction_service.model)
        print(f"  XAI service ready: SHAP KernelExplainer")
        print(f"  Background data: 100 samples (deterministic, random_state=42)")
        print(f"  Explainer: lazy initialization on first request")

    except Exception as e:
        print(f"ERROR: Failed to initialize services: {e}")
        raise

    return app


if __name__ == '__main__':
    app = create_app()

    print("\n" + "=" * 60)
    print("AF-XAI Prediction API")
    print("=" * 60)
    print("Model: AdaptiveAggregatedOCSVM")
    print("XAI: SHAP KernelExplainer")
    print("Server: http://localhost:5000")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)
