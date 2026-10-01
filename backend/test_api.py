"""
AF-XAI API Tests.

Test suite for the Flask REST API including:
1. Health endpoint test
2. Valid prediction test
3. Missing feature validation test
4. Non-numeric feature validation test
"""

import requests
import json
import time
import subprocess
import sys
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:5000"
API_HEALTH = f"{BASE_URL}/api/health"
API_PREDICT = f"{BASE_URL}/api/predict"

# Valid test input (example values only, not experimental results)
VALID_INPUT = {
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


def start_server():
    """Start the Flask server in the background."""
    print("Starting Flask server...")
    # Start server in background
    proc = subprocess.Popen(
        [sys.executable, "backend/app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path(__file__).parent.parent
    )
    # Give server time to start
    time.sleep(5)
    return proc


def stop_server(proc):
    """Stop the Flask server."""
    print("Stopping Flask server...")
    proc.terminate()
    proc.wait(timeout=10)


def test_health_endpoint():
    """Test GET /api/health endpoint."""
    print("\n" + "=" * 60)
    print("TEST 1: GET /api/health")
    print("=" * 60)

    try:
        response = requests.get(API_HEALTH, timeout=10)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")

            # Verify response structure
            assert data["success"] == True, "success should be True"
            assert data["service"] == "AF-XAI Prediction API", "service name mismatch"
            assert data["model"] == "AdaptiveAggregatedOCSVM", "model name mismatch"
            assert data["model_loaded"] == True, "model should be loaded"
            assert data["xai"] == "SHAP KernelExplainer", "XAI method mismatch"

            print("[PASS] Health endpoint test PASSED")
            return True
        else:
            print(f"[FAIL] Health endpoint test FAILED: Expected 200, got {response.status_code}")
            return False

    except Exception as e:
        print(f"[FAIL] Health endpoint test FAILED: {e}")
        return False


def test_valid_prediction():
    """Test POST /api/predict with valid numeric input."""
    print("\n" + "=" * 60)
    print("TEST 2: POST /api/predict with valid input")
    print("=" * 60)

    try:
        response = requests.post(API_PREDICT, json=VALID_INPUT, timeout=30)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")

            # Verify response structure
            assert data["success"] == True, "success should be True"
            assert "prediction" in data, "prediction key missing"

            pred = data["prediction"]
            assert "label" in pred, "label missing"
            assert "predicted_fault" in pred, "predicted_fault missing"
            assert "decision_score" in pred, "decision_score missing"
            assert "anomaly_score" in pred, "anomaly_score missing"
            assert "explanations" in pred, "explanations missing"

            # Verify explanations
            explanations = pred["explanations"]
            assert len(explanations) == 10, f"Expected 10 explanations, got {len(explanations)}"

            # Verify each explanation has required fields
            for exp in explanations:
                assert "feature" in exp, "feature missing from explanation"
                assert "value" in exp, "value missing from explanation"
                assert "shap_value" in exp, "shap_value missing from explanation"
                assert "contribution" in exp, "contribution missing from explanation"

            print("[PASS] Valid prediction test PASSED")
            print(f"\nTest Response Summary:")
            print(f"  Predicted label: {pred['label']}")
            print(f"  Predicted fault: {pred['predicted_fault']}")
            print(f"  Decision score: {pred['decision_score']:.6f}")
            print(f"  Anomaly score: {pred['anomaly_score']:.6f}")

            # Show top 5 SHAP contributors
            sorted_explanations = sorted(explanations, key=lambda x: abs(x['shap_value']), reverse=True)
            print(f"  Top 5 SHAP contributors:")
            for i, exp in enumerate(sorted_explanations[:5], 1):
                print(f"    {i}. {exp['feature']}: {exp['shap_value']:.6f} ({exp['contribution']})")

            return True, data
        else:
            print(f"[FAIL] Valid prediction test FAILED: Expected 200, got {response.status_code}")
            print(f"Response: {response.text}")
            return False, None

    except Exception as e:
        print(f"[FAIL] Valid prediction test FAILED: {e}")
        return False, None


def test_missing_feature():
    """Test POST /api/predict with missing feature."""
    print("\n" + "=" * 60)
    print("TEST 3: POST /api/predict with missing feature")
    print("=" * 60)

    try:
        # Remove voltage from input
        invalid_input = VALID_INPUT.copy()
        del invalid_input["voltage"]

        response = requests.post(API_PREDICT, json=invalid_input, timeout=10)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 400:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")

            assert data["success"] == False, "success should be False"
            assert "error" in data, "error key missing"
            assert "voltage" in data["error"] or "Missing" in data["error"], "error should mention missing feature"

            print("[PASS] Missing feature validation test PASSED")
            return True
        else:
            print(f"[FAIL] Missing feature test FAILED: Expected 400, got {response.status_code}")
            return False

    except Exception as e:
        print(f"[FAIL] Missing feature test FAILED: {e}")
        return False


def test_non_numeric_feature():
    """Test POST /api/predict with non-numeric feature."""
    print("\n" + "=" * 60)
    print("TEST 4: POST /api/predict with non-numeric feature")
    print("=" * 60)

    try:
        # Set voltage to a non-numeric value
        invalid_input = VALID_INPUT.copy()
        invalid_input["voltage"] = "invalid"

        response = requests.post(API_PREDICT, json=invalid_input, timeout=10)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 400:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")

            assert data["success"] == False, "success should be False"
            assert "error" in data, "error key missing"
            assert "numeric" in data["error"] or "voltage" in data["error"], "error should mention non-numeric value"

            print("[PASS] Non-numeric feature validation test PASSED")
            return True
        else:
            print(f"[FAIL] Non-numeric feature test FAILED: Expected 400, got {response.status_code}")
            return False

    except Exception as e:
        print(f"[FAIL] Non-numeric feature test FAILED: {e}")
        return False


def main():
    """Run all API tests."""
    print("\n" + "=" * 60)
    print("AF-XAI API Test Suite")
    print("=" * 60)

    # Start server
    proc = start_server()

    try:
        # Run tests
        results = []

        # Test 1: Health endpoint
        results.append(("Health endpoint", test_health_endpoint()))

        # Test 2: Valid prediction
        success, pred_data = test_valid_prediction()
        results.append(("Valid prediction", success))

        # Test 3: Missing feature
        results.append(("Missing feature validation", test_missing_feature()))

        # Test 4: Non-numeric feature
        results.append(("Non-numeric feature validation", test_non_numeric_feature()))

        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        for test_name, passed in results:
            status = "[PASS]" if passed else "[FAIL]"
            print(f"{test_name}: {status}")

        all_passed = all(result[1] for result in results)
        print("\n" + "=" * 60)
        if all_passed:
            print("ALL TESTS PASSED")
        else:
            print("SOME TESTS FAILED")
        print("=" * 60)

        return 0 if all_passed else 1

    finally:
        # Stop server
        stop_server(proc)


if __name__ == "__main__":
    sys.exit(main())
