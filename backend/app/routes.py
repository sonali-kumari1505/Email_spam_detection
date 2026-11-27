"""
Flask API Blueprint module for the Intelligent Email Spam Detection System.

Defines REST API endpoints:
- POST /api/predict   : Classifies email using all 4 models + Hard Majority Voting.
- GET  /api/models    : Returns actual test performance metrics for all systems.
- GET  /api/analytics : Returns dataset EDA distributions, lengths, and vocabulary.
- GET  /api/health    : Health check and model loading verification.
"""

import os
import sys
import json
import time
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app

# Ensure backend root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from ml.ensemble import VotingEnsemble

api_bp = Blueprint("api", __name__)

# Server boot timestamp for uptime calculation
SERVER_START_TIME = time.time()

# In-memory singleton ensemble instance (loaded once on application start)
_ensemble_instance = None


def get_ensemble() -> VotingEnsemble:
    """
    Singleton getter for the VotingEnsemble.
    Loads models once into RAM and reuses the instance across requests.
    Prevents costly disk I/O and prohibits model retraining on API calls.
    """
    global _ensemble_instance
    if _ensemble_instance is None:
        _ensemble_instance = VotingEnsemble()
    return _ensemble_instance


@api_bp.route("/health", methods=["GET"])
def health_check():
    """
    GET /api/health
    Returns API operational status, uptime, and loaded models confirmation.
    """
    try:
        ensemble = get_ensemble()
        models_loaded = list(ensemble.models.keys())
        is_ready = ensemble.is_loaded
    except Exception as e:
        models_loaded = []
        is_ready = False

    uptime_seconds = round(time.time() - SERVER_START_TIME, 2)

    return jsonify({
        "status": "healthy" if is_ready else "degraded",
        "service": "Intelligent Email Spam Detection API",
        "version": "1.0.0",
        "uptime_seconds": uptime_seconds,
        "models_ready": is_ready,
        "loaded_classifiers": [
            "Multinomial Naive Bayes",
            "Linear Support Vector Machine (Calibrated)",
            "Logistic Regression",
            "Decision Tree Classifier"
        ],
        "ensemble_type": "Hard Majority Voting (with 2-2 Calibrated Probability Tie-Breaking)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }), 200


@api_bp.route("/predict", methods=["POST"])
def predict_message():
    """
    POST /api/predict
    Accepts JSON payload: {"message": "string"}
    Runs all 4 independent classifiers, applies hard majority voting,
    and resolves 2-2 ties via calibrated probability averaging.
    """
    # 1. Validate JSON presence
    if not request.is_json and not request.content_type.startswith("application/json"):
        return jsonify({
            "success": False,
            "error": "Invalid Content-Type. Request must be JSON (application/json)."
        }), 400

    data = request.get_json(silent=True)
    if data is None:
        try:
            raw_text = request.get_data(as_text=True)
            if raw_text:
                data = json.loads(raw_text)
        except Exception:
            data = None

    if data is None:
        return jsonify({
            "success": False,
            "error": "Malformed JSON payload in request body."
        }), 400

    # 2. Extract and validate message text
    # Accept either 'message' or 'text' key
    message = data.get("message")
    if message is None:
        message = data.get("text")

    if message is None or not isinstance(message, str):
        return jsonify({
            "success": False,
            "error": "Missing or invalid 'message' field. A non-empty string is required."
        }), 400

    message_cleaned = message.strip()
    if not message_cleaned:
        return jsonify({
            "success": False,
            "error": "The 'message' field cannot be blank or contain only whitespace."
        }), 400

    # 3. Perform Ensemble Prediction (using in-memory loaded models)
    try:
        ensemble = get_ensemble()
        result = ensemble.predict_single(message_cleaned)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Internal prediction engine error: {str(e)}"
        }), 500

    # 4. Return structured response
    return jsonify({
        "success": True,
        "data": {
            "input_message": result["text"],
            "cleaned_message": result["cleaned_text"],
            "final_prediction": result["final_prediction"],  # 0: Ham, 1: Spam
            "final_label": result["final_label"],            # 'ham' or 'spam'
            "vote_counts": result["vote_counts"],            # {"spam": X, "ham": Y, "total": 4}
            "is_tie": result["is_tie"],
            "tie_details": result["tie_details"],
            "average_spam_probability": result["average_spam_probability"],
            "decision_path": result["decision_path"],
            "individual_models": result["individual_models"]
        }
    }), 200


@api_bp.route("/models", methods=["GET"])
def get_models_metrics():
    """
    GET /api/models
    Returns actual, un-fabricated test evaluation metrics and confusion matrices
    for all four individual classifiers and the majority voting ensemble.
    """
    # Candidate paths for evaluation_results.json
    candidates = [
        config.MODELS_DIR / "evaluation_results.json",
        config.APP_MODELS_DIR / "evaluation_results.json",
        BASE_DIR / "models" / "evaluation_results.json"
    ]

    target_file = next((p for p in candidates if p.exists()), None)
    if target_file is None:
        return jsonify({
            "success": False,
            "error": "Evaluation metrics file not found. Please run ml/evaluate.py first."
        }), 404

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to read evaluation metrics: {str(e)}"
        }), 500

    return jsonify({
        "success": True,
        "data": metrics_data
    }), 200


@api_bp.route("/analytics", methods=["GET"])
def get_dataset_analytics():
    """
    GET /api/analytics
    Returns dataset EDA statistics: class distributions, percentage ratios,
    message length metrics, and top discriminating vocabulary.
    """
    candidates = [
        config.MODELS_DIR / "dataset_analytics.json",
        config.APP_MODELS_DIR / "dataset_analytics.json",
        BASE_DIR / "models" / "dataset_analytics.json"
    ]

    target_file = next((p for p in candidates if p.exists()), None)
    if target_file is None:
        return jsonify({
            "success": False,
            "error": "Dataset analytics file not found."
        }), 404

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            analytics_data = json.load(f)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to read dataset analytics: {str(e)}"
        }), 500

    return jsonify({
        "success": True,
        "data": analytics_data
    }), 200
