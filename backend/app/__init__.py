"""
Flask Application Factory module.
Initializes the Flask app, configures CORS, registers blueprints,
and eagerly loads the Voting Ensemble into memory on startup.
"""

from flask import Flask, jsonify
from flask_cors import CORS
import config


def create_app():
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__)

    # Configure Cross-Origin Resource Sharing (CORS) for React frontend communication
    CORS(
        app,
        resources={r"/api/*": {"origins": config.CORS_ORIGINS}},
        supports_credentials=True,
    )

    # Register API blueprints
    from app.routes import api_bp, get_ensemble
    app.register_blueprint(api_bp, url_prefix="/api")

    # Global JSON error handlers
    @app.errorhandler(404)
    def not_found_handler(e):
        return jsonify({
            "success": False,
            "error": "Endpoint not found. Available endpoints: /api/predict, /api/models, /api/analytics, /api/health"
        }), 404

    @app.errorhandler(405)
    def method_not_allowed_handler(e):
        return jsonify({
            "success": False,
            "error": "HTTP method not allowed for this endpoint."
        }), 405

    @app.errorhandler(500)
    def internal_error_handler(e):
        return jsonify({
            "success": False,
            "error": "Internal server error occurred."
        }), 500

    # Eagerly preload models into memory so the first API request is instantaneous
    with app.app_context():
        try:
            print("[Startup] Preloading Voting Ensemble models into memory...")
            get_ensemble()
            print("[Startup] All 4 ML models and ensemble loaded successfully!")
        except Exception as e:
            print(f"[Startup Warning] Could not preload models: {e}")

    return app
