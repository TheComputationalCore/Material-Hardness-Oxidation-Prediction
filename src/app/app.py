"""
app.py — Flask application factory.
"""

from flask import Flask

from src.app.routes import app_bp


def create_app():
    app = Flask(__name__)

    # Register blueprints
    app.register_blueprint(app_bp)

    # Config
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    return app


# For local development
if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5001, debug=True)
