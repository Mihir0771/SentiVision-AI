"""
AI Sentiment Analyzer - Application Factory

Creates and configures the Flask application, wires up extensions
(MongoDB, Flask-Login, Flask-SocketIO, CORS), registers blueprints,
and sets up logging / error handlers.

Usage::

    from app import create_app
    app = create_app("development")
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Optional

from flask import Flask, jsonify, render_template
from flask_cors import CORS
from flask_login import LoginManager, current_user
from flask_socketio import SocketIO
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from app.config import config as config_map

# ---------------------------------------------------------------------------
# Extension instances (created once, initialised inside create_app)
# ---------------------------------------------------------------------------
login_manager = LoginManager()
login_manager.login_view = "auth.login"  # type: ignore[assignment]
login_manager.login_message_category = "info"

socketio = SocketIO()


def create_app(config_name: Optional[str] = None) -> Flask:
    """
    Application factory.

    Args:
        config_name: One of ``"development"``, ``"production"``, or
            ``"testing"``.  Defaults to the ``FLASK_ENV`` environment
            variable, falling back to ``"development"``.

    Returns:
        A fully configured :class:`~flask.Flask` instance.
    """
    if config_name is None:
        config_name = os.environ.get("FLASK_ENV", "development")

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # ---- Load configuration ------------------------------------------------
    app.config.from_object(config_map[config_name])

    # Ensure UPLOAD_FOLDER is absolute
    upload_folder: str = app.config["UPLOAD_FOLDER"]
    if not os.path.isabs(upload_folder):
        upload_folder = os.path.join(app.root_path, upload_folder)
        app.config["UPLOAD_FOLDER"] = upload_folder
    os.makedirs(upload_folder, exist_ok=True)

    # ---- Logging -----------------------------------------------------------
    _configure_logging(app)

    # ---- MongoDB -----------------------------------------------------------
    _init_mongodb(app)

    # ---- Flask-Login -------------------------------------------------------
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id: str):
        """
        Callback used by Flask-Login to reload the user object from the
        session cookie's stored user ID.
        """
        from app.models.user import User  # local import to avoid circulars

        db = getattr(app, "db", None)
        if db is None:
            return None
        return User.find_by_id(db, user_id)

    # ---- Flask-SocketIO ----------------------------------------------------
    socketio.init_app(app, cors_allowed_origins="*", async_mode="threading")

    # ---- CORS --------------------------------------------------------------
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # ---- Blueprints --------------------------------------------------------
    _register_blueprints(app)

    # ---- Error handlers ----------------------------------------------------
    _register_error_handlers(app)

    # ---- Context processors ------------------------------------------------
    @app.context_processor
    def inject_globals() -> dict:
        """Make ``current_user`` and ``current_year`` available in all templates."""
        return {
            "current_user": current_user,
            "current_year": datetime.now(timezone.utc).year,
        }

    app.logger.info(
        "Sentiment Analyzer app created  [env=%s]", config_name
    )
    return app


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _init_mongodb(app: Flask) -> None:
    """
    Connect to MongoDB and attach the client / database to the app.

    The connection is tested with a ``ping`` command.  If the database is
    unreachable the app still starts but ``app.db`` is set to ``None`` so
    that routes can degrade gracefully.
    """
    mongo_uri: str = app.config.get("MONGO_URI", "")
    timeout_ms = 300 if app.config.get("TESTING") else 3000
    try:
        client: MongoClient = MongoClient(
            mongo_uri, serverSelectionTimeoutMS=timeout_ms
        )
        # Verify the connection is alive
        client.admin.command("ping")
        app.mongo = client  # type: ignore[attr-defined]
        # Extract DB name from URI, falling back to "sentiment_analyzer"
        db_name = mongo_uri.rsplit("/", 1)[-1].split("?")[0] or "sentiment_analyzer"
        app.db = client[db_name]  # type: ignore[attr-defined]
        app.logger.info("MongoDB connected  [uri=%s]", mongo_uri)
    except ConnectionFailure as exc:
        app.logger.warning(
            "MongoDB connection failed — running in degraded mode: %s", exc
        )
        app.mongo = None  # type: ignore[attr-defined]
        app.db = None  # type: ignore[attr-defined]
    except Exception as exc:
        app.logger.warning(
            "MongoDB initialisation error — running in degraded mode: %s", exc
        )
        app.mongo = None  # type: ignore[attr-defined]
        app.db = None  # type: ignore[attr-defined]


def _register_blueprints(app: Flask) -> None:
    """
    Import and register all route blueprints.

    Each blueprint module is expected to expose a ``bp`` variable.  Missing
    modules are logged as warnings so that the app can start even if a
    blueprint hasn't been created yet.
    """
    blueprint_registry: list[tuple[str, str]] = [
        ("app.routes.main", "/"),
        ("app.routes.auth", "/auth"),
        ("app.routes.analysis", "/analyze"),
        ("app.routes.camera", "/camera"),
        ("app.routes.batch", "/batch"),
        ("app.routes.api", "/api/v1"),
        ("app.routes.reports", "/reports"),
    ]
    for module_path, url_prefix in blueprint_registry:
        try:
            import importlib

            module = importlib.import_module(module_path)
            bp = getattr(module, "bp", None)
            if bp is not None:
                app.register_blueprint(bp, url_prefix=url_prefix)
                app.logger.debug(
                    "Registered blueprint %s at %s", module_path, url_prefix
                )
            else:
                app.logger.warning(
                    "Module %s has no 'bp' attribute — skipped", module_path
                )
        except ModuleNotFoundError:
            app.logger.warning(
                "Blueprint module %s not found — skipped", module_path
            )
        except Exception as exc:
            app.logger.error(
                "Error registering blueprint %s: %s", module_path, exc
            )


def _register_error_handlers(app: Flask) -> None:
    """Register JSON-aware error handlers for common HTTP errors."""

    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found."""
        if _wants_json():
            return jsonify({"error": "Resource not found", "status": 404}), 404
        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 Internal Server Error."""
        app.logger.error("Internal server error: %s", error)
        if _wants_json():
            return (
                jsonify({"error": "Internal server error", "status": 500}),
                500,
            )
        return render_template("errors/500.html"), 500

    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle 413 Payload Too Large."""
        if _wants_json():
            return (
                jsonify(
                    {
                        "error": "File too large. Maximum size is 16 MB.",
                        "status": 413,
                    }
                ),
                413,
            )
        return render_template("errors/413.html"), 413


def _wants_json() -> bool:
    """Return ``True`` if the client prefers a JSON response."""
    from flask import request

    best = request.accept_mimetypes.best_match(
        ["application/json", "text/html"]
    )
    return best == "application/json"


def _configure_logging(app: Flask) -> None:
    """
    Set up rotating-file and console log handlers.

    Log files are written next to ``run.py`` (project root).  The format
    includes timestamps, level, logger name, and the message.
    """
    log_level_name: str = app.config.get("LOG_LEVEL", "INFO")
    log_level: int = getattr(logging, log_level_name.upper(), logging.INFO)
    log_file: str = app.config.get("LOG_FILE", "app.log")

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler (5 MB × 5 backups)
    log_path = os.path.join(
        os.path.dirname(app.root_path), log_file
    )
    file_handler = RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    app.logger.setLevel(log_level)
    # Avoid duplicate handlers on repeated create_app calls (e.g. in tests)
    if not app.logger.handlers:
        app.logger.addHandler(file_handler)
        app.logger.addHandler(console_handler)
