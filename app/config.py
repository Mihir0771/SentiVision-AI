"""
Configuration management for Sentiment Analyzer.

Provides environment-specific configuration classes for the Flask application.
All sensitive values are loaded from environment variables with sensible defaults
for development convenience.
"""

import os
from datetime import timedelta


class BaseConfig:
    """
    Base configuration shared across all environments.

    Attributes:
        SECRET_KEY: Flask secret key for session signing and CSRF protection.
        MONGO_URI: MongoDB connection string.
        JWT_SECRET_KEY: Secret key for JWT token generation/verification.
        UPLOAD_FOLDER: Absolute path to the file upload directory.
        MAX_CONTENT_LENGTH: Maximum allowed request payload size (bytes).
        ALLOWED_EXTENSIONS: Set of permitted file extensions for uploads.
        SESSION_TYPE: Flask-Session backend type.
        PERMANENT_SESSION_LIFETIME: Duration before sessions expire.
    """

    # --- Security ---
    SECRET_KEY: str = os.environ.get(
        "SECRET_KEY", "dev-secret-key-change-in-production-immediately"
    )
    JWT_SECRET_KEY: str = os.environ.get(
        "JWT_SECRET_KEY", "jwt-dev-secret-change-in-production"
    )

    # --- Database ---
    MONGO_URI: str = os.environ.get(
        "MONGO_URI", "mongodb://localhost:27017/sentiment_analyzer"
    )

    # --- File Uploads ---
    UPLOAD_FOLDER: str = os.environ.get(
        "UPLOAD_FOLDER",
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "uploads"),
    )
    MAX_CONTENT_LENGTH: int = int(
        os.environ.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024)  # 16 MB
    )
    ALLOWED_EXTENSIONS: set = {
        "txt", "csv", "json", "xlsx", "xls",
        "png", "jpg", "jpeg", "gif", "bmp",
        "pdf", "docx",
    }

    # --- Sessions ---
    SESSION_TYPE: str = "filesystem"
    PERMANENT_SESSION_LIFETIME: timedelta = timedelta(days=7)

    # --- Logging ---
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.environ.get("LOG_FILE", "app.log")


class DevelopmentConfig(BaseConfig):
    """Development configuration with debug mode enabled."""

    DEBUG: bool = True
    TESTING: bool = False


class ProductionConfig(BaseConfig):
    """
    Production configuration.

    All secrets MUST be provided via environment variables.
    Debug mode is disabled and stricter security defaults apply.
    """

    DEBUG: bool = False
    TESTING: bool = False

    # In production these MUST come from env — no fallback defaults
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "")
    JWT_SECRET_KEY: str = os.environ.get("JWT_SECRET_KEY", "")
    MONGO_URI: str = os.environ.get(
        "MONGO_URI", "mongodb://localhost:27017/sentiment_analyzer"
    )

    # Stricter session lifetime in production
    PERMANENT_SESSION_LIFETIME: timedelta = timedelta(days=1)


class TestingConfig(BaseConfig):
    """Testing configuration with a separate test database."""

    DEBUG: bool = True
    TESTING: bool = True
    MONGO_URI: str = os.environ.get(
        "MONGO_URI", "mongodb://localhost:27017/sentiment_analyzer_test"
    )
    # Smaller upload limit for tests
    MAX_CONTENT_LENGTH: int = 4 * 1024 * 1024  # 4 MB


# ---------------------------------------------------------------------------
# Config registry — import this dict wherever you need to resolve a config
# by name (e.g. from FLASK_ENV).
# ---------------------------------------------------------------------------
config: dict[str, type] = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}
