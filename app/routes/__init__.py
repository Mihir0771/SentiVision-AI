"""
Routes package for the Sentiment Analyzer application.
"""

from app.routes.main import bp as main_bp
from app.routes.auth import bp as auth_bp
from app.routes.analysis import bp as analysis_bp
from app.routes.camera import bp as camera_bp
from app.routes.batch import bp as batch_bp
from app.routes.api import bp as api_bp
from app.routes.reports import bp as reports_bp

__all__ = [
    "main_bp",
    "auth_bp",
    "analysis_bp",
    "camera_bp",
    "batch_bp",
    "api_bp",
    "reports_bp",
]
