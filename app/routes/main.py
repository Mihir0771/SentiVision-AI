"""
Main routes blueprint — serves dashboard, landing, about, and health check.
"""

from __future__ import annotations

import logging
from typing import Any, Dict
from flask import Blueprint, current_app, jsonify, render_template
from flask_login import current_user

from app.models.analysis import Analysis

logger = logging.getLogger(__name__)

bp = Blueprint("main", __name__, template_folder="../templates")


@bp.route("/")
@bp.route("/dashboard")
def dashboard():
    """
    Render main dashboard or landing view.
    Aggregates stats if DB is available, or provides graceful demo defaults.
    """
    db = getattr(current_app, "db", None)

    stats = {
        "total_analyses": 0,
        "positive_pct": 0,
        "negative_pct": 0,
        "neutral_pct": 0,
        "avg_confidence": 0.0,
    }
    sentiment_distribution = {"positive": 0, "negative": 0, "neutral": 0}
    recent_analyses: list[dict[str, Any]] = []
    timeline_data = {"labels": [], "positive": [], "negative": [], "neutral": []}

    if db is not None:
        try:
            # Open platform statistics
            stats = Analysis.get_user_stats(db, None)
            dist = Analysis.get_sentiment_distribution(db, None)
            sentiment_distribution = {
                "positive": dist.get("positive", 0),
                "negative": dist.get("negative", 0),
                "neutral": dist.get("neutral", 0),
            }
            recent = Analysis.find_by_user(db, user_id, limit=8)
            recent_analyses = [a.to_dict() for a in recent]
        except Exception as e:
            logger.warning("Error fetching dashboard statistics from database: %s", e)

    return render_template(
        "dashboard.html",
        stats=stats,
        sentiment_distribution=sentiment_distribution,
        recent_analyses=recent_analyses,
        timeline_data=timeline_data,
    )


@bp.route("/about")
def about():
    """Render About and Architecture page."""
    return render_template("about.html")


@bp.route("/api-docs")
def api_docs():
    """Render Interactive REST API Documentation."""
    return render_template("api_docs.html")


@bp.route("/health")
def health():
    """
    Health check endpoint returning system status and MongoDB state.
    """
    db = getattr(current_app, "db", None)
    db_status = "connected" if db is not None else "disconnected"

    return jsonify({
        "status": "healthy",
        "database": db_status,
        "version": "1.0.0",
        "services": {
            "text_analyzer": "active",
            "facial_analyzer": "active",
            "batch_processor": "active",
            "scraper": "active",
        }
    }), 200
