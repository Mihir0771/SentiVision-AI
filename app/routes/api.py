"""
REST API Blueprint (v1) — Programmatic endpoints for sentiment and emotion analysis.
Supports JWT Bearer token authentication and JSON-only request/response workflows.
"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from functools import wraps
import logging
from typing import Any, Callable, Dict, List, Optional
from flask import Blueprint, current_app, jsonify, request
import jwt

from app.models.analysis import Analysis
from app.models.user import User
from app.services.facial_analyzer import FacialEmotionAnalyzer
from app.services.nlp_utils import (
    detect_language,
    extract_entities,
    extract_keywords,
    get_text_statistics,
)
from app.services.scraper import ContentScraper
from app.services.text_analyzer import TextSentimentAnalyzer

logger = logging.getLogger(__name__)

bp = Blueprint("api", __name__, template_folder="../templates")

_text_analyzer: Optional[TextSentimentAnalyzer] = None
_facial_analyzer: Optional[FacialEmotionAnalyzer] = None
_scraper: Optional[ContentScraper] = None


def get_text_analyzer() -> TextSentimentAnalyzer:
    global _text_analyzer
    if _text_analyzer is None:
        _text_analyzer = TextSentimentAnalyzer()
    return _text_analyzer


def get_facial_analyzer() -> FacialEmotionAnalyzer:
    global _facial_analyzer
    if _facial_analyzer is None:
        _facial_analyzer = FacialEmotionAnalyzer()
    return _facial_analyzer


def get_scraper() -> ContentScraper:
    global _scraper
    if _scraper is None:
        _scraper = ContentScraper()
    return _scraper


# ---------------------------------------------------------------------------
# JWT Authentication Helper & Decorator
# ---------------------------------------------------------------------------

def token_required(f: Callable) -> Callable:
    """Decorator requiring a valid JWT Bearer token in the Authorization header."""
    @wraps(f)
    def decorated(*args: Any, **kwargs: Any) -> Any:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"status": "error", "message": "Missing or invalid Authorization header."}), 401

        token = auth_header.split(" ", 1)[1].strip()
        secret = current_app.config.get("JWT_SECRET_KEY", current_app.config.get("SECRET_KEY", "jwt-secret"))

        try:
            payload = jwt.decode(token, secret, algorithms=["HS256"])
            user_id = payload.get("sub")
            db = getattr(current_app, "db", None)
            user = None
            if db is not None:
                user = User.find_by_id(db, user_id)
            if not user:
                # If database is offline, permit valid token payload with simulated user
                user = User(username=payload.get("username", "api_user"), email=payload.get("email", "api@example.com"))
                user.id = user_id
            kwargs["user"] = user
        except jwt.ExpiredSignatureError:
            return jsonify({"status": "error", "message": "API token has expired."}), 401
        except jwt.InvalidTokenError:
            return jsonify({"status": "error", "message": "Invalid authentication token."}), 401

        return f(*args, **kwargs)

    return decorated


# ---------------------------------------------------------------------------
# Auth Endpoints
# ---------------------------------------------------------------------------

@bp.route("/auth/token", methods=["POST"])
def generate_token():
    """
    Authenticate user and issue a JWT access token.
    JSON Body: {"username": "...", "password": "..."}
    """
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"status": "error", "message": "Username and password are required."}), 400

    db = getattr(current_app, "db", None)
    if db is None:
        # Development / demo fallback
        if username and password:
            secret = current_app.config.get("JWT_SECRET_KEY", current_app.config.get("SECRET_KEY", "jwt-secret"))
            exp = datetime.now(timezone.utc) + timedelta(hours=24)
            token = jwt.encode({"sub": "demo_user", "username": username, "exp": exp}, secret, algorithm="HS256")
            return jsonify({
                "status": "success",
                "access_token": token,
                "token_type": "Bearer",
                "expires_in": 86400,
            }), 200
        return jsonify({"status": "error", "message": "Database offline."}), 503

    user = User.find_by_username(db, username) or User.find_by_email(db, username)
    if not user or not user.check_password(password):
        return jsonify({"status": "error", "message": "Invalid username or password."}), 401

    secret = current_app.config.get("JWT_SECRET_KEY", current_app.config.get("SECRET_KEY", "jwt-secret"))
    exp = datetime.now(timezone.utc) + timedelta(hours=24)
    payload = {
        "sub": user.id,
        "username": user.username,
        "email": user.email,
        "exp": exp,
    }
    token = jwt.encode(payload, secret, algorithm="HS256")

    return jsonify({
        "status": "success",
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": 86400,
    }), 200


# ---------------------------------------------------------------------------
# Analysis Endpoints
# ---------------------------------------------------------------------------

@bp.route("/analyze/text", methods=["POST"])
def api_analyze_text():
    """
    Analyze single text with ensemble sentiment scoring.
    JSON Body: {"text": "...", "models": ["transformer", "vader", "textblob"]}
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    models = data.get("models")

    if not text:
        return jsonify({"status": "error", "message": "Field 'text' is required and cannot be empty."}), 400

    try:
        analyzer = get_text_analyzer()
        analysis = analyzer.analyze(text=text, models=models)
        keywords = extract_keywords(text, top_n=6)
        entities = extract_entities(text)
        stats = get_text_statistics(text)
        lang = detect_language(text)

        response_data = {
            "sentiment": analysis,
            "keywords": keywords,
            "entities": entities,
            "statistics": stats,
            "language": lang,
        }

        # Optional save to DB if token provided
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                secret = current_app.config.get("JWT_SECRET_KEY", current_app.config.get("SECRET_KEY", "jwt-secret"))
                token = auth_header.split(" ", 1)[1].strip()
                payload = jwt.decode(token, secret, algorithms=["HS256"])
                db = getattr(current_app, "db", None)
                if db is not None:
                    rec = Analysis(
                        user_id=payload.get("sub", "api"),
                        analysis_type="text",
                        input_data=text[:1000],
                        results=analysis,
                        overall_sentiment=analysis.get("overall_sentiment", "neutral"),
                        confidence=analysis.get("overall_confidence", 0.0),
                        models_used=list(analysis.get("models", {}).keys()),
                        language=lang.get("language", "en"),
                    )
                    rec_id = rec.save(db)
                    response_data["id"] = rec_id
            except Exception:
                pass

        return jsonify({"status": "success", "data": response_data}), 200

    except Exception as exc:
        logger.error("API Text Analysis failure: %s", exc)
        return jsonify({"status": "error", "message": str(exc)}), 500


@bp.route("/analyze/batch", methods=["POST"])
def api_analyze_batch():
    """
    Analyze an array of texts in bulk.
    JSON Body: {"texts": ["text 1", "text 2", ...]}
    """
    data = request.get_json(silent=True) or {}
    texts = data.get("texts", [])

    if not isinstance(texts, list) or not texts:
        return jsonify({"status": "error", "message": "Field 'texts' must be a non-empty list of strings."}), 400

    try:
        analyzer = get_text_analyzer()
        results = analyzer.analyze_batch(texts)

        pos_count = sum(1 for r in results if r.get("overall_sentiment") == "positive")
        neg_count = sum(1 for r in results if r.get("overall_sentiment") == "negative")
        neu_count = sum(1 for r in results if r.get("overall_sentiment") == "neutral")
        total = len(results) or 1

        summary = {
            "total_items": len(results),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": neu_count,
            "positive_pct": round(pos_count / total * 100, 2),
            "negative_pct": round(neg_count / total * 100, 2),
            "neutral_pct": round(neu_count / total * 100, 2),
        }

        return jsonify({"status": "success", "data": {"items": results, "summary": summary}}), 200

    except Exception as exc:
        logger.error("API Batch Analysis failure: %s", exc)
        return jsonify({"status": "error", "message": str(exc)}), 500


@bp.route("/analyze/url", methods=["POST"])
def api_analyze_url():
    """
    Scrape and analyze sentiment of content from a URL.
    JSON Body: {"url": "https://..."}
    """
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()

    if not url:
        return jsonify({"status": "error", "message": "Field 'url' is required."}), 400

    try:
        scraper = get_scraper()
        analyzer = get_text_analyzer()
        result = scraper.scrape_and_analyze(url, analyzer)

        if not result.get("success"):
            return jsonify({"status": "error", "message": result.get("error", "Failed to retrieve URL")}), 400

        return jsonify({"status": "success", "data": result}), 200

    except Exception as exc:
        logger.error("API URL Analysis failure: %s", exc)
        return jsonify({"status": "error", "message": str(exc)}), 500


@bp.route("/analyze/image", methods=["POST"])
def api_analyze_image():
    """
    Analyze facial emotions in an image (base64 in JSON or multipart upload).
    """
    raw_bytes: Optional[bytes] = None

    if "image" in request.files:
        file = request.files["image"]
        raw_bytes = file.read()
    else:
        data = request.get_json(silent=True) or {}
        image_b64 = data.get("image", "")
        if image_b64:
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]
            try:
                raw_bytes = base64.b64decode(image_b64)
            except Exception:
                return jsonify({"status": "error", "message": "Invalid base64 encoding."}), 400

    if not raw_bytes:
        return jsonify({"status": "error", "message": "Please supply an image file or base64 string."}), 400

    try:
        analyzer = get_facial_analyzer()
        result = analyzer.analyze_image_bytes(raw_bytes)
        return jsonify({"status": "success", "data": result}), 200

    except Exception as exc:
        logger.error("API Image Analysis failure: %s", exc)
        return jsonify({"status": "error", "message": str(exc)}), 500


# ---------------------------------------------------------------------------
# Public Open-Access Resource Endpoints (No Authentication Required)
# ---------------------------------------------------------------------------

@bp.route("/analyses", methods=["GET"])
def api_list_analyses():
    """List historical analyses (open public access)."""
    db = getattr(current_app, "db", None)
    if db is None:
        return jsonify({"status": "success", "data": {"items": [], "total": 0}}), 200

    page = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 20))
    analyses = Analysis.find_by_user(db, None, limit=limit, skip=(page - 1) * limit)
    total = db["analyses"].count_documents({})

    return jsonify({
        "status": "success",
        "data": {
            "items": [a.to_dict() for a in analyses],
            "page": page,
            "limit": limit,
            "total": total,
        }
    }), 200


@bp.route("/analyses/<analysis_id>", methods=["GET"])
def api_get_analysis(analysis_id: str):
    """Retrieve details for a specific analysis record."""
    db = getattr(current_app, "db", None)
    if db is None:
        return jsonify({"status": "error", "message": "Database unavailable."}), 503

    record = Analysis.find_by_id(db, analysis_id)
    if not record:
        return jsonify({"status": "error", "message": "Analysis record not found."}), 404

    return jsonify({"status": "success", "data": record.to_dict()}), 200


@bp.route("/analyses/<analysis_id>", methods=["DELETE"])
def api_delete_analysis(analysis_id: str):
    """Delete an analysis record."""
    db = getattr(current_app, "db", None)
    if db is None:
        return jsonify({"status": "error", "message": "Database unavailable."}), 503

    success = Analysis.delete(db, analysis_id)
    if not success:
        return jsonify({"status": "error", "message": "Record not found."}), 404

    return jsonify({"status": "success", "message": "Record deleted successfully."}), 200


@bp.route("/stats", methods=["GET"])
def api_stats():
    """Return platform sentiment distribution statistics."""
    db = getattr(current_app, "db", None)
    if db is None:
        return jsonify({
            "status": "success",
            "data": {
                "total_analyses": 0,
                "positive_pct": 0,
                "negative_pct": 0,
                "neutral_pct": 0,
                "avg_confidence": 0.0,
            }
        }), 200

    stats = Analysis.get_user_stats(db, None)
    return jsonify({"status": "success", "data": stats}), 200
