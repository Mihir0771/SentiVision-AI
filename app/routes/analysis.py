"""
Text analysis blueprint — single text, URL scraping, aspect analysis, and history.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required

from app.models.analysis import Analysis
from app.services.nlp_utils import (
    detect_language,
    extract_entities,
    extract_keywords,
    get_text_statistics,
)
from app.services.scraper import ContentScraper
from app.services.text_analyzer import TextSentimentAnalyzer

logger = logging.getLogger(__name__)

bp = Blueprint("analysis", __name__, template_folder="../templates")

# Lazy-initialised singletons
_text_analyzer: Optional[TextSentimentAnalyzer] = None
_scraper: Optional[ContentScraper] = None


def get_text_analyzer() -> TextSentimentAnalyzer:
    """Retrieve or initialise the shared TextSentimentAnalyzer instance."""
    global _text_analyzer
    if _text_analyzer is None:
        _text_analyzer = TextSentimentAnalyzer()
    return _text_analyzer


def get_scraper() -> ContentScraper:
    """Retrieve or initialise the shared ContentScraper instance."""
    global _scraper
    if _scraper is None:
        _scraper = ContentScraper()
    return _scraper


@bp.route("/", methods=["GET"])
def analyze_text():
    """Render the interactive text sentiment analysis workbench."""
    return render_template("analyze.html")


@bp.route("/batch", methods=["GET"])
def batch_upload():
    """Shortcut link to the batch processing module."""
    return redirect(url_for("batch.index"))


@bp.route("/text", methods=["POST"])
def analyze_text_submit():
    """
    Execute multi-model sentiment analysis on submitted text.
    Accepts application/json or form-encoded requests.
    """
    data = request.get_json(silent=True) or request.form.to_dict()
    text = (data.get("text") or "").strip()
    selected_models = data.get("models")

    if not text:
        return jsonify({"error": "No text provided for analysis."}), 400

    analyzer = get_text_analyzer()
    try:
        results = analyzer.analyze(text=text, models=selected_models)
        keywords = extract_keywords(text, top_n=8)
        entities = extract_entities(text)
        stats = get_text_statistics(text)
        lang_info = detect_language(text)

        user_id = current_user.id if current_user.is_authenticated else "anonymous"

        db = getattr(current_app, "db", None)
        saved_id = None
        if db is not None:
            analysis_record = Analysis(
                user_id=user_id,
                analysis_type="text",
                input_data=text[:1000],  # truncate if very large
                results=results,
                overall_sentiment=results.get("overall_sentiment", "neutral"),
                confidence=results.get("overall_confidence", 0.0),
                models_used=list(results.get("models", {}).keys()),
                language=lang_info.get("language", "en"),
                keywords=[k["keyword"] for k in keywords],
                entities=entities,
                metadata={"statistics": stats},
            )
            try:
                saved_id = analysis_record.save(db)
            except Exception as exc:
                logger.warning("Could not persist analysis record: %s", exc)

        response_payload = {
            "success": True,
            "id": saved_id,
            "results": results,
            "keywords": keywords,
            "entities": entities,
            "statistics": stats,
            "language": lang_info,
        }
        return jsonify(response_payload), 200

    except Exception as exc:
        logger.error("Text analysis error: %s", exc, exc_info=True)
        return jsonify({"error": f"Analysis execution failed: {str(exc)}"}), 500


@bp.route("/url", methods=["POST"])
def analyze_url_submit():
    """
    Scrape content from a public webpage or review URL and perform sentiment analysis.
    """
    data = request.get_json(silent=True) or request.form.to_dict()
    url = (data.get("url") or "").strip()

    if not url:
        return jsonify({"error": "Please enter a valid webpage URL."}), 400

    scraper = get_scraper()
    analyzer = get_text_analyzer()

    try:
        scrape_result = scraper.scrape_and_analyze(url, analyzer)
        if not scrape_result.get("success"):
            return jsonify({"error": scrape_result.get("error", "Failed to retrieve page content.")}), 400

        user_id = current_user.id if current_user.is_authenticated else "anonymous"
        db = getattr(current_app, "db", None)
        saved_id = None

        if db is not None:
            analysis_record = Analysis(
                user_id=user_id,
                analysis_type="url",
                input_data=url,
                results=scrape_result,
                overall_sentiment=scrape_result.get("overall_sentiment", "neutral"),
                confidence=scrape_result.get("overall_confidence", 0.0),
                models_used=["ensemble"],
                metadata={
                    "page_title": scrape_result.get("title", ""),
                    "item_count": scrape_result.get("extracted_count", 0),
                },
            )
            try:
                saved_id = analysis_record.save(db)
            except Exception as exc:
                logger.warning("Could not persist URL analysis record: %s", exc)

        scrape_result["id"] = saved_id
        return jsonify({"success": True, "data": scrape_result}), 200

    except Exception as exc:
        logger.error("URL analysis failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to scrape and analyze URL: {str(exc)}"}), 500


@bp.route("/aspect", methods=["POST"])
def analyze_aspects():
    """
    Extract specific aspects/topics from text and calculate granular aspect-based sentiment.
    """
    data = request.get_json(silent=True) or request.form.to_dict()
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400

    analyzer = get_text_analyzer()
    try:
        aspects = analyzer.get_aspect_sentiments(text)
        return jsonify({"success": True, "aspects": aspects}), 200
    except Exception as exc:
        logger.error("Aspect analysis error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@bp.route("/history", methods=["GET"])
def history():
    """
    Display paginated analysis history with filtering options.
    """
    db = getattr(current_app, "db", None)
    filter_type = request.args.get("type", "all")
    filter_sentiment = request.args.get("sentiment", "all")
    page = int(request.args.get("page", 1))
    per_page = 15

    analyses: List[Dict[str, Any]] = []
    total_count = 0

    if db is not None:
        try:
            # Open-access: show all analyses without authentication requirement
            query: Dict[str, Any] = {}
            if filter_type != "all":
                query["analysis_type"] = filter_type
            if filter_sentiment != "all":
                query["overall_sentiment"] = filter_sentiment

            total_count = db["analyses"].count_documents(query)
            cursor = (
                db["analyses"]
                .find(query)
                .sort("created_at", -1)
                .skip((page - 1) * per_page)
                .limit(per_page)
            )

            for doc in cursor:
                doc["id"] = str(doc.get("_id", ""))
                if "created_at" in doc and hasattr(doc["created_at"], "isoformat"):
                    doc["created_at"] = doc["created_at"].isoformat()
                analyses.append(doc)
        except Exception as exc:
            logger.warning("Error querying history: %s", exc)

    total_pages = max(1, (total_count + per_page - 1) // per_page)

    return render_template(
        "history.html",
        analyses=analyses,
        current_page=page,
        total_pages=total_pages,
        total_count=total_count,
        filter_type=filter_type,
        filter_sentiment=filter_sentiment,
    )


@bp.route("/result/<analysis_id>", methods=["GET"])
def view_result(analysis_id: str):
    """
    Display comprehensive report for a single analysis session.
    """
    db = getattr(current_app, "db", None)
    if db is None:
        flash("Database service offline.", "warning")
        return redirect(url_for("main.dashboard"))

    try:
        record = Analysis.find_by_id(db, analysis_id)
        if not record:
            flash("Analysis record not found.", "error")
            return redirect(url_for("analysis.history"))

        return render_template("report.html", analysis=record.to_dict())
    except Exception as exc:
        logger.error("Error retrieving analysis %s: %s", analysis_id, exc)
        flash("Unable to load analysis report.", "error")
        return redirect(url_for("analysis.history"))


@bp.route("/result/<analysis_id>", methods=["DELETE"])
def delete_result(analysis_id: str):
    """Delete an analysis record from the database."""
    db = getattr(current_app, "db", None)
    if db is None:
        return jsonify({"error": "Database not available"}), 503

    try:
        success = Analysis.delete(db, analysis_id)
        if success:
            return jsonify({"success": True}), 200
        return jsonify({"error": "Record not found or not deleted."}), 404
    except Exception as exc:
        logger.error("Error deleting record %s: %s", analysis_id, exc)
        return jsonify({"error": str(exc)}), 500
