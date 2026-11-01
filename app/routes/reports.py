"""
Reports and analytics blueprint.
Provides chart generation, comparison analysis, PDF reports, and data exports.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_login import current_user

from app.models.analysis import Analysis
from app.services.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

bp = Blueprint("reports", __name__, template_folder="../templates")

_report_generator: Optional[ReportGenerator] = None


def get_report_generator() -> ReportGenerator:
    """Retrieve or initialise the shared ReportGenerator instance."""
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator


@bp.route("/", methods=["GET"])
def index():
    """Render reports and analytics dashboard."""
    db = getattr(current_app, "db", None)
    analyses: List[Dict[str, Any]] = []

    if db is not None:
        try:
            records = Analysis.find_by_user(db, None, limit=30)
            analyses = [r.to_dict() for r in records]
        except Exception as exc:
            logger.warning("Error fetching analyses for reports: %s", exc)

    return render_template("report.html", analyses=analyses, analysis=analyses[0] if analyses else None)


@bp.route("/chart/<analysis_id>/<chart_type>", methods=["GET"])
def get_chart(analysis_id: str, chart_type: str):
    """
    Generate and stream an image representation (PNG) of sentiment or emotion metrics.
    chart_type: 'pie', 'bar', 'radar', or 'timeline'
    """
    db = getattr(current_app, "db", None)
    if db is None:
        return jsonify({"error": "Database unavailable"}), 503

    record = Analysis.find_by_id(db, analysis_id)
    if not record:
        return jsonify({"error": "Analysis not found"}), 404

    generator = get_report_generator()
    try:
        if chart_type == "emotion" or record.analysis_type == "facial":
            faces = record.results.get("faces", [])
            emotions = faces[0].get("emotions", {}) if faces else {}
            chart_path = generator.generate_emotion_chart(emotions)
        else:
            chart_path = generator.generate_sentiment_chart(record.results, chart_type=chart_type)

        if not os.path.exists(chart_path):
            return jsonify({"error": "Chart generation failed."}), 500

        return send_file(chart_path, mimetype="image/png")
    except Exception as exc:
        logger.error("Chart generation error for %s (%s): %s", analysis_id, chart_type, exc)
        return jsonify({"error": str(exc)}), 500


@bp.route("/download/<analysis_id>", methods=["GET"])
def download_report(analysis_id: str):
    """
    Create and stream an export report for an individual analysis.
    """
    db = getattr(current_app, "db", None)
    if db is None:
        flash("Database unavailable.", "warning")
        return redirect(url_for("analysis.history"))

    record = Analysis.find_by_id(db, analysis_id)
    if not record:
        flash("Analysis not found.", "error")
        return redirect(url_for("analysis.history"))

    generator = get_report_generator()
    try:
        report_path = generator.create_pdf_report(record.to_dict())
        if not os.path.exists(report_path):
            flash("Could not generate PDF report.", "error")
            return redirect(url_for("analysis.view_result", analysis_id=analysis_id))

        mimetype = "application/pdf" if report_path.endswith(".pdf") else "text/html"
        filename = f"sentivision_report_{analysis_id[:8]}.{'pdf' if mimetype == 'application/pdf' else 'html'}"
        return send_file(report_path, mimetype=mimetype, as_attachment=True, download_name=filename)

    except Exception as exc:
        logger.error("Download report failure: %s", exc)
        flash(f"Report export error: {str(exc)}", "error")
        return redirect(url_for("analysis.view_result", analysis_id=analysis_id))


@bp.route("/compare", methods=["POST"])
def compare():
    """
    Compare multiple analyses side-by-side and return comparative metrics.
    """
    data = request.get_json(silent=True) or request.form.to_dict()
    ids = data.get("ids", [])
    if isinstance(ids, str):
        ids = [i.strip() for i in ids.split(",") if i.strip()]

    if len(ids) < 2:
        return jsonify({"error": "Please provide at least 2 analysis IDs for comparison."}), 400

    db = getattr(current_app, "db", None)
    if db is None:
        return jsonify({"error": "Database unavailable"}), 503

    records: List[Dict[str, Any]] = []
    for aid in ids:
        r = Analysis.find_by_id(db, aid)
        if r:
            records.append(r.to_dict())

    if len(records) < 2:
        return jsonify({"error": "Could not locate specified analysis records for comparison."}), 404

    generator = get_report_generator()
    try:
        comp_chart = generator.generate_comparison_chart(records)
        return jsonify({
            "success": True,
            "compared_count": len(records),
            "records": records,
            "chart_url": comp_chart,
        }), 200
    except Exception as exc:
        logger.error("Comparison analysis error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@bp.route("/export/csv", methods=["GET"])
def export_all_csv():
    """Export all analyses to CSV."""
    db = getattr(current_app, "db", None)
    if db is None:
        return jsonify({"error": "Database unavailable"}), 503

    records = Analysis.find_by_user(db, None, limit=1000)
    data = [r.to_dict() for r in records]

    generator = get_report_generator()
    out_dir = os.path.join(os.getcwd(), "temp", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sentivision_all_analyses.csv")
    generator.export_to_csv(data, out_path)

    return send_file(out_path, mimetype="text/csv", as_attachment=True, download_name="sentivision_analyses.csv")


@bp.route("/export/json", methods=["GET"])
def export_all_json():
    """Export all analyses to JSON."""
    db = getattr(current_app, "db", None)
    if db is None:
        return jsonify({"error": "Database unavailable"}), 503

    records = Analysis.find_by_user(db, None, limit=1000)
    data = [r.to_dict() for r in records]

    generator = get_report_generator()
    out_dir = os.path.join(os.getcwd(), "temp", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sentivision_all_analyses.json")
    generator.export_to_json(data, out_path)

    return send_file(out_path, mimetype="application/json", as_attachment=True, download_name="sentivision_analyses.json")
