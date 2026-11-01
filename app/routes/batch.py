"""
Batch processing blueprint — handles CSV, JSON, and Excel bulk review uploads.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, Optional
from flask import (
    Blueprint,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
)
from flask_login import current_user
from werkzeug.utils import secure_filename

from app.models.analysis import Analysis
from app.services.batch_processor import BatchProcessor
from app.services.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

bp = Blueprint("batch", __name__, template_folder="../templates")

_batch_processor: Optional[BatchProcessor] = None
_report_generator: Optional[ReportGenerator] = None

ALLOWED_EXTENSIONS = {"csv", "json", "xlsx", "xls"}


def get_batch_processor() -> BatchProcessor:
    """Retrieve or initialise the shared BatchProcessor instance."""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
    return _batch_processor


def get_report_generator() -> ReportGenerator:
    """Retrieve or initialise the shared ReportGenerator instance."""
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator


def allowed_file(filename: str) -> bool:
    """Check if file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route("/", methods=["GET"])
def index():
    """Render batch review upload & analysis page."""
    return render_template("batch.html")


@bp.route("/upload", methods=["POST"])
def upload_file():
    """
    Handle batch file upload. Supports full processing or previewing columns.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Unsupported file format. Please upload CSV, JSON, or Excel files."
        }), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    filepath = os.path.join(upload_folder, unique_name)

    try:
        file.save(filepath)
    except Exception as exc:
        logger.error("Failed to save uploaded file: %s", exc)
        return jsonify({"error": "Could not save file to server."}), 500

    processor = get_batch_processor()
    preview_only = request.form.get("preview") == "true"
    text_column = request.form.get("text_column") or None

    try:
        # If user just wants preview of columns & top rows
        if preview_only:
            preview_data = processor.get_file_preview(filepath)
            return jsonify({
                "success": True,
                "preview": True,
                "filepath": unique_name,
                "columns": preview_data.get("columns", []),
                "detected_column": preview_data.get("detected_column"),
                "sample_rows": preview_data.get("sample_rows", []),
                "total_rows": preview_data.get("total_rows", 0),
            }), 200

        # Run complete batch analysis
        batch_results = processor.process_file(filepath=filepath, text_column=text_column)

        user_id = current_user.id if current_user.is_authenticated else "anonymous"
        db = getattr(current_app, "db", None)
        saved_id = None

        summary = batch_results.get("summary", {})
        dominant_sentiment = "neutral"
        if summary.get("positive_count", 0) > summary.get("negative_count", 0):
            dominant_sentiment = "positive"
        elif summary.get("negative_count", 0) > summary.get("positive_count", 0):
            dominant_sentiment = "negative"

        if db is not None:
            analysis_record = Analysis(
                user_id=user_id,
                analysis_type="batch",
                input_data=f"Batch File: {filename} ({batch_results.get('total_items', 0)} rows)",
                results=batch_results,
                overall_sentiment=dominant_sentiment,
                confidence=summary.get("avg_confidence", 0.0),
                models_used=["ensemble"],
                metadata={
                    "filename": filename,
                    "processed_rows": batch_results.get("processed", 0),
                    "stored_filename": unique_name,
                },
            )
            try:
                saved_id = analysis_record.save(db)
            except Exception as exc:
                logger.warning("Could not persist batch analysis: %s", exc)

        batch_results["id"] = saved_id
        batch_results["filename"] = filename
        batch_results["file_key"] = unique_name

        return jsonify({"success": True, "data": batch_results}), 200

    except Exception as exc:
        logger.error("Batch processing error: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to process batch file: {str(exc)}"}), 500


@bp.route("/results/<analysis_id>", methods=["GET"])
def view_batch_results(analysis_id: str):
    """View details of a specific batch analysis."""
    db = getattr(current_app, "db", None)
    if db is None:
        return jsonify({"error": "Database unavailable"}), 503

    record = Analysis.find_by_id(db, analysis_id)
    if not record:
        return jsonify({"error": "Batch record not found"}), 404

    return jsonify({"success": True, "data": record.to_dict()}), 200


@bp.route("/export/<analysis_id>/<file_format>", methods=["GET"])
def export_batch(analysis_id: str, file_format: str):
    """
    Export batch analysis items into CSV or JSON format for download.
    """
    db = getattr(current_app, "db", None)
    if db is None:
        return jsonify({"error": "Database unavailable"}), 503

    record = Analysis.find_by_id(db, analysis_id)
    if not record:
        return jsonify({"error": "Record not found"}), 404

    processor = get_batch_processor()
    results = record.results.get("results", [])

    try:
        exported_path = processor.export_results(results=record.results, format=file_format)
        if not os.path.exists(exported_path):
            return jsonify({"error": "Export file could not be generated."}), 500

        mimetype = "text/csv" if file_format == "csv" else "application/json"
        download_name = f"sentivision_export_{analysis_id[:8]}.{file_format}"
        return send_file(exported_path, mimetype=mimetype, as_attachment=True, download_name=download_name)
    except Exception as exc:
        logger.error("Export failure: %s", exc)
        return jsonify({"error": str(exc)}), 500
