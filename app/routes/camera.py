"""
Camera and facial emotion detection blueprint.
Supports live webcam streaming via WebSocket and snapshot/upload analysis via REST.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, Optional
from flask import Blueprint, current_app, jsonify, render_template, request
from flask_login import current_user
from flask_socketio import emit

from app import socketio
from app.models.analysis import Analysis
from app.services.facial_analyzer import FacialEmotionAnalyzer

logger = logging.getLogger(__name__)

bp = Blueprint("camera", __name__, template_folder="../templates")

_facial_analyzer: Optional[FacialEmotionAnalyzer] = None


def get_facial_analyzer() -> FacialEmotionAnalyzer:
    """Retrieve or initialise the shared FacialEmotionAnalyzer instance."""
    global _facial_analyzer
    if _facial_analyzer is None:
        _facial_analyzer = FacialEmotionAnalyzer()
    return _facial_analyzer


@bp.route("/", methods=["GET"])
def live_camera():
    """Render live webcam facial sentiment analysis view."""
    return render_template("camera.html")


@bp.route("/snapshot", methods=["POST"])
def analyze_snapshot():
    """
    Analyze a captured video frame sent as base64 string.
    """
    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image", "")

    if not image_b64:
        return jsonify({"error": "No image data received."}), 400

    # Strip data URL header if present (e.g. data:image/jpeg;base64,...)
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(image_b64)
        analyzer = get_facial_analyzer()
        analysis_result = analyzer.analyze_image_bytes(raw_bytes)

        user_id = current_user.id if current_user.is_authenticated else "anonymous"
        db = getattr(current_app, "db", None)
        saved_id = None

        dominant = "neutral"
        confidence = 0.0
        faces = analysis_result.get("faces", [])
        if faces:
            dominant = faces[0].get("dominant_emotion", "neutral")
            confidence = faces[0].get("confidence", 0.0)

        save_requested = data.get("save", False)
        if db is not None and analysis_result.get("faces_detected", 0) > 0 and save_requested:
            rec = Analysis(
                user_id=user_id,
                analysis_type="facial",
                input_data="Webcam snapshot capture",
                results=analysis_result,
                overall_sentiment=dominant,
                confidence=confidence,
                models_used=["fer_facial_model"],
                metadata={"faces_count": len(faces)},
            )
            try:
                saved_id = rec.save(db)
            except Exception as exc:
                logger.warning("Could not persist facial snapshot: %s", exc)

        analysis_result["id"] = saved_id
        return jsonify({"success": True, "data": analysis_result}), 200

    except Exception as exc:
        logger.error("Snapshot analysis failure: %s", exc, exc_info=True)
        return jsonify({"error": f"Failed to analyze frame: {str(exc)}"}), 500


@bp.route("/upload-image", methods=["POST"])
def upload_image():
    """
    Handle static image upload for facial emotion detection.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400

    try:
        raw_bytes = file.read()
        analyzer = get_facial_analyzer()
        analysis_result = analyzer.analyze_image_bytes(raw_bytes)

        user_id = current_user.id if current_user.is_authenticated else "anonymous"
        db = getattr(current_app, "db", None)
        saved_id = None

        dominant = "neutral"
        confidence = 0.0
        faces = analysis_result.get("faces", [])
        if faces:
            dominant = faces[0].get("dominant_emotion", "neutral")
            confidence = faces[0].get("confidence", 0.0)

        if db is not None and analysis_result.get("faces_detected", 0) > 0:
            rec = Analysis(
                user_id=user_id,
                analysis_type="facial",
                input_data=f"Uploaded photo: {file.filename}",
                results=analysis_result,
                overall_sentiment=dominant,
                confidence=confidence,
                models_used=["fer_facial_model"],
                metadata={"filename": file.filename, "faces_count": len(faces)},
            )
            try:
                saved_id = rec.save(db)
            except Exception as exc:
                logger.warning("Could not persist uploaded image record: %s", exc)

        analysis_result["id"] = saved_id
        return jsonify({"success": True, "data": analysis_result}), 200

    except Exception as exc:
        logger.error("Image upload analysis failed: %s", exc, exc_info=True)
        return jsonify({"error": f"Facial detection failed: {str(exc)}"}), 500


# ---------------------------------------------------------------------------
# WebSocket Event Handlers for Real-Time Streaming
# ---------------------------------------------------------------------------

@socketio.on("connect", namespace="/camera")
def on_camera_connect():
    """Client connected to camera stream."""
    logger.debug("SocketIO client connected to /camera namespace.")
    emit("status", {"message": "Connected to facial emotion detection stream."})


@socketio.on("disconnect", namespace="/camera")
def on_camera_disconnect():
    """Client disconnected from camera stream."""
    logger.debug("SocketIO client disconnected from /camera namespace.")


@socketio.on("process_frame", namespace="/camera")
def on_process_frame(data: Dict[str, Any]):
    """
    Receive video frame from client, analyze emotions, and return bounding boxes + emotions.
    """
    image_b64 = data.get("frame", "")
    if not image_b64:
        return

    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(image_b64)
        analyzer = get_facial_analyzer()
        result = analyzer.analyze_image_bytes(raw_bytes)
        emit("emotion_result", result)
    except Exception as exc:
        logger.debug("Frame stream processing error: %s", exc)
        emit("emotion_error", {"error": str(exc)})
