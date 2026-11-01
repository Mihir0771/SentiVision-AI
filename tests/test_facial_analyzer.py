"""
Unit tests for facial emotion recognition engine.
"""

from __future__ import annotations

import pytest
from app.services.facial_analyzer import FacialEmotionAnalyzer


@pytest.fixture
def facial_analyzer() -> FacialEmotionAnalyzer:
    return FacialEmotionAnalyzer()


def test_facial_analyzer_initialization(facial_analyzer: FacialEmotionAnalyzer):
    """Test analyzer boots up and exposes supported emotion lists."""
    assert hasattr(facial_analyzer, "analyze_image_bytes")
    assert hasattr(facial_analyzer, "analyze_frame")
    assert hasattr(facial_analyzer, "draw_annotations")


def test_emotion_color_mapping(facial_analyzer: FacialEmotionAnalyzer):
    """Test that BGR color tuples are assigned to all 7 emotions."""
    for emotion in ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]:
        color = facial_analyzer.get_emotion_color(emotion)
        assert isinstance(color, tuple)
        assert len(color) == 3


def test_invalid_bytes_handling(facial_analyzer: FacialEmotionAnalyzer):
    """Test that random invalid bytes degrade gracefully rather than throwing uncaught crash."""
    result = facial_analyzer.analyze_image_bytes(b"invalid_non_image_bytes")
    assert "faces_detected" in result
    assert result["faces_detected"] == 0
    assert "faces" in result
    assert isinstance(result["faces"], list)


def test_facial_analyzer_on_portrait(facial_analyzer: FacialEmotionAnalyzer):
    """Test facial emotion analysis on a portrait image using YuNet + FERPlus."""
    import os
    if os.path.exists("lena.jpg"):
        result = facial_analyzer.analyze_image("lena.jpg")
        assert "faces_detected" in result
        assert result["faces_detected"] >= 1
        assert len(result["faces"]) >= 1
        face = result["faces"][0]
        assert "box" in face
        assert "emotions" in face
        assert "dominant_emotion" in face
        assert face["dominant_emotion"] in ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear"]
        assert face["confidence"] > 0
