"""
Pytest configuration and shared fixtures for SentiVision AI test suite.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Generator
import pytest
from flask import Flask
from flask.testing import FlaskClient

from app import create_app


@pytest.fixture(scope="session")
def app() -> Generator[Flask, None, None]:
    """Provide a configured test Flask application instance."""
    os.environ["FLASK_ENV"] = "testing"
    os.environ["SECRET_KEY"] = "test-secret-key"
    os.environ["JWT_SECRET_KEY"] = "test-jwt-secret-key"

    test_app = create_app("testing")
    test_app.config["TESTING"] = True
    test_app.config["WTF_CSRF_ENABLED"] = False

    yield test_app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Provide a test HTTP client."""
    return app.test_client()


@pytest.fixture
def sample_positive_text() -> str:
    return "This product is absolutely incredible, works flawlessly and exceeded all expectations!"


@pytest.fixture
def sample_negative_text() -> str:
    return "Terrible quality, broke after one day and support refused to help. Do not buy!"


@pytest.fixture
def sample_neutral_text() -> str:
    return "The parcel arrived on Tuesday at 3pm with standard packaging and documentation."


@pytest.fixture
def sample_texts() -> list[str]:
    return [
        "Outstanding performance and beautiful design!",
        "Poor customer service, very disappointed with the delay.",
        "The item comes in blue and black colors.",
    ]


@pytest.fixture
def sample_analysis_result() -> Dict[str, Any]:
    return {
        "text": "Great quality!",
        "overall_sentiment": "positive",
        "overall_confidence": 0.95,
        "ensemble_scores": {"positive": 0.90, "negative": 0.05, "neutral": 0.05},
        "models": {
            "transformer": {"label": "positive", "confidence": 0.96},
            "vader": {"compound": 0.65, "label": "positive", "pos": 0.6, "neg": 0.0, "neu": 0.4},
            "textblob": {"polarity": 0.8, "subjectivity": 0.75, "label": "positive"},
        },
    }
