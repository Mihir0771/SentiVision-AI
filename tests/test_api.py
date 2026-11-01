"""
Integration tests for SentiVision REST API (v1).
"""

from __future__ import annotations

import json
from flask.testing import FlaskClient


def test_api_analyze_text(client: FlaskClient):
    """Test public API text analysis."""
    payload = {
        "text": "The laptop build is superb, lightweight, and super fast!",
        "models": ["vader", "textblob"],
    }
    response = client.post(
        "/api/v1/analyze/text",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "success"
    assert data["data"]["sentiment"]["overall_sentiment"] == "positive"
    assert "keywords" in data["data"]
    assert "statistics" in data["data"]


def test_api_analyze_empty_text_returns_400(client: FlaskClient):
    """Test validation when text is empty."""
    payload = {"text": "   "}
    response = client.post(
        "/api/v1/analyze/text",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data["status"] == "error"


def test_api_analyze_batch(client: FlaskClient):
    """Test batch API with array of reviews."""
    payload = {
        "texts": [
            "Great sound quality!",
            "Broken headphone jack, very upset.",
            "Color is silver.",
        ]
    }
    response = client.post(
        "/api/v1/analyze/batch",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "success"
    assert len(data["data"]["items"]) == 3
    assert "summary" in data["data"]
    assert data["data"]["summary"]["total_items"] == 3


def test_api_token_generation(client: FlaskClient):
    """Test JWT token generation endpoint."""
    payload = {"username": "test_developer", "password": "password123"}
    response = client.post(
        "/api/v1/auth/token",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code in (200, 401)
    if response.status_code == 200:
        data = json.loads(response.data)
        assert "access_token" in data
        assert data["token_type"] == "Bearer"


def test_open_endpoint_without_token_returns_200(client: FlaskClient):
    """Test that accessing /api/v1/analyses requires no token (open-source public API)."""
    response = client.get("/api/v1/analyses")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "success"
