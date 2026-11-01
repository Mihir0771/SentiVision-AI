"""
Integration tests for Flask frontend routes and HTML templates.
"""

from __future__ import annotations

import json
from flask.testing import FlaskClient


def test_dashboard_route(client: FlaskClient):
    """Test dashboard route returns 200 and loads HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"SentiVision" in response.data


def test_about_route(client: FlaskClient):
    """Test about page route returns 200."""
    response = client.get("/about")
    assert response.status_code == 200
    assert b"About" in response.data


def test_analyze_route(client: FlaskClient):
    """Test interactive text analysis page route."""
    response = client.get("/analyze/")
    assert response.status_code == 200
    assert b"Text Sentiment Analysis" in response.data


def test_camera_route(client: FlaskClient):
    """Test live camera page route."""
    response = client.get("/camera/")
    assert response.status_code == 200
    assert b"Live Facial Emotion" in response.data


def test_batch_route(client: FlaskClient):
    """Test batch tabular upload route."""
    response = client.get("/batch/")
    assert response.status_code == 200
    assert b"Batch Review Analyzer" in response.data


def test_api_docs_route(client: FlaskClient):
    """Test developer REST API docs route."""
    response = client.get("/api-docs")
    assert response.status_code == 200
    assert b"REST API Reference" in response.data


def test_auth_pages_routes(client: FlaskClient):
    """Test login and register view availability."""
    res_login = client.get("/auth/login")
    assert res_login.status_code == 200
    assert b"Sign In" in res_login.data

    res_reg = client.get("/auth/register")
    assert res_reg.status_code == 200
    assert b"Create Account" in res_reg.data


def test_health_check_endpoint(client: FlaskClient):
    """Test JSON health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"
    assert "services" in data


def test_404_handling(client: FlaskClient):
    """Test 404 response for non-existent route."""
    response = client.get("/route-that-does-not-exist")
    assert response.status_code == 404


def test_text_analysis_post(client: FlaskClient):
    """Test POST request to /analyze/text with JSON body."""
    payload = {"text": "This phone has great battery and excellent quality!", "models": ["vader", "textblob"]}
    response = client.post(
        "/analyze/text",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True
    assert "results" in data
    assert data["results"]["overall_sentiment"] == "positive"
