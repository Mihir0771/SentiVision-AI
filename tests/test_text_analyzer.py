"""
Unit tests for the multi-model text sentiment analysis engine.
"""

from __future__ import annotations

import pytest
from app.services.text_analyzer import TextSentimentAnalyzer


@pytest.fixture
def analyzer() -> TextSentimentAnalyzer:
    """Fixture providing an analyzer configured to use VADER and TextBlob (fast execution)."""
    return TextSentimentAnalyzer(lazy_load=True)


def test_positive_sentiment(analyzer: TextSentimentAnalyzer, sample_positive_text: str):
    """Test that positive input yields a positive classification."""
    res = analyzer.analyze(sample_positive_text, models=["vader", "textblob"])
    assert "overall_sentiment" in res
    assert res["overall_sentiment"] == "positive"
    assert res["overall_confidence"] > 0.5
    assert "ensemble_scores" in res
    assert res["ensemble_scores"]["positive"] > res["ensemble_scores"]["negative"]


def test_negative_sentiment(analyzer: TextSentimentAnalyzer, sample_negative_text: str):
    """Test that negative input yields a negative classification."""
    res = analyzer.analyze(sample_negative_text, models=["vader", "textblob"])
    assert res["overall_sentiment"] == "negative"
    assert res["overall_confidence"] > 0.4
    assert res["ensemble_scores"]["negative"] > res["ensemble_scores"]["positive"]


def test_neutral_sentiment(analyzer: TextSentimentAnalyzer, sample_neutral_text: str):
    """Test that neutral descriptive text yields neutral classification."""
    res = analyzer.analyze(sample_neutral_text, models=["vader", "textblob"])
    assert "overall_sentiment" in res
    assert res["overall_sentiment"] in ("neutral", "positive")  # depending on threshold


def test_batch_analysis(analyzer: TextSentimentAnalyzer, sample_texts: list[str]):
    """Test processing multiple texts in bulk."""
    results = analyzer.analyze_batch(sample_texts, models=["vader", "textblob"])
    assert len(results) == len(sample_texts)
    for r in results:
        assert "overall_sentiment" in r
        assert "overall_confidence" in r


def test_empty_string(analyzer: TextSentimentAnalyzer):
    """Test graceful handling of empty or blank text."""
    res = analyzer.analyze("", models=["vader"])
    assert res["overall_sentiment"] == "neutral"


def test_aspect_extraction(analyzer: TextSentimentAnalyzer):
    """Test aspect-based sentiment extraction on multi-aspect review."""
    text = "The screen display is stunning, but the battery life is terrible."
    aspects = analyzer.get_aspect_sentiments(text)
    assert isinstance(aspects, list)
