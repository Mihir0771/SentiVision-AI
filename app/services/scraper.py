"""
Web scraper for extracting reviews and comments from URLs.

Provides general-purpose HTML text extraction via BeautifulSoup + requests,
with User-Agent rotation, rate limiting, and integrated sentiment analysis.
"""

from __future__ import annotations

import logging
import random
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    import requests as _requests  # type: ignore

    _REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _REQUESTS_AVAILABLE = False
    logger.warning("requests not installed – scraping will be unavailable.")

try:
    from bs4 import BeautifulSoup as _BS  # type: ignore

    _BS4_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BS4_AVAILABLE = False
    logger.warning("beautifulsoup4 not installed – scraping will be unavailable.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
]

_DEFAULT_TIMEOUT = 15  # seconds
_MIN_REQUEST_INTERVAL = 1.0  # seconds between requests (rate limit)

# Tags whose text content is generally meaningful
_CONTENT_TAGS = {"p", "article", "blockquote", "li", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "span", "div"}
_REVIEW_CLASSES_RE = re.compile(
    r"review|comment|feedback|testimonial|opinion|rating",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _random_ua() -> str:
    """Return a random User-Agent string."""
    return random.choice(_USER_AGENTS)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ContentScraper:
    """General-purpose web content scraper with rate limiting.

    Parameters
    ----------
    timeout : int, optional
        HTTP request timeout in seconds (default 15).
    min_interval : float, optional
        Minimum seconds between consecutive HTTP requests (rate limiting).
    """

    def __init__(
        self,
        timeout: int = _DEFAULT_TIMEOUT,
        min_interval: float = _MIN_REQUEST_INTERVAL,
    ) -> None:
        self._timeout = timeout
        self._min_interval = min_interval
        self._last_request_time: float = 0.0
        logger.info(
            "ContentScraper initialised (timeout=%ds, interval=%.1fs).",
            timeout,
            min_interval,
        )

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Block until the minimum inter-request interval has elapsed."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    # ------------------------------------------------------------------
    # Core scraping
    # ------------------------------------------------------------------

    def scrape_url(self, url: str) -> Dict[str, Any]:
        """Fetch and extract text content from a URL.

        Parameters
        ----------
        url : str
            The target URL.

        Returns
        -------
        dict
            ``{"url", "title", "texts", "extracted_count", "scraped_at"}``
            or an ``"error"`` key on failure.
        """
        if not _REQUESTS_AVAILABLE or not _BS4_AVAILABLE:
            return {
                "url": url,
                "error": "requests and/or beautifulsoup4 are not installed.",
                "texts": [],
                "extracted_count": 0,
                "scraped_at": _timestamp(),
            }

        # Validate URL
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return {
                "url": url,
                "error": f"Unsupported URL scheme: {parsed.scheme!r}",
                "texts": [],
                "extracted_count": 0,
                "scraped_at": _timestamp(),
            }

        self._rate_limit()

        try:
            headers = {"User-Agent": _random_ua()}
            response = _requests.get(url, headers=headers, timeout=self._timeout)
            response.raise_for_status()
        except _requests.exceptions.Timeout:
            logger.error("Timeout fetching %s", url)
            return {"url": url, "error": "Request timed out.", "texts": [], "extracted_count": 0, "scraped_at": _timestamp()}
        except _requests.exceptions.HTTPError as exc:
            logger.error("HTTP error for %s: %s", url, exc)
            return {"url": url, "error": f"HTTP {exc.response.status_code}", "texts": [], "extracted_count": 0, "scraped_at": _timestamp()}
        except _requests.exceptions.RequestException as exc:
            logger.error("Request failed for %s: %s", url, exc)
            return {"url": url, "error": str(exc), "texts": [], "extracted_count": 0, "scraped_at": _timestamp()}

        html = response.text
        title = self._extract_title(html)
        texts = self.extract_text_content(html)

        return {
            "url": url,
            "title": title,
            "texts": texts,
            "extracted_count": len(texts),
            "scraped_at": _timestamp(),
        }

    # ------------------------------------------------------------------
    # HTML parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_title(html: str) -> str:
        """Extract the ``<title>`` text from raw HTML."""
        if not _BS4_AVAILABLE:
            return ""
        try:
            soup = _BS(html, "html.parser")
            tag = soup.find("title")
            return tag.get_text(strip=True) if tag else ""
        except Exception:
            return ""

    def extract_text_content(self, html: str) -> List[str]:
        """Extract meaningful text paragraphs and review-like blocks from HTML.

        Parameters
        ----------
        html : str
            Raw HTML string.

        Returns
        -------
        list of str
            Cleaned text segments (minimum 20 chars each).
        """
        if not _BS4_AVAILABLE:
            return []

        try:
            soup = _BS(html, "html.parser")

            # Remove non-content tags
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe", "svg"]):
                tag.decompose()

            texts: List[str] = []
            seen: set = set()

            # 1. Prioritise elements with review-like class/id names
            for el in soup.find_all(attrs={"class": _REVIEW_CLASSES_RE}):
                cleaned = self._clean_text(el.get_text(separator=" "))
                if len(cleaned) >= 20 and cleaned not in seen:
                    texts.append(cleaned)
                    seen.add(cleaned)

            for el in soup.find_all(attrs={"id": _REVIEW_CLASSES_RE}):
                cleaned = self._clean_text(el.get_text(separator=" "))
                if len(cleaned) >= 20 and cleaned not in seen:
                    texts.append(cleaned)
                    seen.add(cleaned)

            # 2. General content tags
            for tag in soup.find_all(_CONTENT_TAGS):
                cleaned = self._clean_text(tag.get_text(separator=" "))
                if len(cleaned) >= 20 and cleaned not in seen:
                    texts.append(cleaned)
                    seen.add(cleaned)

            return texts

        except Exception as exc:
            logger.error("Text extraction failed: %s", exc)
            return []

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean extracted text by collapsing whitespace and stripping HTML entities.

        Parameters
        ----------
        text : str
            Raw extracted string.

        Returns
        -------
        str
            Cleaned string.
        """
        import html as _html_mod

        text = _html_mod.unescape(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Integrated analysis
    # ------------------------------------------------------------------

    def scrape_and_analyze(
        self,
        url: str,
        analyzer: Any,
    ) -> Dict[str, Any]:
        """Scrape a URL and run sentiment analysis on each extracted text.

        Parameters
        ----------
        url : str
            Target URL.
        analyzer : TextSentimentAnalyzer
            Analyser instance with an ``analyze(text)`` method.

        Returns
        -------
        dict
            Scrape metadata plus per-text sentiment results and overall
            summary.
        """
        scrape_result = self.scrape_url(url)

        if "error" in scrape_result:
            return scrape_result

        texts = scrape_result.get("texts", [])
        analyses: List[Dict[str, Any]] = []
        sentiments = {"positive": 0, "negative": 0, "neutral": 0}

        for text in texts:
            try:
                result = analyzer.analyze(text)
                sentiment = result.get("overall_sentiment", "neutral")
                sentiments[sentiment] = sentiments.get(sentiment, 0) + 1
                analyses.append(
                    {
                        "text": text[:300],
                        "sentiment": sentiment,
                        "confidence": result.get("overall_confidence", 0.0),
                        "scores": result.get("ensemble_scores", {}),
                    }
                )
            except Exception as exc:
                logger.error("Analysis failed for scraped text: %s", exc)
                analyses.append({"text": text[:300], "error": str(exc)})

        total = max(len(analyses), 1)
        return {
            "url": scrape_result.get("url", url),
            "title": scrape_result.get("title", ""),
            "extracted_count": scrape_result.get("extracted_count", 0),
            "analyses": analyses,
            "summary": {
                "positive_count": sentiments["positive"],
                "negative_count": sentiments["negative"],
                "neutral_count": sentiments["neutral"],
                "positive_pct": round(sentiments["positive"] / total * 100, 2),
                "negative_pct": round(sentiments["negative"] / total * 100, 2),
                "neutral_pct": round(sentiments["neutral"] / total * 100, 2),
            },
            "scraped_at": scrape_result.get("scraped_at", _timestamp()),
        }
