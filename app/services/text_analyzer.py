"""
Multi-model text sentiment analysis engine.

Combines HuggingFace Transformers, VADER, and TextBlob for ensemble sentiment
scoring. Supports single-text analysis, batch processing, and aspect-level
sentiment extraction.

Models
------
- **Transformer**: ``cardiffnlp/twitter-roberta-base-sentiment-latest`` via the
  HuggingFace ``transformers`` pipeline (lazy-loaded on first use).
- **VADER**: Rule-based sentiment from ``vaderSentiment``.
- **TextBlob**: Pattern-based polarity / subjectivity from ``textblob``.

The ensemble method combines normalised scores from each available model using
configurable weights (default: transformer 0.5, VADER 0.3, TextBlob 0.2).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VaderAnalyzer

    _VADER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _VADER_AVAILABLE = False
    logger.warning("vaderSentiment not installed – VADER model will be unavailable.")

try:
    from textblob import TextBlob as _TextBlob

    _TEXTBLOB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TEXTBLOB_AVAILABLE = False
    logger.warning("textblob not installed – TextBlob model will be unavailable.")

try:
    from transformers import pipeline as _hf_pipeline

    _TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "transformers not installed – HuggingFace transformer model will be unavailable."
    )

# ---------------------------------------------------------------------------
# NLTK bootstrap (needed for aspect extraction)
# ---------------------------------------------------------------------------

_NLTK_READY = False


def _ensure_nltk() -> bool:
    """Download required NLTK data if not already present."""
    global _NLTK_READY
    if _NLTK_READY:
        return True
    try:
        import nltk

        for resource in ("punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
            try:
                nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"taggers/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)
        _NLTK_READY = True
        return True
    except Exception as exc:  # pragma: no cover
        logger.warning("NLTK data download failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRANSFORMER_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "transformer": 0.5,
    "vader": 0.3,
    "textblob": 0.2,
}

_ALL_MODELS = ("transformer", "vader", "textblob")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _label_from_scores(scores: Dict[str, float]) -> str:
    """Return the label key with the highest score."""
    return max(scores, key=scores.get)  # type: ignore[arg-type]


def _timestamp() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class TextSentimentAnalyzer:
    """Ensemble text sentiment analyser backed by three NLP models.

    Parameters
    ----------
    weights : dict, optional
        Custom model weights for the ensemble.  Keys must be a subset of
        ``{"transformer", "vader", "textblob"}``.  Weights are automatically
        re-normalised so they sum to 1.0.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None, lazy_load: bool = True, **kwargs: Any) -> None:
        self._weights: Dict[str, float] = dict(weights or _DEFAULT_WEIGHTS)
        self._transformer_pipeline: Any = None  # lazy-loaded
        self._vader: Any = None
        self._init_vader()
        logger.info("TextSentimentAnalyzer initialised (weights=%s).", self._weights)

    # ------------------------------------------------------------------
    # Lazy-loading helpers
    # ------------------------------------------------------------------

    def _init_vader(self) -> None:
        """Eagerly initialise VADER (it's lightweight)."""
        if _VADER_AVAILABLE:
            try:
                self._vader = _VaderAnalyzer()
            except Exception as exc:  # pragma: no cover
                logger.error("Failed to initialise VADER: %s", exc)

    def _get_transformer(self) -> Any:
        """Return the HuggingFace sentiment pipeline, loading it on first call."""
        if self._transformer_pipeline is not None:
            return self._transformer_pipeline
        if not _TRANSFORMERS_AVAILABLE:
            return None
        try:
            logger.info("Loading transformer model '%s' …", _TRANSFORMER_MODEL)
            self._transformer_pipeline = _hf_pipeline(
                "sentiment-analysis",
                model=_TRANSFORMER_MODEL,
                top_k=None,
                truncation=True,
            )
            logger.info("Transformer model loaded successfully.")
            return self._transformer_pipeline
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to load transformer model: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Individual model analyses
    # ------------------------------------------------------------------

    def _analyze_transformer(self, text: str) -> Optional[Dict[str, Any]]:
        """Run sentiment analysis via the HuggingFace transformer model.

        Returns
        -------
        dict or None
            ``{"label": str, "scores": {…}, "confidence": float}``
        """
        pipe = self._get_transformer()
        if pipe is None:
            return None
        try:
            outputs = pipe(text[:512])  # limit token length
            # outputs is [[{"label": …, "score": …}, …]]
            raw = outputs[0] if isinstance(outputs[0], list) else outputs
            # Normalise labels → positive / negative / neutral
            label_map = {
                "positive": "positive",
                "negative": "negative",
                "neutral": "neutral",
                "LABEL_0": "negative",
                "LABEL_1": "neutral",
                "LABEL_2": "positive",
            }
            scores: Dict[str, float] = {}
            for item in raw:
                mapped = label_map.get(item["label"], item["label"].lower())
                scores[mapped] = round(float(item["score"]), 4)

            dominant = _label_from_scores(scores)
            return {
                "label": dominant,
                "scores": scores,
                "confidence": round(scores[dominant], 4),
            }
        except Exception as exc:
            logger.error("Transformer analysis failed: %s", exc)
            return None

    def _analyze_vader(self, text: str) -> Optional[Dict[str, Any]]:
        """Run VADER sentiment analysis.

        Returns
        -------
        dict or None
            ``{"compound": float, "pos": float, "neg": float, "neu": float, "label": str}``
        """
        if self._vader is None:
            return None
        try:
            scores = self._vader.polarity_scores(text)
            compound = scores["compound"]
            if compound >= 0.05:
                label = "positive"
            elif compound <= -0.05:
                label = "negative"
            else:
                label = "neutral"
            return {
                "compound": round(compound, 4),
                "pos": round(scores["pos"], 4),
                "neg": round(scores["neg"], 4),
                "neu": round(scores["neu"], 4),
                "label": label,
            }
        except Exception as exc:
            logger.error("VADER analysis failed: %s", exc)
            return None

    def _analyze_textblob(self, text: str) -> Optional[Dict[str, Any]]:
        """Run TextBlob sentiment analysis.

        Returns
        -------
        dict or None
            ``{"polarity": float, "subjectivity": float, "label": str}``
        """
        if not _TEXTBLOB_AVAILABLE:
            return None
        try:
            blob = _TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"
            return {
                "polarity": round(polarity, 4),
                "subjectivity": round(subjectivity, 4),
                "label": label,
            }
        except Exception as exc:
            logger.error("TextBlob analysis failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------

    def _ensemble(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine individual model results into an ensemble score.

        Missing models are excluded and weights are re-normalised.

        Returns
        -------
        dict
            ``{"positive": float, "negative": float, "neutral": float}``
        """
        combined: Dict[str, float] = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        total_weight = 0.0

        for model_name, result in results.items():
            if result is None:
                continue
            w = self._weights.get(model_name, 0.0)
            if w <= 0:
                continue

            # Normalise each model's output to {positive, negative, neutral} scores
            if model_name == "transformer":
                scores = result.get("scores", {})
            elif model_name == "vader":
                scores = {
                    "positive": result.get("pos", 0.0),
                    "negative": result.get("neg", 0.0),
                    "neutral": result.get("neu", 0.0),
                }
            elif model_name == "textblob":
                polarity = result.get("polarity", 0.0)
                # Map [-1, 1] polarity → positive/negative/neutral distribution
                if polarity > 0:
                    scores = {
                        "positive": min(polarity, 1.0),
                        "negative": 0.0,
                        "neutral": 1.0 - min(polarity, 1.0),
                    }
                elif polarity < 0:
                    scores = {
                        "positive": 0.0,
                        "negative": min(abs(polarity), 1.0),
                        "neutral": 1.0 - min(abs(polarity), 1.0),
                    }
                else:
                    scores = {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            else:
                continue

            for key in combined:
                combined[key] += w * scores.get(key, 0.0)
            total_weight += w

        # Re-normalise by total weight used
        if total_weight > 0:
            for key in combined:
                combined[key] = round(combined[key] / total_weight, 4)

        return combined

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        text: str,
        models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Analyse a single text with specified (or all) models.

        Parameters
        ----------
        text : str
            The input text to analyse.
        models : list of str, optional
            Subset of ``["transformer", "vader", "textblob"]``.  Defaults to
            all available models.

        Returns
        -------
        dict
            Full analysis result including per-model outputs and ensemble
            scores.
        """
        if not text or not text.strip():
            return {
                "text": text,
                "overall_sentiment": "neutral",
                "overall_confidence": 0.0,
                "ensemble_scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                "models": {},
                "analyzed_at": _timestamp(),
                "error": "Empty or whitespace-only input.",
            }

        use_models = models or list(_ALL_MODELS)
        model_results: Dict[str, Any] = {}

        if "transformer" in use_models:
            model_results["transformer"] = self._analyze_transformer(text)
        if "vader" in use_models:
            model_results["vader"] = self._analyze_vader(text)
        if "textblob" in use_models:
            model_results["textblob"] = self._analyze_textblob(text)

        ensemble_scores = self._ensemble(model_results)
        overall_sentiment = _label_from_scores(ensemble_scores)
        overall_confidence = round(ensemble_scores.get(overall_sentiment, 0.0), 4)

        return {
            "text": text,
            "overall_sentiment": overall_sentiment,
            "overall_confidence": overall_confidence,
            "ensemble_scores": ensemble_scores,
            "models": {k: v for k, v in model_results.items() if v is not None},
            "analyzed_at": _timestamp(),
        }

    def analyze_batch(
        self,
        texts: List[str],
        models: Optional[List[str]] = None,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Analyse a list of texts sequentially.

        Parameters
        ----------
        texts : list of str
            Texts to analyse.
        models : list of str, optional
            Models to use.
        callback : callable, optional
            Progress callback ``(current_index, total)`` called after each item.

        Returns
        -------
        list of dict
            One analysis result per input text.
        """
        results: List[Dict[str, Any]] = []
        total = len(texts)
        for idx, text in enumerate(texts):
            try:
                results.append(self.analyze(text, models=models))
            except Exception as exc:
                logger.error("Batch item %d failed: %s", idx, exc)
                results.append(
                    {
                        "text": text,
                        "overall_sentiment": "neutral",
                        "overall_confidence": 0.0,
                        "ensemble_scores": {},
                        "models": {},
                        "analyzed_at": _timestamp(),
                        "error": str(exc),
                    }
                )
            if callback:
                try:
                    callback(idx + 1, total)
                except Exception:
                    pass
        return results

    # ------------------------------------------------------------------
    # Aspect-level sentiment
    # ------------------------------------------------------------------

    def get_aspect_sentiments(self, text: str) -> List[Dict[str, Any]]:
        """Extract noun-phrase aspects and compute per-aspect sentiment.

        Each sentence containing a recognised noun phrase is analysed
        independently.  The result is a list of aspects with their
        surrounding sentence's sentiment.

        Parameters
        ----------
        text : str
            The input text.

        Returns
        -------
        list of dict
            ``[{"aspect": str, "sentence": str, "sentiment": str, "confidence": float}]``
        """
        if not _ensure_nltk():
            logger.warning("NLTK unavailable – falling back to sentence-level aspects.")
            return self._aspect_fallback(text)

        try:
            import nltk

            sentences = nltk.sent_tokenize(text)
            aspects: List[Dict[str, Any]] = []
            seen_aspects: set = set()

            for sentence in sentences:
                tokens = nltk.word_tokenize(sentence)
                tagged = nltk.pos_tag(tokens)

                # Simple noun-phrase extraction: sequences of JJ* NN*
                noun_phrases: List[str] = []
                current_np: List[str] = []
                for word, tag in tagged:
                    if tag.startswith("NN") or tag.startswith("JJ"):
                        current_np.append(word)
                    else:
                        if current_np:
                            np_text = " ".join(current_np)
                            if len(np_text) > 2:
                                noun_phrases.append(np_text)
                            current_np = []
                if current_np:
                    np_text = " ".join(current_np)
                    if len(np_text) > 2:
                        noun_phrases.append(np_text)

                if not noun_phrases:
                    continue

                # Analyse the sentence once
                sent_result = self.analyze(sentence, models=["vader", "textblob"])

                for np_text in noun_phrases:
                    np_lower = np_text.lower()
                    if np_lower in seen_aspects:
                        continue
                    seen_aspects.add(np_lower)
                    aspects.append(
                        {
                            "aspect": np_text,
                            "sentence": sentence,
                            "sentiment": sent_result["overall_sentiment"],
                            "confidence": sent_result["overall_confidence"],
                        }
                    )

            return aspects

        except Exception as exc:
            logger.error("Aspect extraction failed: %s", exc)
            return self._aspect_fallback(text)

    def _aspect_fallback(self, text: str) -> List[Dict[str, Any]]:
        """Simple regex-based fallback for aspect extraction."""
        sentences = re.split(r"[.!?]+", text)
        results: List[Dict[str, Any]] = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 5:
                continue
            analysis = self.analyze(sent, models=["vader", "textblob"])
            results.append(
                {
                    "aspect": sent[:50],
                    "sentence": sent,
                    "sentiment": analysis["overall_sentiment"],
                    "confidence": analysis["overall_confidence"],
                }
            )
        return results
