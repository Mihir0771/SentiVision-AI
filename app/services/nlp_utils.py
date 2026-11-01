"""
NLP utility functions for the Sentiment Analyzer platform.

Provides keyword extraction, named-entity recognition (regex-based), language
detection, text summarisation (extractive), word-frequency analysis, and
general text cleaning / statistics.

All public functions in this module are standalone (no class required) and
handle missing optional dependencies gracefully.
"""

from __future__ import annotations

import logging
import math
import re
import string
from collections import Counter
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency probes
# ---------------------------------------------------------------------------

try:
    from langdetect import detect as _langdetect_detect  # type: ignore
    from langdetect import detect_langs as _langdetect_langs  # type: ignore

    _LANGDETECT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LANGDETECT_AVAILABLE = False
    logger.info("langdetect not installed – language detection unavailable.")

# ---------------------------------------------------------------------------
# NLTK bootstrap
# ---------------------------------------------------------------------------

_NLTK_READY = False
_STOPWORDS: set = set()


def _ensure_nltk() -> bool:
    """Download required NLTK resources and populate the stopwords set."""
    global _NLTK_READY, _STOPWORDS
    if _NLTK_READY:
        return True
    try:
        import nltk  # type: ignore

        for pkg, subdir in (
            ("punkt", "tokenizers"),
            ("punkt_tab", "tokenizers"),
            ("stopwords", "corpora"),
            ("averaged_perceptron_tagger", "taggers"),
            ("averaged_perceptron_tagger_eng", "taggers"),
        ):
            try:
                nltk.data.find(f"{subdir}/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)

        from nltk.corpus import stopwords as _sw  # type: ignore

        _STOPWORDS = set(_sw.words("english"))
        _NLTK_READY = True
        return True
    except Exception as exc:  # pragma: no cover
        logger.warning("NLTK bootstrap failed: %s", exc)
        # Provide a minimal fallback stopword set
        _STOPWORDS = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "is", "it", "this", "that", "was", "are", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "shall",
            "can", "with", "from", "by", "as", "not", "no", "so", "if",
            "its", "my", "your", "our", "their", "his", "her", "i", "me",
            "we", "you", "he", "she", "they", "them", "what", "which",
            "who", "whom", "how", "when", "where", "why", "all", "each",
            "every", "both", "few", "more", "most", "other", "some", "such",
            "than", "too", "very", "just", "about", "above", "after",
        }
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_keywords(text: str, top_n: int = 10) -> List[Dict[str, Any]]:
    """Extract keywords using a simple TF-IDF–inspired scoring approach.

    The text is split into sentences; each sentence acts as a "document".
    Term frequency is computed per-sentence and inverse document frequency
    across all sentences.  The top-scoring terms are returned.

    Parameters
    ----------
    text : str
        Input text.
    top_n : int
        Maximum number of keywords to return.

    Returns
    -------
    list of dict
        ``[{"keyword": str, "score": float}, ...]`` sorted by descending score.
    """
    _ensure_nltk()

    try:
        sentences = _split_sentences(text)
        if not sentences:
            return []

        # Tokenise each sentence
        docs: List[List[str]] = []
        for sent in sentences:
            tokens = _tokenise(sent)
            docs.append(tokens)

        # IDF
        total_docs = len(docs)
        doc_freq: Counter = Counter()
        for tokens in docs:
            doc_freq.update(set(tokens))

        idf: Dict[str, float] = {}
        for term, df in doc_freq.items():
            idf[term] = math.log((1 + total_docs) / (1 + df)) + 1

        # Aggregate TF across all sentences then multiply by IDF
        tf_global: Counter = Counter()
        for tokens in docs:
            tf_global.update(tokens)

        scored: Dict[str, float] = {}
        for term, freq in tf_global.items():
            scored[term] = round(freq * idf.get(term, 1.0), 4)

        top = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"keyword": kw, "score": sc} for kw, sc in top]

    except Exception as exc:
        logger.error("Keyword extraction failed: %s", exc)
        return []


def detect_language(text: str) -> Dict[str, Any]:
    """Detect the language of *text*.

    Parameters
    ----------
    text : str

    Returns
    -------
    dict
        ``{"language": str, "confidence": float}``
    """
    if not _LANGDETECT_AVAILABLE:
        return {"language": "unknown", "confidence": 0.0, "error": "langdetect not installed."}
    try:
        lang = _langdetect_detect(text)
        # Get probability list
        probs = _langdetect_langs(text)
        confidence = 0.0
        for item in probs:
            if item.lang == lang:
                confidence = round(float(item.prob), 4)
                break
        return {"language": lang, "confidence": confidence}
    except Exception as exc:
        logger.error("Language detection failed: %s", exc)
        return {"language": "unknown", "confidence": 0.0, "error": str(exc)}


def extract_entities(text: str) -> List[Dict[str, str]]:
    """Extract common named entities using regex patterns.

    Recognised types: ``email``, ``url``, ``hashtag``, ``mention``,
    ``phone``, ``capitalized_phrase`` (likely names / organisations).

    Parameters
    ----------
    text : str

    Returns
    -------
    list of dict
        ``[{"text": str, "type": str}, ...]``
    """
    entities: List[Dict[str, str]] = []
    seen: set = set()

    patterns = [
        ("email", r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
        ("url", r"https?://[^\s<>\"']+"),
        ("hashtag", r"#\w{2,}"),
        ("mention", r"@\w{2,}"),
        ("phone", r"\+?\d[\d\-\s()]{7,}\d"),
        (
            "capitalized_phrase",
            r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b",
        ),
    ]

    for etype, pattern in patterns:
        for match in re.finditer(pattern, text):
            value = match.group().strip()
            if value not in seen:
                seen.add(value)
                entities.append({"text": value, "type": etype})

    return entities


def get_word_frequency(text: str, top_n: int = 50) -> List[Dict[str, Any]]:
    """Compute word frequencies after removing stopwords.

    Parameters
    ----------
    text : str
    top_n : int

    Returns
    -------
    list of dict
        ``[{"word": str, "count": int}, ...]`` sorted by descending count.
    """
    _ensure_nltk()

    tokens = _tokenise(text)
    counts = Counter(tokens)
    top = counts.most_common(top_n)
    return [{"word": w, "count": c} for w, c in top]


def clean_text(text: str) -> str:
    """Remove URLs, mentions, hashtags, and special characters from *text*.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Cleaned text.
    """
    text = re.sub(r"https?://\S+", "", text)          # URLs
    text = re.sub(r"@\w+", "", text)                   # Mentions
    text = re.sub(r"#\w+", "", text)                   # Hashtags
    text = re.sub(r"[^\w\s.,!?;:'\"-]", "", text)      # Special chars
    text = re.sub(r"\s+", " ", text)                    # Collapse whitespace
    return text.strip()


def get_text_statistics(text: str) -> Dict[str, Any]:
    """Compute basic statistics for *text*.

    Parameters
    ----------
    text : str

    Returns
    -------
    dict
        ``{"word_count", "sentence_count", "char_count", "avg_word_length",
        "reading_time_minutes"}``
    """
    words = text.split()
    word_count = len(words)
    sentences = _split_sentences(text)
    sentence_count = len(sentences)
    char_count = len(text)
    avg_word_length = (
        round(sum(len(w) for w in words) / word_count, 2) if word_count else 0.0
    )
    # Average adult reading speed ≈ 200 words per minute
    reading_time = round(word_count / 200, 2)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "char_count": char_count,
        "avg_word_length": avg_word_length,
        "reading_time_minutes": reading_time,
    }


def generate_word_cloud_data(text: str, top_n: int = 80) -> List[Dict[str, Any]]:
    """Return word-frequency data suitable for rendering a word cloud.

    Sizes are scaled linearly from 10 to 100 based on frequency rank.

    Parameters
    ----------
    text : str
    top_n : int

    Returns
    -------
    list of dict
        ``[{"text": str, "size": int}, ...]``
    """
    freq = get_word_frequency(text, top_n=top_n)
    if not freq:
        return []

    max_count = freq[0]["count"]
    min_count = freq[-1]["count"] if len(freq) > 1 else max_count
    count_range = max(max_count - min_count, 1)

    cloud: List[Dict[str, Any]] = []
    for item in freq:
        size = int(10 + 90 * (item["count"] - min_count) / count_range)
        cloud.append({"text": item["word"], "size": size})

    return cloud


def summarize_text(text: str, sentences: int = 3) -> str:
    """Produce an extractive summary by scoring and selecting top sentences.

    Scoring heuristic:
    - Each sentence is scored by the sum of its non-stopword token frequencies
      divided by the sentence length (in tokens).

    Parameters
    ----------
    text : str
    sentences : int
        Number of sentences in the summary.

    Returns
    -------
    str
        The summary text.
    """
    _ensure_nltk()

    sents = _split_sentences(text)
    if len(sents) <= sentences:
        return text.strip()

    # Global word frequencies (no stopwords)
    all_tokens = _tokenise(text)
    freq = Counter(all_tokens)

    scored: List[tuple] = []
    for idx, sent in enumerate(sents):
        tokens = _tokenise(sent)
        if not tokens:
            scored.append((idx, sent, 0.0))
            continue
        score = sum(freq.get(t, 0) for t in tokens) / len(tokens)
        scored.append((idx, sent, score))

    # Pick top-scoring sentences, preserving original order
    top = sorted(scored, key=lambda x: x[2], reverse=True)[:sentences]
    top_ordered = sorted(top, key=lambda x: x[0])

    return " ".join(s[1] for s in top_ordered)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> List[str]:
    """Split *text* into sentences using NLTK when available."""
    try:
        import nltk  # type: ignore

        _ensure_nltk()
        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except Exception:
        # Fallback: simple regex split
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]


def _tokenise(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stopwords, return token list."""
    _ensure_nltk()
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
