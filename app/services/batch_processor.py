"""
Batch processing engine for CSV, JSON, and Excel file uploads.

Processes large files row-by-row with progress tracking, produces summary
statistics, and supports exporting results in CSV, JSON, and Excel formats.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    import pandas as pd  # type: ignore

    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PANDAS_AVAILABLE = False
    logger.warning("pandas not installed – batch processing will be unavailable.")

# Local import of the text analyser
from app.services.text_analyzer import TextSentimentAnalyzer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS = {".csv", ".json", ".xlsx", ".xls"}

# Heuristic column-name patterns for auto-detecting the text column
_TEXT_COLUMN_HINTS = (
    "text",
    "content",
    "review",
    "comment",
    "message",
    "body",
    "description",
    "tweet",
    "post",
    "feedback",
    "sentence",
    "title",
    "summary",
    "note",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_output_dir() -> str:
    """Create and return the path to the batch-output directory."""
    out_dir = os.path.join(os.getcwd(), "temp", "batch_output")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class BatchProcessor:
    """Batch sentiment analysis processor for tabular files.

    Parameters
    ----------
    analyzer : TextSentimentAnalyzer, optional
        A pre-configured text analyser instance.  If ``None``, a default
        instance is created internally.
    """

    def __init__(self, analyzer: Optional[TextSentimentAnalyzer] = None) -> None:
        self._analyzer = analyzer or TextSentimentAnalyzer()
        logger.info("BatchProcessor initialised.")

    # ------------------------------------------------------------------
    # File readers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_csv(filepath: str) -> "pd.DataFrame":
        """Read a CSV file into a DataFrame.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
        """
        if not _PANDAS_AVAILABLE:
            raise ImportError("pandas is required for CSV processing.")
        return pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")

    @staticmethod
    def _read_json(filepath: str) -> "pd.DataFrame":
        """Read a JSON file (array of objects or JSON-lines) into a DataFrame.

        Parameters
        ----------
        filepath : str
            Path to the JSON file.

        Returns
        -------
        pd.DataFrame
        """
        if not _PANDAS_AVAILABLE:
            raise ImportError("pandas is required for JSON processing.")
        try:
            return pd.read_json(filepath, orient="records")
        except ValueError:
            return pd.read_json(filepath, lines=True)

    @staticmethod
    def _read_excel(filepath: str) -> "pd.DataFrame":
        """Read an Excel file into a DataFrame.

        Parameters
        ----------
        filepath : str
            Path to the ``.xlsx`` / ``.xls`` file.

        Returns
        -------
        pd.DataFrame
        """
        if not _PANDAS_AVAILABLE:
            raise ImportError("pandas is required for Excel processing.")
        return pd.read_excel(filepath)

    # ------------------------------------------------------------------
    # Column detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_text_column(df: "pd.DataFrame") -> str:
        """Heuristically choose the column most likely to contain free text.

        Strategy
        --------
        1. Exact (case-insensitive) match against known hint names.
        2. Substring match (e.g. ``review_text``).
        3. First ``object``-dtype column with the highest average string
           length.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        str
            The selected column name.

        Raises
        ------
        ValueError
            If no suitable text column can be identified.
        """
        columns_lower = {c.lower(): c for c in df.columns}

        # Exact match
        for hint in _TEXT_COLUMN_HINTS:
            if hint in columns_lower:
                return columns_lower[hint]

        # Substring match
        for hint in _TEXT_COLUMN_HINTS:
            for col_lower, col_orig in columns_lower.items():
                if hint in col_lower:
                    return col_orig

        # Longest average string length among object columns
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if obj_cols:
            avg_lengths = {
                c: df[c].dropna().astype(str).str.len().mean() for c in obj_cols
            }
            return max(avg_lengths, key=avg_lengths.get)  # type: ignore[arg-type]

        raise ValueError(
            "Could not auto-detect a text column. "
            "Please specify the column name explicitly."
        )

    # ------------------------------------------------------------------
    # File Preview
    # ------------------------------------------------------------------

    def get_file_preview(self, filepath: str, max_rows: int = 5) -> Dict[str, Any]:
        """Read column names and sample rows from an uploaded tabular file."""
        if not _PANDAS_AVAILABLE:
            return {"columns": [], "sample_rows": [], "detected_column": None, "total_rows": 0}

        ext = os.path.splitext(filepath)[1].lower()
        try:
            if ext == ".csv":
                df = self._read_csv(filepath)
            elif ext == ".json":
                df = self._read_json(filepath)
            elif ext in (".xlsx", ".xls"):
                df = self._read_excel(filepath)
            else:
                return {"columns": [], "sample_rows": [], "detected_column": None, "total_rows": 0}

            columns = list(df.columns)
            detected = None
            try:
                detected = self._detect_text_column(df)
            except Exception:
                detected = columns[0] if columns else None

            sample_rows = df.head(max_rows).fillna("").to_dict(orient="records")
            return {
                "columns": columns,
                "detected_column": detected,
                "sample_rows": sample_rows,
                "total_rows": len(df),
            }
        except Exception as exc:
            logger.error("Error reading file preview for %s: %s", filepath, exc)
            return {"columns": [], "sample_rows": [], "detected_column": None, "total_rows": 0}

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process_file(
        self,
        filepath: str,
        text_column: Optional[str] = None,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Read a file, detect the text column, and process every row.

        Parameters
        ----------
        filepath : str
            Path to a CSV, JSON, or Excel file.
        text_column : str, optional
            Name of the column containing text.  Auto-detected if omitted.
        callback : callable, optional
            Progress callback ``(processed, total)`` invoked after each row.

        Returns
        -------
        dict
            Processing results with per-row sentiments and summary statistics.
        """
        if not _PANDAS_AVAILABLE:
            return {
                "error": "pandas is not installed.",
                "total_items": 0,
                "processed": 0,
                "failed": 0,
                "results": [],
                "summary": {},
                "processing_time": 0.0,
            }

        ext = os.path.splitext(filepath)[1].lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            return {
                "error": f"Unsupported file extension: {ext}",
                "total_items": 0,
                "processed": 0,
                "failed": 0,
                "results": [],
                "summary": {},
                "processing_time": 0.0,
            }

        try:
            if ext == ".csv":
                df = self._read_csv(filepath)
            elif ext == ".json":
                df = self._read_json(filepath)
            else:
                df = self._read_excel(filepath)
        except Exception as exc:
            logger.error("Failed to read '%s': %s", filepath, exc)
            return {
                "error": str(exc),
                "total_items": 0,
                "processed": 0,
                "failed": 0,
                "results": [],
                "summary": {},
                "processing_time": 0.0,
            }

        if text_column is None:
            try:
                text_column = self._detect_text_column(df)
                logger.info("Auto-detected text column: '%s'", text_column)
            except ValueError as exc:
                return {
                    "error": str(exc),
                    "total_items": len(df),
                    "processed": 0,
                    "failed": 0,
                    "results": [],
                    "summary": {},
                    "processing_time": 0.0,
                }

        return self.process_dataframe(df, text_column, callback=callback)

    def process_dataframe(
        self,
        df: "pd.DataFrame",
        text_column: str,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Process every row of a DataFrame through the sentiment analyser.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        text_column : str
            Name of the column to analyse.
        callback : callable, optional
            ``(processed, total)`` progress callback.

        Returns
        -------
        dict
            Full batch result including ``results``, ``summary``, and timing.
        """
        if text_column not in df.columns:
            return {
                "error": f"Column '{text_column}' not found in data.",
                "total_items": len(df),
                "processed": 0,
                "failed": 0,
                "results": [],
                "summary": {},
                "processing_time": 0.0,
            }

        total = len(df)
        results: List[Dict[str, Any]] = []
        failed = 0
        start_time = time.time()

        for idx, row in df.iterrows():
            text = str(row.get(text_column, ""))
            try:
                analysis = self._analyzer.analyze(text, models=["vader", "textblob"])
                results.append(
                    {
                        "index": int(idx),
                        "text": text[:500],  # truncate for storage
                        "sentiment": analysis["overall_sentiment"],
                        "confidence": analysis["overall_confidence"],
                        "scores": analysis.get("ensemble_scores", {}),
                    }
                )
            except Exception as exc:
                logger.error("Row %s failed: %s", idx, exc)
                failed += 1
                results.append(
                    {
                        "index": int(idx),
                        "text": text[:500],
                        "sentiment": "error",
                        "confidence": 0.0,
                        "error": str(exc),
                    }
                )

            if callback:
                try:
                    callback(len(results), total)
                except Exception:
                    pass

        processing_time = round(time.time() - start_time, 3)
        summary = self.generate_summary_stats(results)

        return {
            "total_items": total,
            "processed": total - failed,
            "failed": failed,
            "results": results,
            "summary": summary,
            "processing_time": processing_time,
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @staticmethod
    def generate_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate statistics from a list of row results.

        Parameters
        ----------
        results : list of dict
            Each entry must contain ``"sentiment"`` and ``"confidence"`` keys.

        Returns
        -------
        dict
            Counts, percentages, and average confidence.
        """
        total = len(results) or 1  # avoid division by zero
        pos = sum(1 for r in results if r.get("sentiment") == "positive")
        neg = sum(1 for r in results if r.get("sentiment") == "negative")
        neu = sum(1 for r in results if r.get("sentiment") == "neutral")
        confidences = [r.get("confidence", 0.0) for r in results if r.get("confidence")]

        return {
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neu,
            "positive_pct": round(pos / total * 100, 2),
            "negative_pct": round(neg / total * 100, 2),
            "neutral_pct": round(neu / total * 100, 2),
            "avg_confidence": round(sum(confidences) / max(len(confidences), 1), 4),
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_results(
        self,
        results: Dict[str, Any],
        format: str = "csv",
    ) -> str:
        """Export batch results to a file.

        Parameters
        ----------
        results : dict
            Full batch result (as returned by :meth:`process_file`).
        format : str
            ``"csv"``, ``"json"``, or ``"excel"``.

        Returns
        -------
        str
            Absolute path of the exported file.
        """
        out_dir = _ensure_output_dir()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rows = results.get("results", [])

        if format == "json":
            filepath = os.path.join(out_dir, f"batch_results_{ts}.json")
            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2, default=str)
            logger.info("Exported results to %s", filepath)
            return filepath

        if not _PANDAS_AVAILABLE:
            # Fallback: write JSON if pandas unavailable
            filepath = os.path.join(out_dir, f"batch_results_{ts}.json")
            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2, default=str)
            logger.warning("pandas unavailable; exported as JSON instead of %s.", format)
            return filepath

        df = pd.DataFrame(rows)

        if format == "excel":
            filepath = os.path.join(out_dir, f"batch_results_{ts}.xlsx")
            df.to_excel(filepath, index=False)
        else:  # default to CSV
            filepath = os.path.join(out_dir, f"batch_results_{ts}.csv")
            df.to_csv(filepath, index=False)

        logger.info("Exported results to %s", filepath)
        return filepath
