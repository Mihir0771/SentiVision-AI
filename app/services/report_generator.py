"""
Report generation engine for creating PDF reports and chart images.

Uses matplotlib (``Agg`` backend) to produce sentiment distribution charts,
emotion radar plots, word-cloud images, and timeline visualisations.  Results
can be exported as CSV / JSON or bundled into a PDF report.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Matplotlib setup (non-interactive backend)
# ---------------------------------------------------------------------------

try:
    import matplotlib

    matplotlib.use("Agg")  # must precede pyplot import
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import numpy as np  # type: ignore

    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False
    logger.warning("matplotlib/numpy not installed – chart generation unavailable.")

# ---------------------------------------------------------------------------
# Optional PDF backend
# ---------------------------------------------------------------------------

try:
    from matplotlib.backends.backend_pdf import PdfPages as _PdfPages  # type: ignore

    _PDF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PDF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants / palette
# ---------------------------------------------------------------------------

_COLORS = {
    "positive": "#4CAF50",
    "negative": "#F44336",
    "neutral": "#FFC107",
    "primary": "#2196F3",
    "secondary": "#9C27B0",
    "background": "#FAFAFA",
}

_EMOTION_COLORS = {
    "angry": "#F44336",
    "disgust": "#FF9800",
    "fear": "#FFEB3B",
    "happy": "#4CAF50",
    "sad": "#2196F3",
    "surprise": "#9C27B0",
    "neutral": "#9E9E9E",
}


def _output_dir() -> str:
    """Return (and create) the reports output directory."""
    d = os.path.join(os.getcwd(), "temp", "reports")
    os.makedirs(d, exist_ok=True)
    return d


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ReportGenerator:
    """Chart and report generator for sentiment / emotion analysis results.

    All ``generate_*`` methods save an image to the reports directory and
    return the absolute file path.
    """

    def __init__(self) -> None:
        logger.info("ReportGenerator initialised.")

    # ------------------------------------------------------------------
    # Sentiment charts
    # ------------------------------------------------------------------

    def generate_sentiment_chart(
        self,
        results: Dict[str, Any],
        chart_type: str = "pie",
    ) -> str:
        """Generate a sentiment distribution chart image.

        Parameters
        ----------
        results : dict
            Must contain ``summary`` with ``positive_count``,
            ``negative_count``, ``neutral_count`` **or** top-level
            ``ensemble_scores``.
        chart_type : str
            ``"pie"``, ``"bar"``, ``"radar"``, or ``"line"``.

        Returns
        -------
        str
            Absolute path to the saved PNG file.
        """
        if not _MPL_AVAILABLE:
            return self._no_mpl()

        # Extract data from summary or ensemble_scores
        summary = results.get("summary", {})
        if summary:
            labels = ["Positive", "Negative", "Neutral"]
            values = [
                summary.get("positive_count", summary.get("positive_pct", 0)),
                summary.get("negative_count", summary.get("negative_pct", 0)),
                summary.get("neutral_count", summary.get("neutral_pct", 0)),
            ]
        else:
            scores = results.get("ensemble_scores", {})
            labels = [k.capitalize() for k in scores]
            values = list(scores.values())

        colors = [_COLORS["positive"], _COLORS["negative"], _COLORS["neutral"]]

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(_COLORS["background"])

        if chart_type == "bar":
            bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)
            ax.set_ylabel("Count / Score")
            ax.set_title("Sentiment Distribution", fontsize=14, fontweight="bold")
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.2f}" if isinstance(val, float) else str(val),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
        elif chart_type == "radar":
            self._draw_radar(ax, labels, values, colors)
        elif chart_type == "line":
            ax.plot(labels, values, marker="o", color=_COLORS["primary"], linewidth=2)
            ax.fill_between(labels, values, alpha=0.15, color=_COLORS["primary"])
            ax.set_title("Sentiment Scores", fontsize=14, fontweight="bold")
            ax.set_ylabel("Score")
        else:  # pie (default)
            non_zero = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
            if non_zero:
                l, v, c = zip(*non_zero)
                ax.pie(v, labels=l, colors=c, autopct="%1.1f%%", startangle=140,
                       textprops={"fontsize": 11})
            ax.set_title("Sentiment Distribution", fontsize=14, fontweight="bold")

        plt.tight_layout()
        path = os.path.join(_output_dir(), f"sentiment_{chart_type}_{_timestamp_slug()}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved sentiment chart → %s", path)
        return path

    # ------------------------------------------------------------------
    # Emotion chart
    # ------------------------------------------------------------------

    def generate_emotion_chart(self, emotions: Dict[str, float]) -> str:
        """Generate an emotion distribution bar chart.

        Parameters
        ----------
        emotions : dict
            ``{"happy": 0.72, "sad": 0.1, …}``

        Returns
        -------
        str
            Path to saved PNG.
        """
        if not _MPL_AVAILABLE:
            return self._no_mpl()

        labels = list(emotions.keys())
        values = list(emotions.values())
        colors = [_EMOTION_COLORS.get(e.lower(), "#607D8B") for e in labels]

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(_COLORS["background"])
        bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)
        ax.set_xlabel("Score")
        ax.set_title("Emotion Distribution", fontsize=14, fontweight="bold")
        ax.set_xlim(0, 1.05)

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2%}", va="center", fontsize=10)

        plt.tight_layout()
        path = os.path.join(_output_dir(), f"emotion_{_timestamp_slug()}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved emotion chart → %s", path)
        return path

    # ------------------------------------------------------------------
    # Batch report charts
    # ------------------------------------------------------------------

    def generate_batch_report_charts(self, batch_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate a set of charts for a batch analysis result.

        Parameters
        ----------
        batch_results : dict
            As returned by ``BatchProcessor.process_file()``.

        Returns
        -------
        dict
            ``{"pie": path, "bar": path, "confidence_hist": path}``
        """
        charts: Dict[str, str] = {}
        try:
            charts["pie"] = self.generate_sentiment_chart(batch_results, chart_type="pie")
            charts["bar"] = self.generate_sentiment_chart(batch_results, chart_type="bar")
            charts["confidence_hist"] = self._generate_confidence_histogram(batch_results)
        except Exception as exc:
            logger.error("Batch chart generation failed: %s", exc)
        return charts

    def _generate_confidence_histogram(self, batch_results: Dict[str, Any]) -> str:
        """Histogram of confidence scores across batch results."""
        if not _MPL_AVAILABLE:
            return self._no_mpl()

        confidences = [
            r.get("confidence", 0.0)
            for r in batch_results.get("results", [])
            if isinstance(r.get("confidence"), (int, float))
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(_COLORS["background"])
        ax.hist(confidences, bins=20, color=_COLORS["primary"], edgecolor="white", alpha=0.85)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Frequency")
        ax.set_title("Confidence Score Distribution", fontsize=14, fontweight="bold")
        plt.tight_layout()

        path = os.path.join(_output_dir(), f"confidence_hist_{_timestamp_slug()}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Comparison & timeline
    # ------------------------------------------------------------------

    def generate_comparison_chart(self, analyses: List[Dict[str, Any]]) -> str:
        """Compare sentiment scores across multiple analyses.

        Parameters
        ----------
        analyses : list of dict
            Each must have ``"text"`` (or ``"label"``) and ``"ensemble_scores"``.

        Returns
        -------
        str
            Path to saved PNG.
        """
        if not _MPL_AVAILABLE:
            return self._no_mpl()

        labels = []
        pos_vals, neg_vals, neu_vals = [], [], []
        for idx, a in enumerate(analyses):
            lbl = a.get("label", a.get("text", f"#{idx + 1}"))
            labels.append(lbl[:30])
            scores = a.get("ensemble_scores", {})
            pos_vals.append(scores.get("positive", 0))
            neg_vals.append(scores.get("negative", 0))
            neu_vals.append(scores.get("neutral", 0))

        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 6))
        fig.patch.set_facecolor(_COLORS["background"])
        ax.bar(x - width, pos_vals, width, label="Positive", color=_COLORS["positive"])
        ax.bar(x, neg_vals, width, label="Negative", color=_COLORS["negative"])
        ax.bar(x + width, neu_vals, width, label="Neutral", color=_COLORS["neutral"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Score")
        ax.set_title("Sentiment Comparison", fontsize=14, fontweight="bold")
        ax.legend()
        plt.tight_layout()

        path = os.path.join(_output_dir(), f"comparison_{_timestamp_slug()}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved comparison chart → %s", path)
        return path

    def generate_timeline_chart(self, analyses: List[Dict[str, Any]]) -> str:
        """Plot sentiment scores over time.

        Parameters
        ----------
        analyses : list of dict
            Each must have ``"analyzed_at"`` (ISO timestamp) and
            ``"overall_confidence"`` (or ``"ensemble_scores"``).

        Returns
        -------
        str
            Path to saved PNG.
        """
        if not _MPL_AVAILABLE:
            return self._no_mpl()

        timestamps = []
        pos_vals, neg_vals, neu_vals = [], [], []

        for a in analyses:
            ts = a.get("analyzed_at", "")
            try:
                dt = datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                dt = datetime.now(timezone.utc)
            timestamps.append(dt)
            scores = a.get("ensemble_scores", {})
            pos_vals.append(scores.get("positive", 0))
            neg_vals.append(scores.get("negative", 0))
            neu_vals.append(scores.get("neutral", 0))

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(_COLORS["background"])
        ax.plot(timestamps, pos_vals, marker="o", label="Positive", color=_COLORS["positive"])
        ax.plot(timestamps, neg_vals, marker="s", label="Negative", color=_COLORS["negative"])
        ax.plot(timestamps, neu_vals, marker="^", label="Neutral", color=_COLORS["neutral"])
        ax.set_xlabel("Time")
        ax.set_ylabel("Score")
        ax.set_title("Sentiment Over Time", fontsize=14, fontweight="bold")
        ax.legend()
        fig.autofmt_xdate()
        plt.tight_layout()

        path = os.path.join(_output_dir(), f"timeline_{_timestamp_slug()}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved timeline chart → %s", path)
        return path

    # ------------------------------------------------------------------
    # Word cloud
    # ------------------------------------------------------------------

    def generate_word_cloud_image(self, word_freq: List[Dict[str, Any]]) -> str:
        """Render a word cloud as a scatter-text plot.

        Parameters
        ----------
        word_freq : list of dict
            ``[{"text": str, "size": int}, ...]`` as returned by
            ``nlp_utils.generate_word_cloud_data``.

        Returns
        -------
        str
            Path to saved PNG.
        """
        if not _MPL_AVAILABLE:
            return self._no_mpl()

        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor("white")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Word Cloud", fontsize=16, fontweight="bold", pad=20)

        import random
        random.seed(42)
        cmap = plt.cm.get_cmap("viridis", len(word_freq) or 1)

        for idx, item in enumerate(word_freq):
            x = random.uniform(0.08, 0.92)
            y = random.uniform(0.08, 0.92)
            size = max(item.get("size", 10), 6)
            fontsize = int(size * 0.35) + 6
            color = cmap(idx / max(len(word_freq), 1))
            ax.text(x, y, item["text"], fontsize=fontsize, color=color,
                    ha="center", va="center", fontweight="bold", alpha=0.85)

        plt.tight_layout()
        path = os.path.join(_output_dir(), f"wordcloud_{_timestamp_slug()}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved word cloud → %s", path)
        return path

    # ------------------------------------------------------------------
    # PDF report
    # ------------------------------------------------------------------

    def create_pdf_report(
        self,
        analysis: Dict[str, Any],
        charts: List[str],
    ) -> str:
        """Bundle analysis text and chart images into a multi-page PDF.

        If ``matplotlib.backends.backend_pdf`` is available the charts are
        embedded via ``PdfPages``; otherwise the charts are saved individually
        and a JSON summary is written alongside them.

        Parameters
        ----------
        analysis : dict
            The analysis result dict to include as text.
        charts : list of str
            Paths to chart image files.

        Returns
        -------
        str
            Path to the generated PDF (or fallback JSON).
        """
        if _PDF_AVAILABLE and _MPL_AVAILABLE:
            return self._create_pdf_matplotlib(analysis, charts)

        # Fallback: save JSON summary next to charts
        path = os.path.join(_output_dir(), f"report_{_timestamp_slug()}.json")
        report = {
            "analysis": analysis,
            "charts": charts,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info("PDF backend unavailable; saved JSON report → %s", path)
        return path

    def _create_pdf_matplotlib(
        self,
        analysis: Dict[str, Any],
        charts: List[str],
    ) -> str:
        """Build a PDF using matplotlib's PdfPages backend."""
        path = os.path.join(_output_dir(), f"report_{_timestamp_slug()}.pdf")

        with _PdfPages(path) as pdf:
            # Title / summary page
            fig, ax = plt.subplots(figsize=(8.5, 11))
            fig.patch.set_facecolor("white")
            ax.axis("off")
            lines = [
                "Sentiment Analysis Report",
                "=" * 40,
                f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
                "",
            ]
            # Flatten key analysis fields
            for key in ("text", "overall_sentiment", "overall_confidence"):
                val = analysis.get(key)
                if val is not None:
                    display = str(val)[:120]
                    lines.append(f"{key}: {display}")
            summary = analysis.get("summary", {})
            if summary:
                lines.append("")
                lines.append("Summary Statistics")
                lines.append("-" * 30)
                for k, v in summary.items():
                    lines.append(f"  {k}: {v}")

            ax.text(
                0.05, 0.95, "\n".join(lines),
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
            )
            pdf.savefig(fig)
            plt.close(fig)

            # Chart pages
            for chart_path in charts:
                if not os.path.isfile(chart_path):
                    continue
                try:
                    img = plt.imread(chart_path)
                    fig, ax = plt.subplots(figsize=(8.5, 6.5))
                    ax.imshow(img)
                    ax.axis("off")
                    pdf.savefig(fig)
                    plt.close(fig)
                except Exception as exc:
                    logger.warning("Could not embed chart '%s': %s", chart_path, exc)

        logger.info("Saved PDF report → %s", path)
        return path

    # ------------------------------------------------------------------
    # CSV / JSON export
    # ------------------------------------------------------------------

    @staticmethod
    def export_to_csv(results: List[Dict[str, Any]], filepath: str) -> str:
        """Export a list of result dicts to a CSV file.

        Parameters
        ----------
        results : list of dict
        filepath : str

        Returns
        -------
        str
            Absolute path of the written file.
        """
        try:
            import pandas as pd  # type: ignore

            df = pd.DataFrame(results)
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            df.to_csv(filepath, index=False)
        except ImportError:
            import csv

            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            if not results:
                open(filepath, "w").close()
                return os.path.abspath(filepath)
            keys = list(results[0].keys())
            with open(filepath, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
        logger.info("Exported CSV → %s", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def export_to_json(results: List[Dict[str, Any]], filepath: str) -> str:
        """Export a list of result dicts to a JSON file.

        Parameters
        ----------
        results : list of dict
        filepath : str

        Returns
        -------
        str
            Absolute path of the written file.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        logger.info("Exported JSON → %s", filepath)
        return os.path.abspath(filepath)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _no_mpl() -> str:
        """Return an error path placeholder when matplotlib is missing."""
        logger.error("matplotlib is required for chart generation.")
        return ""

    @staticmethod
    def _draw_radar(ax: Any, labels: list, values: list, colors: list) -> None:
        """Draw a radar/spider chart on *ax*."""
        try:
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            values_closed = values + [values[0]]
            angles += [angles[0]]

            ax.clear()
            ax = plt.subplot(111, polar=True)
            ax.plot(angles, values_closed, "o-", linewidth=2, color=_COLORS["primary"])
            ax.fill(angles, values_closed, alpha=0.2, color=_COLORS["primary"])
            ax.set_thetagrids(
                [a * 180 / np.pi for a in angles[:-1]], labels, fontsize=11
            )
            ax.set_title("Sentiment Radar", fontsize=14, fontweight="bold", pad=20)
        except Exception as exc:
            logger.warning("Radar chart drawing failed: %s", exc)
