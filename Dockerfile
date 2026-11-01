# =============================================================================
# SentiVision AI — Dockerfile
# =============================================================================
# Multi-stage production build
# Usage:
#   docker build -t sentivision-ai .
#   docker run -p 5000:5000 sentivision-ai
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — install Python dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Runtime — lean production image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="SentiVision AI Team <team@sentivision.ai>"
LABEL description="AI-Based Sentiment Analyzer with facial emotion detection"
LABEL version="1.0.0"

# Install runtime system dependencies (OpenCV needs libgl1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN groupadd -r sentivision && useradd -r -g sentivision -m sentivision

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create required runtime directories
RUN mkdir -p uploads reports temp \
    && chown -R sentivision:sentivision /app

# Download NLTK data
RUN python -c "import nltk; nltk.download('vader_lexicon', download_dir='/usr/local/nltk_data'); nltk.download('punkt', download_dir='/usr/local/nltk_data'); nltk.download('averaged_perceptron_tagger', download_dir='/usr/local/nltk_data')" 2>/dev/null || true

# Switch to non-root user
USER sentivision

# Expose the application port
EXPOSE 5000

# Environment defaults
ENV FLASK_ENV=production \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start with gunicorn + eventlet worker
CMD ["gunicorn", \
     "--worker-class", "eventlet", \
     "--workers", "1", \
     "--bind", "0.0.0.0:5000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "run:app"]
