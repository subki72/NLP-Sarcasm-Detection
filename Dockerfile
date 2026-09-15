# ==============================================================================
# Production Dockerfile for NLP Sarcasm Detection Microservice
# ==============================================================================
FROM python:3.10-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

# Install system dependencies if required (keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies (CPU-optimized PyTorch if specified)
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-privileged user for security (fail-safe against container escapes)
RUN useradd -u 1001 -m -s /bin/bash appuser && \
    mkdir -p /app/models && \
    chown -R appuser:appuser /app

# Copy application source code
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser README.md /app/README.md

# Switch to non-root user
USER appuser

# Expose microservice HTTP port
EXPOSE 8000

# Container Healthcheck (uses native python urllib - no curl package overhead)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; res = urllib.request.urlopen('http://127.0.0.1:8000/health'); exit(0 if res.getcode() == 200 else 1)"

# Start production server with Uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
