# Production-Ready NLP Sarcasm Detection Microservice

[![CI/CD Pipeline](https://github.com/subki72/NLP-Sarcasm-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/subki72/NLP-Sarcasm-Detection/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

An enterprise-grade Natural Language Processing (NLP) microservice for detecting sarcasm in text. Built using a fine-tuned **DistilBERT** transformer model (`distilbert-base-uncased`) and served via an asynchronous **FastAPI** application layer.

The architecture emphasizes production engineering standards: high throughput, defensive input validation, automated Prometheus metrics, rate limiting, Docker containerization, and an automated test suite.

---

## Key Production Features

- **High-Performance Inference**: DistilBERT cached as a memory-efficient singleton loaded through FastAPI's async lifespan manager.
- **Strict Input Validation and Sanitization**: Pydantic v2 schemas enforce 1 to 500 characters, strip whitespace automatically, and validate data types.
- **Rate Limiting**: Built-in SlowAPI limiter (30 requests/minute per client IP on inference, 120 requests/minute global) to prevent traffic bursts.
- **Observability and Health Checks**:
  - `GET /health`: Liveness and readiness probe for container orchestration.
  - `GET /metrics`: Prometheus metrics endpoint tracking total request count and inference latency histograms.
- **Resilience and Graceful Fallback**: Returns standard HTTP 503 if weights are not loaded, with internal error logging without stack trace leakage.
- **API Versioning**: Canonical `/api/v1/predict` endpoint with backward-compatible `/predict` alias.
- **CORS Configurable**: Environment-driven CORS middleware supporting cross-origin clients.
- **Full Test Suite**: 20 automated unit and integration tests with 100% pass rate.
- **Containerization**: Multi-layer, non-root user (`appuser`) Docker container with automated health check.

---

## Project Structure

```text
NLP-Sarcasm-Detection/
|
+-- .github/workflows/
|   +-- ci.yml             # GitHub Actions automated lint, test, and Docker build
+-- notebooks/             # Exploratory Data Analysis and training experiments
|   +-- Sarcasm_Detection.ipynb
+-- src/                   # Core production microservice package
|   +-- __init__.py        # Package identifier
|   +-- config.py          # Environment-aware path and parameter configuration
|   +-- schemas.py         # Pydantic v2 DTOs (TextInput, PredictionResponse, etc.)
|   +-- inference.py       # OOP DistilBERT prediction engine
|   +-- main.py            # FastAPI ASGI application, routes, and middleware
+-- tests/                 # Automated test suite
|   +-- __init__.py
|   +-- test_schemas.py    # Schema edge cases and sanitization tests
|   +-- test_config.py     # Environment variable override tests
|   +-- test_api.py        # Integration tests using TestClient and mocks
+-- .dockerignore          # Docker build exclusions
+-- .env.example           # Environment variable template
+-- .gitignore             # Git exclusion rules
+-- docker-compose.yml     # Local multi-service orchestration
+-- Dockerfile             # Production-grade multi-stage containerfile
+-- LICENSE                # MIT License
+-- pyproject.toml         # Packaging metadata, Ruff linter, and Pytest config
+-- README.md              # Project documentation
+-- requirements.txt       # Production dependencies
```

---

## Model Evaluation

Evaluated on a held-out test set of news headlines:

| Metric | Score |
|---|---|
| Accuracy | 92.24% |
| F1-Score | 0.9192 |

*Domain Note*: The model is trained primarily on journalistic headlines (satirical vs. factual). Sarcasm in informal slang or conversational dialog may require additional domain fine-tuning.

---

## Quickstart and Installation

### Option 1: Running with Docker (Recommended)

Run the entire service in a production container with one command:

```bash
docker compose up --build -d
```

The API will start at `http://127.0.0.1:8000`.

Check container health:
```bash
docker compose ps
```

---

### Option 2: Running Locally

#### 1. Clone and Setup Environment

```bash
git clone https://github.com/subki72/NLP-Sarcasm-Detection.git
cd NLP-Sarcasm-Detection

# Create virtual environment
conda create -n sarcasm_detect python=3.10 -y
conda activate sarcasm_detect

# Install dependencies
pip install -r requirements.txt
```

#### 2. Configure Environment

```bash
cp .env.example .env
```

#### 3. Run Automated Tests

```bash
pytest tests -v
```

#### 4. Launch the API Server

```bash
uvicorn src.main:app --reload --port 8000
```

---

## API Endpoints and Usage

Interactive Swagger UI documentation is available at: `http://127.0.0.1:8000/docs`

### 1. Detect Sarcasm (`POST /api/v1/predict`)

**cURL Example:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/v1/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Man Finally Finishes Reading Terms and Conditions Agreement"
  }'
```

**Expected JSON Response (HTTP 200):**
```json
{
  "status": "success",
  "data": {
    "text": "Man Finally Finishes Reading Terms and Conditions Agreement",
    "prediction": "SARCASTIC",
    "confidence": 99.48,
    "is_sarcastic": true
  }
}
```

### 2. Health and Readiness Probe (`GET /health`)

```bash
curl 'http://127.0.0.1:8000/health'
```

**Response (HTTP 200):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### 3. Prometheus Metrics (`GET /metrics`)

```bash
curl 'http://127.0.0.1:8000/metrics'
```

Exposes standard metrics including:
- `sarcasm_prediction_requests_total`
- `sarcasm_prediction_latency_seconds`

---

## Code Quality and Linting

Verify formatting and lint standards with Ruff:

```bash
# Check code style
ruff check src tests

# Format code
ruff format src tests
```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
