"""
FastAPI application entry point for the Sarcasm Detection API.
Production-grade service with lifespan lifecycle, CORS, rate limiting,
Prometheus metrics, health probes, and API versioning.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.inference import SarcasmPredictor
from src.schemas import HealthResponse, PredictionResponse, TextInput

# Setup basic logging
logger = logging.getLogger("sarcasm_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Setup Rate Limiter (key by client IP, default 120 req/min globally)
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])

# Prometheus Metrics
PREDICTION_REQUESTS_TOTAL = Counter(
    "sarcasm_prediction_requests_total",
    "Total sarcasm prediction requests",
    ["status", "prediction"],
)
PREDICTION_LATENCY_SECONDS = Histogram(
    "sarcasm_prediction_latency_seconds",
    "Time spent processing sarcasm prediction in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for startup and shutdown events.
    Loads the ML model once on startup and stores it in app.state.
    """
    logger.info("Initializing Sarcasm Detection Service...")
    try:
        app.state.predictor = SarcasmPredictor()
        logger.info("SarcasmPredictor successfully loaded and ready for inference.")
    except Exception as exc:
        logger.error(f"Failed to initialize SarcasmPredictor on startup: {exc}")
        app.state.predictor = None

    yield

    # Cleanup resources on shutdown
    logger.info("Shutting down Sarcasm Detection Service...")
    if hasattr(app.state, "predictor"):
        del app.state.predictor


# OpenAPI tags metadata for Swagger UI
tags_metadata = [
    {"name": "System", "description": "Liveness, readiness, and metrics endpoints."},
    {"name": "Prediction", "description": "NLP Sarcasm detection classification endpoints."},
]

app = FastAPI(
    title="Production-Ready NLP Sarcasm Detection API",
    description=(
        "Production-grade microservice for detecting sarcasm in news headlines "
        "using fine-tuned DistilBERT and FastAPI."
    ),
    version="1.0.0",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

# Attach rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Attach CORS middleware
cors_origins_raw = os.environ.get("CORS_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_predictor(request: Request) -> SarcasmPredictor:
    """
    Dependency to retrieve the predictor instance from app.state.
    Raises HTTP 503 if the model is not loaded.
    """
    predictor: SarcasmPredictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or temporarily unavailable. Please check server logs.",
        )
    return predictor


@app.get("/", tags=["System"])
def home():
    """Root endpoint providing service status and documentation links."""
    return {
        "message": "Welcome to Sarcasm Detection API!",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "metrics_url": "/metrics",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(request: Request):
    """
    Health check endpoint for Docker/Kubernetes liveness and readiness probes.
    """
    is_loaded = getattr(request.app.state, "predictor", None) is not None
    predictor_instance = getattr(request.app.state, "predictor", None)

    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "device": getattr(predictor_instance, "device", None) if is_loaded else None,
    }


@app.get("/metrics", tags=["System"])
def metrics():
    """
    Prometheus metrics exposition endpoint.
    Scraped by Prometheus server to monitor throughput, latency, and prediction distributions.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Detect Sarcasm (v1)",
)
@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Detect Sarcasm (Legacy Alias)",
    include_in_schema=True,
)
@limiter.limit("30/minute")
def predict_sarcasm(
    request: Request,
    input_data: TextInput,
    predictor: SarcasmPredictor = Depends(get_predictor),
):
    """
    Classifies whether the input headline is SARCASTIC or GENUINE.
    Guarded by rate limiting (30 requests/minute per client IP) and length validation (1-500 chars).
    """
    start_time = time.perf_counter()
    try:
        result = predictor.predict(input_data.text)
        duration = time.perf_counter() - start_time

        # Track observability metrics
        PREDICTION_LATENCY_SECONDS.observe(duration)
        PREDICTION_REQUESTS_TOTAL.labels(
            status="success", prediction=result.get("prediction", "UNKNOWN")
        ).inc()

        return {"status": "success", "data": result}
    except Exception as e:
        PREDICTION_REQUESTS_TOTAL.labels(status="error", prediction="NONE").inc()
        logger.exception(f"Prediction failed unexpectedly: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during prediction inference.",
        )
