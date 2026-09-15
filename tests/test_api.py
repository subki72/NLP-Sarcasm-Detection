"""
Integration and API endpoint tests using FastAPI TestClient and mock predictor.
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app, get_predictor


class DummySarcasmPredictor:
    """Mock predictor to test API routes without requiring heavyweight model weights."""

    def __init__(self):
        self.device = "cpu"

    def predict(self, text: str) -> dict:
        return {"text": text, "prediction": "SARCASTIC", "confidence": 98.75, "is_sarcastic": True}


@pytest.fixture
def client(monkeypatch):
    """Provides a TestClient with a mock predictor injected."""
    mock_predictor = DummySarcasmPredictor()
    monkeypatch.setattr("src.main.SarcasmPredictor", lambda: mock_predictor)

    with TestClient(app) as test_client:
        test_client.app.state.predictor = mock_predictor
        app.dependency_overrides[get_predictor] = lambda: mock_predictor
        yield test_client

    app.dependency_overrides.clear()
    if hasattr(app.state, "predictor"):
        app.state.predictor = None


def test_root_endpoint(client):
    """Verify root GET / returns welcome message and documentation links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Welcome" in data["message"]
    assert data["docs_url"] == "/docs"
    assert data["health_url"] == "/health"


def test_health_check_healthy(client):
    """Verify GET /health reports healthy when model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["device"] == "cpu"


def test_health_check_degraded():
    """Verify GET /health reports degraded when predictor is None."""
    app.dependency_overrides.clear()
    app.state.predictor = None

    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False


def test_predict_success(client):
    """Verify POST /predict returns valid prediction on normal input."""
    payload = {"text": "Man Finally Finishes Reading Terms and Conditions"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["prediction"] == "SARCASTIC"
    assert data["data"]["confidence"] == 98.75
    assert data["data"]["is_sarcastic"] is True


def test_predict_empty_text_rejected(client):
    """Verify POST /predict returns 422 on empty string."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_whitespace_only_rejected(client):
    """Verify POST /predict returns 422 on blank whitespace string."""
    response = client.post("/predict", json={"text": "     "})
    assert response.status_code == 422


def test_predict_too_long_rejected(client):
    """Verify POST /predict returns 422 when text exceeds 500 characters."""
    response = client.post("/predict", json={"text": "x" * 501})
    assert response.status_code == 422


def test_predict_service_unavailable_when_model_missing():
    """Verify POST /predict returns 503 if model is not loaded."""
    app.dependency_overrides.clear()
    app.state.predictor = None

    with TestClient(app) as test_client:
        response = test_client.post("/predict", json={"text": "Valid headline text here"})
        assert response.status_code == 503
        assert "not loaded or temporarily unavailable" in response.json()["detail"]


def test_versioned_predict_endpoint(client):
    """Verify POST /api/v1/predict works identically and conforms to PredictionResponse."""
    payload = {"text": "Scientists Invent New Flavor of Water"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "prediction" in data["data"]
    assert "confidence" in data["data"]
    assert "is_sarcastic" in data["data"]


def test_metrics_endpoint(client):
    """Verify GET /metrics exposes Prometheus metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "sarcasm_prediction_requests_total" in response.text


def test_cors_headers(client):
    """Verify that CORS middleware returns allowed origin headers."""
    response = client.options(
        "/api/v1/predict",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
