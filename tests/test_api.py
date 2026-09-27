from fastapi.testclient import TestClient
from app.main import app, model
client = TestClient(app, raise_server_exceptions=False)

def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Ticket classifier API is running"
    )


def test_valid_prediction():
    response = client.post(
        "/predict",
        json={"text": "I want to return my order"}
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data["queue"], str)
    assert isinstance(data["latency_ms"], float)
    assert data["latency_ms"] >= 0


def test_empty_ticket():
    response = client.post(
        "/predict",
        json={"text": ""}
    )

    assert response.status_code == 422


def test_missing_text():
    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422

def test_prediction_error(monkeypatch):
    def failing_predict(text):
        raise RuntimeError("Test model failure")

    monkeypatch.setattr(
        model,
        "predict",
        failing_predict,
    )

    response = client.post(
        "/predict",
        json={"text": "This should trigger an error"}
    )

    assert response.status_code == 500

def test_metrics_requires_authentication():
    response = client.get("/metrics")

    assert response.status_code == 401

def test_metrics_with_authentication(monkeypatch):
    monkeypatch.setenv("METRICS_USERNAME", "testuser")
    monkeypatch.setenv("METRICS_PASSWORD", "testpassword")

    response = client.get(
        "/metrics",
        auth=("testuser", "testpassword"),
    )

    assert response.status_code == 200
    assert "ticket_predictions_total" in response.text