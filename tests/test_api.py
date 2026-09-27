from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


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
    assert isinstance(response.json()["queue"], str)


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