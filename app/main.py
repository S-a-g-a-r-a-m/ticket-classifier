import json
import logging
import os
import secrets
import time

import joblib
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

app = FastAPI()

security = HTTPBasic()

model = joblib.load("models/ticket_classifier.joblib")

prediction_counter = Counter(
    "ticket_predictions_total",
    "Total number of ticket predictions",
    ["queue"],
)

prediction_latency = Histogram(
    "ticket_prediction_latency_seconds",
    "Time spent performing ticket predictions",
)

prediction_errors = Counter(
    "ticket_prediction_errors_total",
    "Total number of prediction errors",
)


class TicketRequest(BaseModel):
    text: str = Field(min_length=1)


@app.get("/")
def home():
    return {"message": "Ticket classifier API is running"}

def verify_metrics_credentials(
    credentials: HTTPBasicCredentials = Depends(security),
):
    expected_username = os.getenv("METRICS_USERNAME")
    expected_password = os.getenv("METRICS_PASSWORD")

    if not expected_username or not expected_password:
        raise HTTPException(
            status_code=500,
            detail="Metrics authentication is not configured",
        )

    username_correct = secrets.compare_digest(
        credentials.username,
        expected_username,
    )

    password_correct = secrets.compare_digest(
        credentials.password,
        expected_password,
    )

    if not (username_correct and password_correct):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid metrics credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

@app.get("/metrics")
def metrics(
    _: None = Depends(verify_metrics_credentials),
):
    return Response(
        generate_latest(),
        media_type="text/plain",
    )

@app.post("/predict")
def predict(request: TicketRequest):
    start_time = time.perf_counter()

    try:
        with prediction_latency.time():
            prediction = model.predict([request.text])[0]

    except Exception:
        prediction_errors.inc()

        logger.exception(
            json.dumps({
                "event": "prediction_error"
            })
        )

        raise

    latency_ms = (time.perf_counter() - start_time) * 1000

    prediction_counter.labels(queue=prediction).inc()

    logger.info(
        json.dumps({
            "event": "prediction",
            "prediction": prediction,
            "latency_ms": round(latency_ms, 2),
        })
    )

    return {
        "queue": prediction,
        "latency_ms": round(latency_ms, 2),
    }