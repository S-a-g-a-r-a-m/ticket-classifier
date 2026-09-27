import json
import logging
import time

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

app = FastAPI()

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

@app.get("/metrics")
def metrics():
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