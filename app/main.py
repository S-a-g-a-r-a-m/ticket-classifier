import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

model = joblib.load("models/ticket_classifier.joblib")


class TicketRequest(BaseModel):
    text: str = Field(min_length=1)


@app.get("/")
def home():
    return {"message": "Ticket classifier API is running"}


@app.post("/predict")
def predict(request: TicketRequest):
    prediction = model.predict([request.text])

    return {"queue": prediction[0]}