import joblib
from fastapi import FastAPI

app = FastAPI()

model = joblib.load("models/ticket_classifier.joblib")


@app.get("/")
def home():
    return {"message": "Ticket classifier API is running"}


@app.post("/predict")
def predict(text: str):
    prediction = model.predict([text])

    return {"queue": prediction[0]}