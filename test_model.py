import joblib

model = joblib.load("models/ticket_classifier.joblib")

text = "I cannot login to my account"

prediction = model.predict([text])

print(prediction)