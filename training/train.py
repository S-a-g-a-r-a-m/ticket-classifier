import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
import joblib
import json

DATA_PATH = "data/tickets.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df[df["language"] == "en"].copy()

    df["text"] = (
        df["subject"].fillna("")
        + " "
        + df["body"].fillna("")
    )

    df = df[["text", "queue"]].dropna().drop_duplicates()

    df = df[
        df["text"].str.strip().str.len() > 10
    ]

    return df


if __name__ == "__main__":
    df = load_data()
    X = df["text"]
    y = df["queue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTraining samples:", len(X_train))
    print("Test samples:", len(X_test))

    print("Dataset shape:", df.shape)
    print("\nClass distribution:")
    print(df["queue"].value_counts())

PRODUCTION_MODEL_PATH = "models/ticket_classifier.joblib"

production_model = joblib.load(PRODUCTION_MODEL_PATH)

production_predictions = production_model.predict(X_test)

production_f1 = f1_score(
    y_test,
    production_predictions,
    average="macro",
)

print("\nProduction model Macro F1:", round(production_f1, 4))

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        ),
    ),
    (
        "clf",
        LinearSVC(
            class_weight="balanced"
        ),
    ),
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
)

print("\nMacro F1:", round(macro_f1, 4))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
    )
)

print("\nModel trained successfully.")
print("Predictions:", len(predictions))


MODEL_PATH = "models/ticket_classifier_candidate.joblib"
METADATA_PATH = "models/candidate_metadata.json"

joblib.dump(model, MODEL_PATH)

metadata = {
    "model_version": "candidate",
    "data_version": "v1",
    "algorithm": "LinearSVC",
    "macro_f1": round(float(macro_f1), 4),
    "tfidf_ngram_range": [1, 2],
    "tfidf_min_df": 2,
    "test_size": 0.2,
    "random_state": 42,
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Metadata saved to: {METADATA_PATH}")