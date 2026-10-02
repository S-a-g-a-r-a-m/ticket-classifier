# Ticket Classifier — MLOps Pipeline

An end-to-end machine learning project that classifies customer support tickets into relevant support queues, with automated model training, evaluation, promotion, deployment, and production monitoring.

## Project Overview

This project extends a traditional machine learning classifier into a production-oriented MLOps system.

**Key features**

* Text classification using TF-IDF and LinearSVC.
* Automated candidate model training and evaluation.
* Model promotion based on Macro F1-score.
* Model versioning and rollback.
* REST API built with FastAPI.
* Automated testing and Docker image builds using GitHub Actions.
* Automated deployment to Render.
* Production metrics collection using Prometheus.
* Monitoring dashboards using Grafana Cloud.
* Basic authentication for the metrics endpoint.

## Architecture

```text
              Dataset
                 |
                 v
         Data Preprocessing
                 |
                 v
          Model Training
                 |
                 v
         Candidate Model
                 |
                 v
         Model Evaluation
                 |
          Compare Macro F1
                 |
         +-------+-------+
         |               |
      Better          Not Better
         |               |
         v               v
      Promote           Stop
         |
         v
    Commit Model
         |
         v
    GitHub Actions
         |
    +----+----+
    |         |
   Tests   Docker Build
    |         |
    +----+----+
         |
         v
    Render Deploy
         |
         v
     FastAPI API
         |
         v
 Prometheus Metrics
         |
         v
    Grafana Cloud
```

## Tech Stack

| Category            | Technology     |
| ------------------- | -------------- |
| Language            | Python         |
| Machine Learning    | Scikit-learn   |
| Feature Extraction  | TF-IDF         |
| Classifier          | LinearSVC      |
| API                 | FastAPI        |
| Testing             | Pytest         |
| Model Serialization | Joblib         |
| Containerization    | Docker         |
| CI/CD               | GitHub Actions |
| Deployment          | Render         |
| Metrics             | Prometheus     |
| Monitoring          | Grafana Cloud  |

## Model Development

### Data preprocessing

The dataset is filtered and prepared using the following steps:

* Keep English-language tickets.
* Combine ticket subject and body into one text field.
* Remove missing values and duplicate records.
* Remove very short ticket texts.
* Use the support queue as the target label.

### Model selection

Several machine learning algorithms were evaluated.

| Model               | Macro F1 |
| ------------------- | -------: |
| LinearSVC           |   0.7153 |
| Logistic Regression |   0.5744 |
| Naive Bayes         |   0.1843 |

LinearSVC was selected for the initial production model.

### Model improvement

The TF-IDF configuration was adjusted by changing `min_df` from `2` to `1`.

| Version | Algorithm | TF-IDF min_df | Macro F1 |
| ------- | --------- | ------------: | -------: |
| v1      | LinearSVC |             2 |   0.7153 |
| v2      | LinearSVC |             1 |   0.7297 |

The candidate model is evaluated against the current production model. A candidate is promoted only when its Macro F1-score is strictly higher.

## Project Structure

```text
ticket-classifier/
│
├── app/
│   └── main.py
│
├── data/
│   └── tickets.csv
│
├── models/
│   ├── ticket_classifier.joblib
│   ├── production_metadata.json
│   └── candidate_metadata.json
│
├── notebooks/
│   └── 01_explore.ipynb
│
├── training/
│   ├── train.py
│   ├── promote.py
│   └── rollback.py
│
├── tests/
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── train.yml
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Local Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd ticket-classifier
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 4. Run the API

```bash
uvicorn app.main:app --reload
```

The API is available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

| Method | Endpoint   | Description                               |
| ------ | ---------- | ----------------------------------------- |
| GET    | `/`        | API health check                          |
| POST   | `/predict` | Predict a ticket's support queue          |
| GET    | `/metrics` | Expose Prometheus metrics (authenticated) |

### Prediction example

Request:

```json
{
  "text": "I cannot access my account after resetting my password."
}
```

Example response:

```json
{
  "queue": "IT Support",
  "latency_ms": 2.1
}
```

The prediction response includes the predicted support queue and inference latency.

### Metrics

The `/metrics` endpoint exposes:

* `ticket_predictions_total`: Total predictions grouped by queue.
* `ticket_prediction_latency_seconds`: Prediction latency histogram.
* `ticket_prediction_errors_total`: Prediction errors.

Metrics access uses HTTP Basic authentication configured through environment variables:

```text
METRICS_USERNAME
METRICS_PASSWORD
```

## Model Lifecycle

### Train a candidate model

Run locally:

```bash
python training/train.py
```

This trains and evaluates a candidate model, saves the candidate artifact, and writes its metadata.

### Promote a candidate

```bash
python training/promote.py
```

The promotion script:

* Compares candidate and production Macro F1.
* Rejects candidates that do not improve the score.
* Backs up the current production model and metadata.
* Promotes an improved candidate to the next version.

### Rollback

```bash
python training/rollback.py
```

The current rollback script restores the v1 model and its metadata.

## Automated Training

The GitHub Actions workflow `train.yml` can be triggered manually.

It performs the following steps:

1. Checks out the repository.
2. Sets up Python.
3. Installs training dependencies.
4. Verifies the Python environment.
5. Trains a candidate model.
6. Evaluates and attempts promotion.
7. Commits the updated production model and metadata if they changed.

If the candidate does not outperform the production model, no production model commit is created.

## CI/CD

The `ci.yml` workflow runs on pushes and pull requests targeting `main`.

It performs:

* Automated API tests.
* Docker image build validation.
* Render deployment through a configured deploy hook.

Required GitHub secret:

```text
RENDER_DEPLOY_HOOK
```

The Render deployment is triggered after the test and Docker build jobs succeed.

## Docker

Build the image:

```bash
docker build -t ticket-classifier .
```

Run the container:

```bash
docker run -p 8000:8000 ticket-classifier
```

The API will be available at:

```text
http://localhost:8000
```

## Deployment

The API is deployed using Render.

Production URL:

https://ticket-classifier-tlzy.onrender.com

The deployed API provides prediction functionality and an authenticated metrics endpoint.

## Monitoring

Prometheus-compatible metrics are scraped by Grafana Cloud.

The monitoring dashboard includes:

### Predictions by queue

```promql
sum by (queue) (ticket_predictions_total)
```

### Average prediction latency

```promql
1000 * rate(ticket_prediction_latency_seconds_sum[5m])
/
rate(ticket_prediction_latency_seconds_count[5m])
```

### Prediction error rate

```promql
rate(ticket_prediction_errors_total[5m])
```

These panels provide visibility into prediction volume, latency, and API prediction errors.

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

The API tests cover:

* Health endpoint.
* Valid prediction.
* Invalid and missing input.
* Prediction error handling.
* Metrics authentication.
* Metrics endpoint access.

## Future Improvements

* Add model drift detection.
* Track model experiments and artifacts using MLflow.
* Add scheduled retraining.
* Improve rollback support for multiple model versions.
* Add data validation and model quality thresholds.
* Add more detailed production performance monitoring.

## Summary

This project demonstrates a practical MLOps workflow built around a traditional machine learning model. It connects model training and evaluation with version management, automated testing, containerization, cloud deployment, and production monitoring.

The focus is on making a machine learning model maintainable and deployable rather than stopping at offline model accuracy.
