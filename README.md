# Support Ticket Classifier
Classifies customer support tickets into queues using TF-IDF + scikit-learn.
## Dataset
Customer IT Support Ticket Dataset (Kaggle) – English subset, N rows, K 
queues.
## Approach
Cleaning → stratified 80/20 split → TF-IDF (1-2 grams) → 3 classifiers 
compared.
## Results
| Model | Macro F1 |
|---|---|
| Baseline (most frequent) | 0.xx |
| Logistic Regression | 0.xx |
| Naive Bayes | 0.xx |
| Linear SVC | 0.xx |
## What I'd improve- Sentence embeddings instead of TF-IDF- Compare with an LLM zero-shot classifier- Tune hyperparameters with GridSearchCV
## Run it
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r 
requirements.txt