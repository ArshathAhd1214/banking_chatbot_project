import pickle
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .nlp import normalize
from .db import connect, get_feedback_training_data
from .config import MODEL_PATH


def load_training_data():
    """
    Load training data from intent_examples plus approved feedback corrections.
    Returns X (texts) and y (labels).
    """
    con = connect()
    cur = con.cursor()
    cur.execute("""
        SELECT i.name as intent, e.example as example
        FROM intent_examples e
        JOIN intents i ON i.id = e.intent_id
    """)
    rows = cur.fetchall()
    X = [normalize(r["example"]) for r in rows]
    y = [r["intent"] for r in rows]

    # Add approved feedback corrections
    fb = get_feedback_training_data()
    for item in fb:
        if item["correction_intent"]:
            X.append(normalize(item["user_text"]))
            y.append(item["correction_intent"])
        # Optionally: "fix <answer>" could be mined into KB or smalltalk

    return X, y


def train_model(save: bool = True) -> Tuple[Pipeline, str]:
    """
    Train a Logistic Regression intent classifier.
    Returns the model and a classification report.
    """
    X, y = load_training_data()
    if not X:
        raise RuntimeError("No training data found.")

    model = Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Mini report
    if len(set(y)) > 1:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model.fit(Xtr, ytr)
        report = classification_report(yte, model.predict(Xte), zero_division=0)
    else:
        # Only one class → train on all, skip report
        model.fit(X, y)
        report = "Only one intent class present. Model trained without test split."

    if save:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

    return model, report


def load_model() -> Pipeline:
    """Load the saved model from disk."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)
