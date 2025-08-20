import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "banking_knowledge.db"

# ML artifacts
MODEL_DIR = BASE_DIR / "data"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
MODEL_PATH = MODEL_DIR / "intent_model.pkl"

CONFIDENCE_THRESHOLD = 0.45  # below this, ask user to teach
