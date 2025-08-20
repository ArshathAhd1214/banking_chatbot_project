import re
from typing import List
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

lemmatizer = WordNetLemmatizer()

def _safe_word_tokenize(text: str):
    try:
        return word_tokenize(text)
    except LookupError:
        # Try to fetch missing tokenizers on the fly
        for pkg in ("punkt", "punkt_tab"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        try:
            return word_tokenize(text)
        except Exception:
            # Last-resort regex fallback: split on non-word chars
            return re.findall(r"[A-Za-z0-9]+", text.lower())

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = _safe_word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmas)

def tokenize(text: str) -> List[str]:
    return normalize(text).split()
