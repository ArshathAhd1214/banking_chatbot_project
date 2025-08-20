from typing import Tuple, Optional
from .nlp import normalize
from .db import get_fact, get_smalltalk_matches
from .config import CONFIDENCE_THRESHOLD

# For business logic mapping from intent to responses
def respond_for_intent(intent: str) -> Optional[str]:
    if intent == "account_types":
        facts = get_fact("account_types")
        return f"We currently offer: {facts}." if facts else "We offer Savings, Current, and FD accounts."
    if intent == "loan_rates":
        personal = get_fact("loan_personal_rate") or "N/A"
        home = get_fact("loan_home_rate") or "N/A"
        auto = get_fact("loan_auto_rate") or "N/A"
        return f"Loan interest rates — Personal: {personal}, Home: {home}, Auto: {auto}."
    if intent == "branch_hours":
        wd = get_fact("branch_hours_weekday") or "Mon–Fri: 9–3"
        we = get_fact("branch_hours_weekend") or "Sat: 9–12; Sun: Closed"
        return f"Branch hours — Weekdays: {wd}. Weekends: {we}."
    if intent == "atm_availability":
        return "ATMs are available 24/7 at most branches. Please share your city to suggest nearby ATMs."
    if intent == "greeting":
        return "Hello! I’m your banking assistant. How can I help?"
    if intent == "goodbye":
        return "Goodbye! Happy to help anytime."
    if intent == "thanks":
        return "You're welcome! Anything else I can do?"
    return None

def smalltalk_or_none(text_norm: str) -> Optional[str]:
    return get_smalltalk_matches(text_norm)

def infer_intent_and_answer(model, user_text: str) -> Tuple[Optional[str], Optional[str], float]:
    text_norm = normalize(user_text)
    # smalltalk first
    st = smalltalk_or_none(text_norm)
    if st:
        return "smalltalk", st, 1.0
    # ML prediction
    probas = model.predict_proba([text_norm])[0]
    labels = model.classes_
    top_idx = probas.argmax()
    top_label = labels[top_idx]
    confidence = float(probas[top_idx])
    if confidence < CONFIDENCE_THRESHOLD:
        return None, None, confidence
    answer = respond_for_intent(top_label)
    return top_label, answer, confidence
