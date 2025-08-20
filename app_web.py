from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='.', static_url_path='')
# Allow frontends on other ports (e.g., Live Server :5500). If serving same-origin, CORS is harmless but optional.
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.get("/")
def home():
    return send_from_directory(".", "index.html")

# --- Fake inference (replace with your real pipeline) ---
def fake_infer(user_text: str):
    t = user_text.lower()
    if "account" in t:
        return ("account_types", "We currently offer: Savings, Current, FD, Student Savings, Senior Savings.", 0.99)
    if "loan" in t or "rate" in t:
        return ("loan_rates", "Personal: 16.5% p.a., Home: 13.9% p.a., Auto: 14.7% p.a.", 0.98)
    if "hour" in t or "open" in t:
        return ("branch_hours", "Weekdays 9:00–15:00, Sat 9:00–12:00, Sun closed.", 0.97)
    if "atm" in t:
        return ("atm_availability", "ATMs are available 24/7 at most branches.", 0.95)
    if any(x in t for x in ["hi","hello","hey"]):
        return ("greeting", "Hi there! What banking info do you need?", 1.00)
    return ("unknown", "I'm not fully sure. Please teach me a better answer using the feedback buttons.", 0.2)

# --- APIs ---
@app.route("/api/chat", methods=["POST", "OPTIONS"])
def api_chat():
    if request.method == "OPTIONS":
        return ("", 204)  # preflight OK for cross-origin
    data = request.get_json(silent=True) or {}
    text = (data.get("message") or "").strip()
    intent, reply, conf = fake_infer(text)
    # in a real app, record to DB and return that interaction id
    return jsonify({"reply": reply, "intent": intent, "confidence": conf, "interaction_id": 1})

@app.route("/api/feedback", methods=["POST", "OPTIONS"])
def api_feedback():
    if request.method == "OPTIONS":
        return ("", 204)
    # accept { interaction_id, helpful?, correction_intent?, corrected_answer? }
    return jsonify({"ok": True})

@app.route("/api/train", methods=["POST", "OPTIONS"])
def api_train():
    if request.method == "OPTIONS":
        return ("", 204)
    # call your training routine here
    return jsonify({"ok": True, "report": "Retrained on latest examples."})

if __name__ == "__main__":
    # Run on :5000 so the index.html autodetection works
    app.run(host="127.0.0.1", port=5000, debug=False)
