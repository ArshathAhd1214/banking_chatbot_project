# Banking Assistant Chatbot (Text-Based)

A text-based **banking assistant** chatbot that uses:
- **NLP** (tokenization, lemmatization) with NLTK
- **Simple ML classifier** (Logistic Regression) for intent recognition
- **Hybrid inference engine** (rule-based + ML)
- **SQLite database** for dynamic facts and learned Q&A
- **On-the-fly learning** from user feedback (stored into the DB)
- **CLI interface**

This project is tailored to the assignment brief:
- Natural Language Interface (text-based)
- Inference Engine mapping questions to answers
- Database/Knowledge Base with persistent storage
- Indication of **machine learning** via training and online learning storage
- Documentation with PEAS, architecture, snippets, algorithms, test data

## Quick Start

```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) First-time NLTK setup (downloads wordnet + punkt if needed)
python -m chatbot.setup_nltk

# 4) Run the chatbot (CLI)
python -m chatbot.app
```

## Project Structure

```
chatbot/
  app.py                 # CLI loop + user interaction
  nlp.py                 # NLP pipeline (tokenize, lemmatize, normalize)
  inference.py           # Rule-based + ML hybrid inference
  db.py                  # SQLite helpers and seed data
  training.py            # Train/load the ML model
  setup_nltk.py          # One-time NLTK downloads
  config.py              # Config and constants
data/
  seed.sql               # DB schema and seed values
docs/
  Technical_Documentation.md  # Assignment-aligned documentation
  Viva_Demo_Script.md         # Short presentation + demo flow
tests/
  test_cases.csv         # Example test inputs/expected intent/notes
requirements.txt
README.md
```

## Build a Standalone Executable (Optional)

> _This is optional and for your local machine._  
> You can make a single-file exe using **PyInstaller**.

```bash
pip install pyinstaller
pyinstaller --onefile -n banking_chatbot chatbot/app.py
# The executable will be in the dist/ folder.
```

## Learning Feature (Machine Learning)

- Unknown queries are stored in `user_learned_qa` with the user's provided answer (after confirmation).
- These can be reviewed later and optionally merged into the main knowledge base or used to retrain the ML classifier.

## Domain

Focused on **banking**: account types, interest rates, branch locations/hours, loan info, ATM availability, and small talk.
```
