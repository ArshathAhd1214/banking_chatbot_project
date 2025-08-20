# Technical Documentation – Banking Assistant Chatbot

## 1. Problem Description
A text-based **banking assistant** that answers common banking questions (account types, loan rates, branch hours, ATM availability) and handles small talk. It combines NLP + ML intent classification with a rule-based inference layer and a **SQLite** knowledge base. Unknown questions can be **taught** by the user and stored in persistent storage for later review and retraining.

## 2. Research (Brief)
- **Chatbots**: Retrieval-based (rules/DB) vs **Generative** (LLMs). This project uses a **hybrid retrieval** approach for interpretability and grading alignment.
- **NLP Preprocessing**: tokenization, lemmatization improve generalization (e.g., “rates” → “rate”).
- **Intent Classification**: Lightweight **TF-IDF + Logistic Regression** is sufficient for small domains.
- **Knowledge Base**: SQLite for dynamic facts; static small talk via simple patterns.
- **Online Learning**: Store unknown Q&A for supervised curation and model retraining.

### Key References (examples)
- Jurafsky & Martin, *Speech and Language Processing* (NLP concepts)
- Scikit-learn user guide (text classification)
- NLTK documentation (tokenization/lemmatization)

## 3. Design Architecture
```
User → NLP (normalize) → Intent Classifier (ML) → Inference Engine
           ↓                              ↓
       Smalltalk DB                 SQLite Knowledge Base (facts)
           ↓                              ↓
        Direct reply                 Domain answers (SQL-backed)
```
- **Natural Language Interface**: CLI (stdin/stdout)
- **Inference Engine**: `inference.py` maps intent→response; first tries smalltalk, then ML intent; below threshold → ask user to teach → store to DB.
- **Database**: `facts`, `smalltalk`, `intents`, `intent_examples`, `user_learned_qa`.

## 4. P.E.A.S of the Bot
- **Performance**: Accuracy of intent classification; response relevance; successful DB retrieval; user satisfaction.
- **Environment**: Text console; local SQLite DB.
- **Actuators**: Text responses to console; writes to DB.
- **Sensors**: User text input; current knowledge base values.

## 5. Intelligence Traits Used
- **NLP**: tokenization + lemmatization in `nlp.py`.
- **Decision Making**: Confidence threshold routes to teach/clarify.
- **Learning**: Unknown Q&A persisted in `user_learned_qa` for later inclusion and retraining.
- **(Optional)**: You can extend with entity recognition (spaCy) to detect cities/amounts.

## 6. Algorithms (Flow)
**Conversation Handling (simplified pseudo-code):**
```
loop:
  user = read_input()
  if command: handle
  text_norm = normalize(user)
  if smalltalk_match(text_norm): reply_smalltalk; continue
  (intent, prob) = model.predict(text_norm)
  if prob >= threshold:
      answer = respond_for_intent(intent)  # may query SQLite facts
      print(answer)
  else:
      ask_user_to_teach()
      if yes: save to user_learned_qa(question, answer)
```

## 7. Key Code Snippets
### Lemmatizing (see `nlp.py`)
```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(t) for t in tokens]
```

### Random Small Talk (see DB `smalltalk` table)
Stored patterns like `%hello%` → “Hello! …”.

### Getting DB Answers (see `inference.py`)
```python
personal = get_fact("loan_personal_rate")
```

### Training the Bot (see `training.py`)
```python
model.fit(Xtr, ytr)
```

## 8. Test Plans and Data
See `tests/test_cases.csv` (inputs + expected intents). Manually validate responses and adjust thresholds/examples if needed.

## 9. Conclusion
The chatbot meets the assignment requirements with a hybrid design: deterministic where necessary and ML-driven for flexibility, plus persistent learning. It is easily extensible to more intents and richer entities.

## 10. References
- Scikit-learn documentation
- NLTK documentation
- SQLite documentation
```
