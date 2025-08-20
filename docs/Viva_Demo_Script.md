# Viva Demo Script (5–7 minutes)

1) **Intro (30s)**
- Domain: Banking assistant (text-based)
- Components: NLP, ML intent classification, SQLite knowledge base, learning

2) **Architecture (1 min)**
- Show the diagram from Technical_Documentation.md
- Explain flow: user → NLP → classifier → inference → DB

3) **Live Demo (3–4 min)**
- Greeting: “hello” → smalltalk response
- Account types: “what account types do you have?”
- Loan rates: “what is the loan interest rate today?”
- Branch hours: “when are you open on weekends?”
- Unknown query (e.g., “do you have student loan top-ups?”) → bot asks to learn → provide answer → confirm saved

4) **Learning (30s)**
- Show `user_learned_qa` entries exist in DB

5) **Wrap-up (30s)**
- Summarize PEAS, ML + rule-based hybrid, persistence
- Mention how to extend with entities (cities) and more intents
