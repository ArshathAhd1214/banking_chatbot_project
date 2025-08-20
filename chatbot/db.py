import sqlite3
from pathlib import Path
from typing import Optional, List, Dict
from .config import DB_PATH

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS smalltalk (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern TEXT NOT NULL,  -- simple LIKE pattern (handled in code)
    response TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS intents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS intent_examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    intent_id INTEGER NOT NULL,
    example TEXT NOT NULL,
    FOREIGN KEY(intent_id) REFERENCES intents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_learned_qa (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    approved INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_text TEXT NOT NULL,
    bot_intent TEXT,
    confidence REAL,
    bot_answer TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL,
    helpful INTEGER,                 -- 1=thumbs up, 0=thumbs down, NULL=not given
    correction_intent TEXT,          -- optional: user supplied true intent
    corrected_answer TEXT,           -- optional: better answer text
    approved INTEGER DEFAULT 1,      -- human-in-the-loop approval
    created_at TEXT NOT NULL,
    FOREIGN KEY(interaction_id) REFERENCES interactions(id) ON DELETE CASCADE
);
"""

SEED = [
    ("account_types", [
        "what account do you have", "type of account", "list account type",
        "account options", "savings or current"
    ]),
    ("loan_rates", [
        "what is loan rate", "interest rate for loan",
        "today loan interest", "loan interest"
    ]),
    ("branch_hours", [
        "when branch open", "branch timing", "what are working hour",
        "bank open time"
    ]),
    ("atm_availability", [
        "nearest atm", "is atm available", "atm location", "atm near me"
    ]),
    ("greeting", ["hi", "hello", "good morning", "good evening"]),
    ("goodbye", ["bye", "goodbye", "see you"]),
    ("thanks", ["thanks", "thank you", "much appreciated"]),
]

SMALLTALK = [
    ("%hello%", "Hello! How can I assist you with banking today?"),
    ("%hi%", "Hi there! What banking info do you need?"),
    ("%good morning%", "Good morning! How can I help?"),
    ("%thank%", "You're welcome! Anything else?"),
    ("%bye%", "Goodbye! Have a great day."),
]

FACTS = [
    ("account_types", "Savings Account, Current Account, Fixed Deposit (FD), Student Savings, Senior Savings"),
    ("loan_personal_rate", "16.5% p.a."),
    ("loan_home_rate", "13.9% p.a."),
    ("loan_auto_rate", "14.7% p.a."),
    ("branch_hours_weekday", "Mon–Fri: 9:00–15:00"),
    ("branch_hours_weekend", "Sat: 9:00–12:00; Sun: Closed"),
]


# ---------------------------
# Connection & Initialization
# ---------------------------
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    con = connect()
    cur = con.cursor()
    cur.executescript(SCHEMA)

    # Seed intents & examples
    for intent_name, examples in SEED:
        cur.execute(
            "INSERT OR IGNORE INTO intents(name, description) VALUES (?, ?)",
            (intent_name, f"Intent for {intent_name}")
        )
        cur.execute("SELECT id FROM intents WHERE name=?", (intent_name,))
        intent_id_row = cur.fetchone()
        if intent_id_row:
            intent_id = intent_id_row[0]
            for ex in examples:
                cur.execute(
                    "INSERT INTO intent_examples(intent_id, example) VALUES (?, ?)",
                    (intent_id, ex)
                )

    # Seed smalltalk
    for pattern, resp in SMALLTALK:
        cur.execute(
            "INSERT INTO smalltalk(pattern, response) VALUES (?, ?)",
            (pattern, resp)
        )

    # Seed facts
    from datetime import datetime
    now = datetime.utcnow().isoformat()
    for k, v in FACTS:
        cur.execute(
            "INSERT OR REPLACE INTO facts(key, value, updated_at) VALUES (?, ?, ?)",
            (k, v, now)
        )

    con.commit()
    con.close()


# ---------------------------
# Query helpers
# ---------------------------
def get_smalltalk_matches(text_norm: str) -> Optional[str]:
    con = connect()
    cur = con.cursor()
    cur.execute("SELECT pattern, response FROM smalltalk")
    for row in cur.fetchall():
        patt = row["pattern"].lower().strip("%")
        if patt and patt in text_norm:
            return row["response"]
    return None


def get_fact(key: str) -> Optional[str]:
    con = connect()
    cur = con.cursor()
    cur.execute("SELECT value FROM facts WHERE key=?", (key,))
    r = cur.fetchone()
    return r[0] if r else None


# ---------------------------
# Learning & Feedback helpers
# ---------------------------
def record_user_learning(question: str, answer: str) -> None:
    con = connect()
    cur = con.cursor()
    from datetime import datetime
    cur.execute(
        "INSERT INTO user_learned_qa(question, answer, created_at) VALUES (?, ?, ?)",
        (question, answer, datetime.utcnow().isoformat())
    )
    con.commit()


def list_user_learned(only_unapproved: bool = True) -> List[Dict]:
    con = connect()
    cur = con.cursor()
    if only_unapproved:
        cur.execute("""
            SELECT id, question, answer, approved, created_at
            FROM user_learned_qa
            WHERE approved=0
            ORDER BY id DESC
        """)
    else:
        cur.execute("""
            SELECT id, question, answer, approved, created_at
            FROM user_learned_qa
            ORDER BY id DESC
        """)
    return [dict(r) for r in cur.fetchall()]


def record_interaction(
    user_text: str,
    bot_intent: Optional[str],
    confidence: Optional[float],
    bot_answer: str
) -> int:
    con = connect()
    cur = con.cursor()
    from datetime import datetime
    cur.execute(
        "INSERT INTO interactions(user_text, bot_intent, confidence, bot_answer, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (user_text, bot_intent, confidence or 0.0, bot_answer, datetime.utcnow().isoformat())
    )
    con.commit()
    return cur.lastrowid


def record_feedback(
    interaction_id: int,
    helpful: Optional[bool],
    correction_intent: Optional[str],
    corrected_answer: Optional[str]
) -> None:
    con = connect()
    cur = con.cursor()
    from datetime import datetime
    cur.execute(
        "INSERT INTO feedback(interaction_id, helpful, correction_intent, corrected_answer, approved, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            interaction_id,
            (1 if helpful else 0) if helpful is not None else None,
            correction_intent,
            corrected_answer,
            1,
            datetime.utcnow().isoformat()
        )
    )
    con.commit()


def get_feedback_training_data() -> List[Dict]:
    """
    Returns approved feedback items joined with their original user_text.
    """
    con = connect()
    cur = con.cursor()
    cur.execute("""
        SELECT i.user_text, f.correction_intent, f.corrected_answer, f.helpful
        FROM feedback f
        JOIN interactions i ON i.id = f.interaction_id
        WHERE f.approved = 1
        ORDER BY f.id DESC
    """)
    return [dict(r) for r in cur.fetchall()]
