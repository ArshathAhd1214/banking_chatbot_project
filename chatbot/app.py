from .db import (
    init_db,
    record_user_learning,
    record_interaction,
    record_feedback,
)
from .training import train_model, load_model
from .inference import infer_intent_and_answer

BANNER = """
============================
  BANKING ASSISTANT (CLI)
============================
Type your question, or:
  :help   Show commands
  :train  Retrain ML model from DB examples
  :quit   Exit
"""

def handle_help():
    print(":help   Show this message\n:train  Retrain model\n:quit   Exit\n")

def main():
    print(BANNER)
    init_db()

    # Try to train; if it fails, try loading an existing model
    try:
        model, report = train_model(save=True)
        print("Model trained. Mini-report:\n" + report)
    except Exception as e:
        print("Warning: Could not train model:", e)
        try:
            model = load_model()
            print("Loaded existing model from disk.")
        except Exception as e2:
            print("No trained model available:", e2)
            print("Please install requirements and run a training step first.")
            return

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Bye!")
            break

        if not user:
            continue

        if user.startswith(":"):
            cmd = user[1:].strip().lower()
            if cmd == "quit":
                break
            if cmd == "help":
                handle_help()
                continue
            if cmd == "train":
                try:
                    model, report = train_model(save=True)
                    print("Model retrained. Mini-report:\n" + report)
                except Exception as e:
                    print("Training failed:", e)
                continue
            print("Unknown command. Type :help")
            continue

        # -------- Inference --------
        intent, answer, conf = infer_intent_and_answer(model, user)

        if answer:
            print(f"bot> {answer} (intent={intent}, conf={conf:.2f})")

            # Log the interaction
            iid = record_interaction(
                user_text=user,
                bot_intent=intent,
                confidence=conf,
                bot_answer=answer,
            )

            # Quick feedback loop
            fb = input(
                "bot> Was this helpful? (y/n) or type 'correct <intent>' or 'fix <better answer>': "
            ).strip().lower()

            if fb in ("y", "yes"):
                record_feedback(iid, helpful=True, correction_intent=None, corrected_answer=None)
            elif fb in ("n", "no"):
                record_feedback(iid, helpful=False, correction_intent=None, corrected_answer=None)
            elif fb.startswith("correct "):
                true_intent = fb.split(" ", 1)[1].strip() or None
                record_feedback(iid, helpful=None, correction_intent=true_intent, corrected_answer=None)
            elif fb.startswith("fix "):
                better = fb.split(" ", 1)[1].strip() or None
                record_feedback(iid, helpful=None, correction_intent=None, corrected_answer=better)
            else:
                # no explicit feedback given
                pass

            continue

        # -------- Low-confidence path (teach flow) --------
        print(
            f"bot> I'm not sure about that (confidence={conf:.2f}). "
            "Would you like to teach me the answer? (yes/no)"
        )
        yn = input("you> ").strip().lower()
        if yn in ("y", "yes"):
            print("bot> Please type the correct answer I should give next time:")
            teach = input("you> ").strip()
            if teach:
                record_user_learning(user, teach)
                # Log interaction with no intent/low conf, using the user-provided answer as bot_answer for traceability
                record_interaction(
                    user_text=user,
                    bot_intent=None,
                    confidence=conf,
                    bot_answer=f"[learned candidate] {teach}",
                )
                print("bot> Thanks! I've saved that. A human can review and add it to my knowledge.")
            else:
                print("bot> No worries. Ask me something else!")
        else:
            print("bot> Okay! Ask me something else.")

if __name__ == "__main__":
    main()
