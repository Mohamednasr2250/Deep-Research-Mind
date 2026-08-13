import json
import os
import mlflow
from datetime import datetime

FEEDBACK_FILE = "feedback_log.json"


def load_feedback() -> list:
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return []


def save_feedback(feedback: list):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback, f, indent=2)


def log_feedback(query: str, answer: str, rating: bool, comment: str = ""):
    feedback = load_feedback()

    entry = {
        "timestamp": datetime.now().isoformat(),
        "query":     query,
        "answer":    answer[:200],
        "rating":    rating,
        "comment":   comment
    }

    feedback.append(entry)
    save_feedback(feedback)

    with mlflow.start_run(run_name="user_feedback"):
        mlflow.log_param("query",   query)
        mlflow.log_param("rating",  "positive" if rating else "negative")
        mlflow.log_metric("is_positive", 1 if rating else 0)

    return entry


def get_feedback_stats() -> dict:
    feedback = load_feedback()

    if not feedback:
        return {"total": 0, "positive": 0, "negative": 0, "satisfaction_rate": 0}

    positive = sum(1 for f in feedback if f["rating"])
    negative = len(feedback) - positive

    return {
        "total":             len(feedback),
        "positive":          positive,
        "negative":          negative,
        "satisfaction_rate": round(positive / len(feedback) * 100, 1),
        "recent_negative":   [f for f in feedback if not f["rating"]][-5:]
    }
