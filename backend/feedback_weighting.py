"""
feedback_weighting.py — Chunk-Level Relevance Feedback (heuristic demo)
"""

from typing import Dict, List
import json
import os
import mlflow

WEIGHT_LOG_FILE = "retrieval_feedback_log.json"


class FeedbackWeightTracker:
    def __init__(self, nudge_step: float = 0.05, min_weight: float = 0.2, max_weight: float = 0.8):
        self.nudge_step = nudge_step
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.bm25_weight = 0.4
        self.semantic_weight = 0.6
        self._history: List[Dict] = []
        self._load()

    def record_feedback(self, dominant_source: str, rating: bool):
        self._history.append({"source": dominant_source, "rating": rating})
        self._adjust_weights(dominant_source, rating)
        self._save()

    def _adjust_weights(self, dominant_source: str, rating: bool):
        if rating:
            return

        if dominant_source == "bm25":
            self.bm25_weight     = max(self.min_weight, self.bm25_weight - self.nudge_step)
            self.semantic_weight = 1.0 - self.bm25_weight
        elif dominant_source == "semantic":
            self.semantic_weight = max(self.min_weight, self.semantic_weight - self.nudge_step)
            self.bm25_weight     = 1.0 - self.semantic_weight

        with mlflow.start_run(run_name="feedback_weight_adjustment"):
            mlflow.log_metric("bm25_weight", round(self.bm25_weight, 3))
            mlflow.log_metric("semantic_weight", round(self.semantic_weight, 3))
            mlflow.log_param("triggered_by", dominant_source)

    def get_weights(self) -> Dict[str, float]:
        return {"bm25": round(self.bm25_weight, 3), "semantic": round(self.semantic_weight, 3)}

    def _save(self):
        try:
            with open(WEIGHT_LOG_FILE, "w") as f:
                json.dump({
                    "bm25_weight": self.bm25_weight,
                    "semantic_weight": self.semantic_weight,
                    "history": self._history[-200:]
                }, f, indent=2)
        except Exception:
            pass

    def _load(self):
        if not os.path.exists(WEIGHT_LOG_FILE):
            return
        try:
            with open(WEIGHT_LOG_FILE) as f:
                data = json.load(f)
            self.bm25_weight     = data.get("bm25_weight", 0.4)
            self.semantic_weight = data.get("semantic_weight", 0.6)
            self._history        = data.get("history", [])
        except Exception:
            pass


def determine_dominant_source(docs: list, bm25_doc_keys: set) -> str:
    bm25_count = sum(1 for doc in docs if doc.page_content[:150] in bm25_doc_keys)
    semantic_count = len(docs) - bm25_count
    return "bm25" if bm25_count >= semantic_count else "semantic"


feedback_weight_tracker = FeedbackWeightTracker()
