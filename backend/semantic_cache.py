"""
semantic_cache.py — Semantic Query Caching
"""

import time
from typing import Optional, List, Dict
import mlflow


class SemanticCache:
    def __init__(self, embeddings_model, similarity_threshold: float = 0.92, ttl_seconds: int = 3600):
        self.embeddings_model     = embeddings_model
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds          = ttl_seconds
        self._entries: List[Dict] = []

    def _cosine_sim(self, v1: list, v2: list) -> float:
        import numpy as np
        v1, v2 = np.array(v1), np.array(v2)
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0:
            return 0.0
        return float(np.dot(v1, v2) / denom)

    def _is_expired(self, entry: Dict) -> bool:
        return (time.time() - entry["timestamp"]) > self.ttl_seconds

    def get(self, query: str, filters: Optional[dict] = None) -> Optional[dict]:
        self._entries = [e for e in self._entries if not self._is_expired(e)]

        if not self._entries:
            return None

        query_vec = self.embeddings_model.embed_query(query)

        best_match = None
        best_score = 0.0

        for entry in self._entries:
            if entry.get("filters") != filters:
                continue
            score = self._cosine_sim(query_vec, entry["embedding"])
            if score > best_score:
                best_score = score
                best_match = entry

        if best_match and best_score >= self.similarity_threshold:
            with mlflow.start_run(run_name="semantic_cache_hit"):
                mlflow.log_metric("similarity_score", round(best_score, 3))
            return {
                "answer":     best_match["answer"],
                "citations":  best_match.get("citations", []),
                "cache_hit":  True,
                "similarity": round(best_score, 3)
            }

        return None

    def set(self, query: str, answer: str, citations: Optional[list] = None, filters: Optional[dict] = None):
        query_vec = self.embeddings_model.embed_query(query)
        self._entries.append({
            "query":     query,
            "embedding": query_vec,
            "answer":    answer,
            "citations": citations or [],
            "filters":   filters,
            "timestamp": time.time()
        })

        if len(self._entries) > 500:
            self._entries = self._entries[-500:]

    def clear(self):
        self._entries = []

    def stats(self) -> dict:
        return {"cached_entries": len(self._entries), "threshold": self.similarity_threshold}
