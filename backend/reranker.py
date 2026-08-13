from langchain_core.documents import Document
from typing import List
import mlflow

# Lazy-loaded — the model is only downloaded/instantiated on first actual
# use, not at import time. This keeps module import fast, avoids blocking
# app startup on a network call, and lets pure-logic functions in this
# file (e.g. reorder_lost_in_the_middle) be imported/tested without
# needing model weights available at all.
_reranker_model = None


def _get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_model


def rerank(query: str, docs: List[Document], top_k: int = 3) -> List[Document]:
    if not docs:
        return []

    reranker_model = _get_reranker_model()
    pairs  = [(query, doc.page_content) for doc in docs]
    scores = reranker_model.predict(pairs)

    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    reranked    = [doc for _, doc in scored_docs[:top_k]]

    with mlflow.start_run(run_name="reranking"):
        mlflow.log_metric("docs_before_rerank", len(docs))
        mlflow.log_metric("docs_after_rerank",  len(reranked))
        mlflow.log_metric("top_score",    round(float(scored_docs[0][0]), 3))
        mlflow.log_metric("bottom_score", round(float(scored_docs[-1][0]), 3))

    return reranked


def reorder_lost_in_the_middle(docs: List[Document]) -> List[Document]:
    if len(docs) <= 2:
        return docs

    front, back = [], []
    for i, doc in enumerate(docs):
        if i % 2 == 0:
            front.append(doc)
        else:
            back.append(doc)

    reordered = front + list(reversed(back))

    with mlflow.start_run(run_name="lost_in_middle_reorder"):
        mlflow.log_metric("docs_reordered", len(reordered))

    return reordered


def rerank_and_reorder(query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    reranked  = rerank(query, docs, top_k=top_k)
    reordered = reorder_lost_in_the_middle(reranked)
    return reordered


def rerank_with_scores(query: str, docs: List[Document], top_k: int = 3) -> list:
    if not docs:
        return []
    reranker_model = _get_reranker_model()
    pairs  = [(query, doc.page_content) for doc in docs]
    scores = reranker_model.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [
        {"document": doc, "score": round(float(score), 4), "content": doc.page_content[:200]}
        for score, doc in scored_docs[:top_k]
    ]
