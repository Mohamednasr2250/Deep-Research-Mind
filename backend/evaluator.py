"""
evaluator.py — RAG Evaluation Suite

Two evaluation layers:
1. RAGAS — answer quality (faithfulness, relevancy, context precision/recall)
2. MRR + NDCG — retrieval ranking quality (did we rank the relevant chunks near the top?)
"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
from sklearn.metrics import ndcg_score
import numpy as np
import mlflow


def evaluate_rag(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str]
) -> dict:
    """
    Evaluate RAG answer quality using RAGAS metrics.

    faithfulness      → is answer faithful to retrieved context?
    answer_relevancy  → is answer relevant to the question?
    context_precision → are retrieved chunks actually useful?
    context_recall    → did we retrieve all needed information?
    """
    data = {
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )

    scores = {
        "faithfulness":      round(float(results["faithfulness"]), 3),
        "answer_relevancy":  round(float(results["answer_relevancy"]), 3),
        "context_precision": round(float(results["context_precision"]), 3),
        "context_recall":    round(float(results["context_recall"]), 3),
    }
    scores["overall"] = round(sum(scores.values()) / len(scores), 3)

    with mlflow.start_run(run_name="ragas_evaluation"):
        for metric, score in scores.items():
            mlflow.log_metric(metric, score)

    return scores


def evaluate_retrieval(
    retrieved_docs: list,
    relevant_docs: list,
    k: int = 5
) -> dict:
    """
    Evaluate retrieval RANKING quality — separate from answer quality.

    MRR (Mean Reciprocal Rank):
        Finds the rank of the FIRST relevant doc, scores 1/rank.
        Rank 1 → 1.0, Rank 4 → 0.25. Only cares about the first hit.

    NDCG (Normalized Discounted Cumulative Gain):
        Rewards relevant docs near the top, accounts for the WHOLE ranked list,
        not just the first hit. Score normalized 0–1 against the ideal ranking.

    retrieved_docs: list of doc identifiers (e.g. page_content[:100] or doc IDs)
                    in the order they were retrieved/reranked
    relevant_docs:  list of doc identifiers known to actually be relevant
                    (from a labeled eval set)
    """
    # ── MRR ──
    mrr = 0.0
    for i, doc in enumerate(retrieved_docs[:k], start=1):
        if doc in relevant_docs:
            mrr = 1.0 / i
            break

    # ── NDCG ──
    relevance_scores = [
        1 if doc in relevant_docs else 0
        for doc in retrieved_docs[:k]
    ]
    ideal = sorted(relevance_scores, reverse=True)

    if sum(ideal) == 0:
        ndcg = 0.0
    else:
        ndcg = float(ndcg_score([ideal], [relevance_scores]))

    result = {
        "mrr":  round(mrr, 3),
        "ndcg": round(ndcg, 3),
        "k":    k
    }

    with mlflow.start_run(run_name="retrieval_evaluation"):
        mlflow.log_metric("mrr",  result["mrr"])
        mlflow.log_metric("ndcg", result["ndcg"])
        mlflow.log_param("k",     k)

    return result


def evaluate_retrieval_batch(
    eval_set: list[dict],
    retriever,
    k: int = 5
) -> dict:
    """
    Run MRR + NDCG across a full labeled eval set and average results.

    eval_set format:
    [
        {"query": "what is attention?", "relevant_docs": ["doc_id_1", "doc_id_3"]},
        ...
    ]

    retriever must expose .retrieve(query, k) -> list of docs with .page_content
    """
    all_mrr  = []
    all_ndcg = []

    for item in eval_set:
        query         = item["query"]
        relevant_docs = item["relevant_docs"]

        docs       = retriever.retrieve(query, k=k)
        doc_ids    = [doc.page_content[:100] for doc in docs]

        scores = evaluate_retrieval(doc_ids, relevant_docs, k=k)
        all_mrr.append(scores["mrr"])
        all_ndcg.append(scores["ndcg"])

    avg_result = {
        "avg_mrr":   round(float(np.mean(all_mrr)), 3),
        "avg_ndcg":  round(float(np.mean(all_ndcg)), 3),
        "num_queries": len(eval_set)
    }

    with mlflow.start_run(run_name="retrieval_eval_batch"):
        mlflow.log_metric("avg_mrr",  avg_result["avg_mrr"])
        mlflow.log_metric("avg_ndcg", avg_result["avg_ndcg"])
        mlflow.log_metric("num_queries", avg_result["num_queries"])

    return avg_result
