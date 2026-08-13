import json
import os
from datetime import datetime
from typing import List, Dict
import mlflow

from evaluator import evaluate_rag, evaluate_retrieval_batch

EVAL_HISTORY_FILE = "eval_drift_history.json"
REGRESSION_THRESHOLD = 0.05


def load_eval_set(path: str = "fixed_eval_set.json") -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Create a fixed eval set of ~20-30 labeled "
            "Q&A examples before running continuous evaluation."
        )
    with open(path) as f:
        return json.load(f)


def run_scheduled_evaluation(hybrid_retriever, llm, eval_set_path: str = "fixed_eval_set.json") -> dict:
    eval_set = load_eval_set(eval_set_path)

    questions, answers, contexts, ground_truths = [], [], [], []
    retrieval_scores = []

    for item in eval_set:
        docs = hybrid_retriever.retrieve(item["question"], k=5)
        context_texts = [d.page_content for d in docs]
        answer = llm.invoke(f"Context:\n{chr(10).join(context_texts)}\n\nQuestion: {item['question']}\nAnswer:")

        questions.append(item["question"])
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(item["ground_truth"])

        retrieval_scores.append({"query": item["question"], "relevant_docs": item.get("relevant_docs", [])})

    ragas_scores = evaluate_rag(questions, answers, contexts, ground_truths)
    retrieval_batch_scores = evaluate_retrieval_batch(retrieval_scores, hybrid_retriever, k=5)

    run_record = {
        "timestamp": datetime.now().isoformat(),
        "ragas_scores": ragas_scores,
        "retrieval_scores": retrieval_batch_scores,
        "eval_set_size": len(eval_set)
    }

    regressions = _check_regressions(run_record)
    run_record["regressions_detected"] = regressions

    _save_run(run_record)

    with mlflow.start_run(run_name="continuous_evaluation"):
        for k, v in ragas_scores.items():
            mlflow.log_metric(f"ragas_{k}", v)
        mlflow.log_metric("mrr", retrieval_batch_scores["avg_mrr"])
        mlflow.log_metric("ndcg", retrieval_batch_scores["avg_ndcg"])
        mlflow.log_metric("regressions_flagged", len(regressions))

    if regressions:
        _send_alert(regressions)

    return run_record


def _check_regressions(current_run: dict) -> List[Dict]:
    history = _load_history()
    if not history:
        return []

    last_run = history[-1]
    regressions = []

    for metric, current_value in current_run["ragas_scores"].items():
        last_value = last_run["ragas_scores"].get(metric)
        if last_value is not None and (last_value - current_value) > REGRESSION_THRESHOLD:
            regressions.append({
                "metric": metric,
                "previous": last_value,
                "current": current_value,
                "drop": round(last_value - current_value, 3)
            })

    return regressions


def _send_alert(regressions: List[Dict]):
    message = "⚠️ RAG QUALITY REGRESSION DETECTED:\n" + "\n".join(
        f"- {r['metric']}: {r['previous']} -> {r['current']} (dropped {r['drop']})"
        for r in regressions
    )
    print(message)
    with mlflow.start_run(run_name="regression_alert"):
        mlflow.log_param("alert_message", message[:500])


def _load_history() -> List[Dict]:
    if not os.path.exists(EVAL_HISTORY_FILE):
        return []
    try:
        with open(EVAL_HISTORY_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def _save_run(record: Dict):
    history = _load_history()
    history.append(record)
    history = history[-100:]
    try:
        with open(EVAL_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass


def get_drift_trend(metric: str = "faithfulness") -> List[Dict]:
    history = _load_history()
    return [
        {"timestamp": r["timestamp"], "value": r["ragas_scores"].get(metric)}
        for r in history if metric in r["ragas_scores"]
    ]
