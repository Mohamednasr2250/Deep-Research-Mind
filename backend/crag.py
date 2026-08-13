from langchain_core.documents import Document
from typing import List
import mlflow


def grade_chunk_relevance(query: str, doc: Document, llm) -> str:
    prompt = f"""Question: {query}
Retrieved passage: {doc.page_content[:500]}

Is this passage relevant and useful for answering the question?
Answer with exactly one word: CORRECT, INCORRECT, or AMBIGUOUS."""

    try:
        grade = llm.invoke(prompt).strip().upper()
    except Exception:
        grade = "AMBIGUOUS"

    if "CORRECT" in grade and "INCORRECT" not in grade:
        return "CORRECT"
    elif "INCORRECT" in grade:
        return "INCORRECT"
    return "AMBIGUOUS"


def grade_all_chunks(query: str, docs: List[Document], llm) -> dict:
    graded = {"CORRECT": [], "INCORRECT": [], "AMBIGUOUS": []}

    for doc in docs:
        grade = grade_chunk_relevance(query, doc, llm)
        graded[grade].append(doc)

    correct_ratio = len(graded["CORRECT"]) / len(docs) if docs else 0.0

    with mlflow.start_run(run_name="crag_grading"):
        mlflow.log_metric("correct_ratio", round(correct_ratio, 3))
        mlflow.log_metric("num_correct", len(graded["CORRECT"]))
        mlflow.log_metric("num_incorrect", len(graded["INCORRECT"]))

    return {**graded, "correct_ratio": correct_ratio}


def reformulate_query(query: str, llm) -> str:
    prompt = f"""This search query did not retrieve good results from the paper database:
"{query}"

Rewrite it with different keywords or phrasing to try to get better results:"""
    return llm.invoke(prompt).strip()


def corrective_rag_retrieve(
    query: str,
    hybrid_retriever,
    llm,
    web_search_fn=None,
    k: int = 8,
    correct_ratio_threshold: float = 0.4,
    max_retries: int = 1
) -> dict:
    docs = hybrid_retriever.retrieve(query, k=k)
    graded = grade_all_chunks(query, docs, llm)
    trace = [{"query": query, "correct_ratio": graded["correct_ratio"]}]

    retries = 0
    current_query = query
    while graded["correct_ratio"] < correct_ratio_threshold and retries < max_retries:
        current_query = reformulate_query(current_query, llm)
        docs = hybrid_retriever.retrieve(current_query, k=k)
        graded = grade_all_chunks(current_query, docs, llm)
        trace.append({"query": current_query, "correct_ratio": graded["correct_ratio"], "retry": retries + 1})
        retries += 1

    fallback_used = False
    final_docs = graded["CORRECT"] + graded["AMBIGUOUS"]

    if graded["correct_ratio"] < correct_ratio_threshold and web_search_fn:
        web_results = web_search_fn(query)
        fallback_used = True
        trace.append({"fallback": "web_search", "results_found": len(web_results)})
        return {
            "docs": final_docs,
            "web_fallback_results": web_results,
            "fallback_used": True,
            "trace": trace
        }

    with mlflow.start_run(run_name="corrective_rag"):
        mlflow.log_metric("retries_used", retries)
        mlflow.log_param("fallback_used", str(fallback_used))

    return {
        "docs": final_docs,
        "web_fallback_results": [],
        "fallback_used": fallback_used,
        "trace": trace
    }
