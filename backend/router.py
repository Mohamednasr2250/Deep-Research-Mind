"""
router.py — Query Routing + Adaptive RAG
"""

import mlflow


def route_query(query: str, llm) -> str:
    prompt = f"""Classify this user request into exactly ONE category:
- qa              (a specific question about paper content)
- summarize       (wants a summary of one or more papers)
- compare         (wants papers compared against each other)
- literature_review (wants a broader literature review on a topic)
- general         (small talk, unrelated to papers, e.g. greetings)

Request: {query}

Category (one word only):"""

    try:
        category = llm.invoke(prompt).strip().lower()
    except Exception:
        category = "qa"

    valid = ["qa", "summarize", "compare", "literature_review", "general"]
    if category not in valid:
        q = query.lower()
        if "compare" in q or "vs" in q:
            category = "compare"
        elif "summar" in q:
            category = "summarize"
        elif "literature review" in q or "review" in q:
            category = "literature_review"
        elif any(g in q for g in ["hi", "hello", "hey", "thanks"]):
            category = "general"
        else:
            category = "qa"

    with mlflow.start_run(run_name="query_routing"):
        mlflow.log_param("query",    query[:100])
        mlflow.log_param("category", category)

    return category


def needs_retrieval(query: str, llm) -> bool:
    prompt = f"""Does answering this question require searching uploaded research papers?
Answer YES or NO only.

Question: {query}

Answer:"""

    try:
        decision = llm.invoke(prompt).strip().upper()
    except Exception:
        decision = "YES"

    result = "YES" in decision

    with mlflow.start_run(run_name="adaptive_rag_check"):
        mlflow.log_param("query", query[:100])
        mlflow.log_param("needs_retrieval", str(result))

    return result


def handle_general_query(query: str, llm) -> str:
    prompt = f"""You are ResearchMind, an AI assistant for analyzing research papers.
Respond briefly and naturally to this message (it doesn't require searching papers):

{query}

Response:"""
    return llm.invoke(prompt).strip()
