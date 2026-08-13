from langchain_huggingface import HuggingFaceEndpoint
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from typing import List
import mlflow


def rewrite_query(query: str, llm) -> str:
    prompt = f"""Rewrite this question to be more specific and effective for
searching academic research papers. Make it more detailed and technical.

Original question: {query}

Rewritten search query (one sentence, no explanation):"""

    rewritten = llm.invoke(prompt).strip()

    with mlflow.start_run(run_name="query_rewriting"):
        mlflow.log_param("original_query",  query)
        mlflow.log_param("rewritten_query", rewritten)

    return rewritten


def hyde_query(query: str, llm, vector_store: PineconeVectorStore, k: int = 5) -> List[Document]:
    prompt = f"""Write a short paragraph as if you are answering this research question.
This is for search purposes — write what you think a research paper would say.

Question: {query}

Hypothetical research paper answer (2-3 sentences):"""

    hypothetical_answer = llm.invoke(prompt).strip()

    docs = vector_store.similarity_search(hypothetical_answer, k=k)

    with mlflow.start_run(run_name="hyde_retrieval"):
        mlflow.log_param("original_query",      query)
        mlflow.log_param("hypothetical_answer", hypothetical_answer[:200])
        mlflow.log_metric("docs_retrieved",     len(docs))

    return docs


def multi_query_retrieve(query: str, llm, vector_store: PineconeVectorStore, k: int = 3) -> List[Document]:
    prompt = f"""Generate 3 different versions of this research question.
Each version should use different words but ask the same thing.
Output ONLY the 3 questions, one per line, no numbering.

Original question: {query}

3 variations:"""

    variations_text = llm.invoke(prompt).strip()
    variations      = [q.strip() for q in variations_text.split("\n") if q.strip()][:3]

    all_queries = [query] + variations

    all_docs = []
    seen_contents = set()

    for q in all_queries:
        docs = vector_store.similarity_search(q, k=k)
        for doc in docs:
            content_key = doc.page_content[:100]
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                all_docs.append(doc)

    with mlflow.start_run(run_name="multi_query_retrieval"):
        mlflow.log_metric("query_variations",  len(variations))
        mlflow.log_metric("unique_docs_found", len(all_docs))
        mlflow.log_param("original_query",     query)

    return all_docs


def transform_query(query: str, llm, vector_store: PineconeVectorStore, method: str = "rewrite") -> dict:
    if method == "rewrite":
        rewritten = rewrite_query(query, llm)
        docs      = vector_store.similarity_search(rewritten, k=10)
        return {"transformed_query": rewritten, "docs": docs, "method": "rewrite"}
    elif method == "hyde":
        docs = hyde_query(query, llm, vector_store, k=10)
        return {"transformed_query": query, "docs": docs, "method": "hyde"}
    elif method == "multi_query":
        docs = multi_query_retrieve(query, llm, vector_store, k=3)
        return {"transformed_query": query, "docs": docs, "method": "multi_query"}
    else:
        docs = vector_store.similarity_search(query, k=10)
        return {"transformed_query": query, "docs": docs, "method": "none"}


# ── Multi-hop decomposition ─────────────────────────────────

def decompose_query(query: str, llm) -> List[str]:
    prompt = f"""Break this question into separate simple sub-questions that would
each need to be answered individually before combining into a final answer.
If the question is already simple and doesn't need breaking down, return it as-is.
Output ONLY the sub-questions, one per line, no numbering.

Question: {query}

Sub-questions:"""

    response = llm.invoke(prompt).strip()
    sub_questions = [q.strip() for q in response.split("\n") if q.strip()]

    with mlflow.start_run(run_name="multi_hop_decomposition"):
        mlflow.log_param("original_query", query)
        mlflow.log_metric("num_sub_questions", len(sub_questions))

    return sub_questions if sub_questions else [query]


def multi_hop_answer(query: str, llm, hybrid_retriever, k_per_hop: int = 5) -> dict:
    sub_questions = decompose_query(query, llm)

    if len(sub_questions) == 1:
        docs = hybrid_retriever.retrieve(query, k=k_per_hop)
        context = "\n\n".join(d.page_content for d in docs)
        answer = llm.invoke(f"Context:\n{context}\n\nQuestion: {query}\nAnswer:")
        return {"sub_answers": [], "final_answer": answer, "hops": 1}

    sub_answers = []
    for sub_q in sub_questions:
        docs = hybrid_retriever.retrieve(sub_q, k=k_per_hop)
        context = "\n\n".join(d.page_content for d in docs)
        sub_answer = llm.invoke(f"Context:\n{context}\n\nQuestion: {sub_q}\nAnswer:")
        sub_answers.append({"sub_question": sub_q, "answer": sub_answer})

    combined = "\n".join(f"Q: {sa['sub_question']}\nA: {sa['answer']}" for sa in sub_answers)
    synthesis_prompt = f"""Original question: {query}

The following sub-questions were answered individually:
{combined}

Combine these into one coherent final answer to the original question:"""

    final_answer = llm.invoke(synthesis_prompt).strip()

    with mlflow.start_run(run_name="multi_hop_answer"):
        mlflow.log_metric("hops", len(sub_questions))

    return {"sub_answers": sub_answers, "final_answer": final_answer, "hops": len(sub_questions)}


# ── Agentic RAG loop ─────────────────────────────────────────

def agentic_rag_loop(query: str, llm, hybrid_retriever, max_iterations: int = 3, k: int = 5) -> dict:
    current_query = query
    all_docs = []
    trace = []

    for iteration in range(1, max_iterations + 1):
        docs = hybrid_retriever.retrieve(current_query, k=k)
        all_docs.extend(docs)
        context = "\n\n".join(d.page_content for d in docs)

        sufficiency_prompt = f"""Question: {query}
Retrieved context so far:
{context[:1500]}

Is this context SUFFICIENT to answer the question confidently? Answer YES or NO,
and if NO, state what specific information is still missing.
Format:
Sufficient: YES/NO
Missing: [what's missing, or "none"]"""

        check = llm.invoke(sufficiency_prompt).strip()
        is_sufficient = "YES" in check.upper().split("\n")[0]

        trace.append({
            "iteration": iteration,
            "query_used": current_query,
            "docs_retrieved": len(docs),
            "sufficient": is_sufficient
        })

        if is_sufficient or iteration == max_iterations:
            final_context = "\n\n".join(d.page_content for d in all_docs)
            answer = llm.invoke(f"Context:\n{final_context}\n\nQuestion: {query}\nAnswer:")

            with mlflow.start_run(run_name="agentic_rag_loop"):
                mlflow.log_metric("iterations_used", iteration)
                mlflow.log_metric("total_docs_retrieved", len(all_docs))

            return {"answer": answer, "trace": trace, "iterations_used": iteration}

        missing_line = check.split("Missing:")[-1].strip() if "Missing:" in check else ""
        current_query = f"{query} — specifically regarding: {missing_line}" if missing_line else query

    return {"answer": "Could not gather sufficient context.", "trace": trace, "iterations_used": max_iterations}
