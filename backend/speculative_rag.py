import mlflow


def generate_draft(context: str, query: str, llm) -> str:
    prompt = f"""Context:
{context[:1000]}

Question: {query}
Give a brief, quick draft answer (1-2 sentences):"""
    return llm.invoke(prompt).strip()


def verify_draft(draft: str, context: str, query: str, llm) -> dict:
    prompt = f"""Question: {query}
Context: {context[:1500]}
Draft answer: {draft}

Is this draft answer well-supported by the context and reasonably complete?
Answer YES or NO, with a one-sentence reason.
Format:
Verdict: YES/NO
Reason: ..."""

    response = llm.invoke(prompt).strip()
    verdict = "YES" in response.upper().split("\n")[0]
    reason = response.split("Reason:")[-1].strip() if "Reason:" in response else ""

    return {"accepted": verdict, "reason": reason}


def refine_answer(draft: str, context: str, query: str, reason: str, llm) -> str:
    prompt = f"""The following draft answer was found insufficient: "{reason}"

Context:
{context}

Question: {query}
Draft that needs improvement: {draft}

Provide a complete, well-grounded final answer:"""
    return llm.invoke(prompt).strip()


def speculative_rag_answer(query: str, context: str, llm) -> dict:
    draft = generate_draft(context, query, llm)
    verification = verify_draft(draft, context, query, llm)

    if verification["accepted"]:
        with mlflow.start_run(run_name="speculative_rag"):
            mlflow.log_param("refined", "false")
        return {"answer": draft, "refined": False, "reason": verification["reason"]}

    final_answer = refine_answer(draft, context, query, verification["reason"], llm)

    with mlflow.start_run(run_name="speculative_rag"):
        mlflow.log_param("refined", "true")
        mlflow.log_param("rejection_reason", verification["reason"][:200])

    return {"answer": final_answer, "refined": True, "reason": verification["reason"]}
