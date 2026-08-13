import re
import mlflow
from typing import List


def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


def generate_next_sentence(question: str, context: str, so_far: str, llm) -> str:
    prompt = f"""Context: {context[:1000]}

Question: {question}
Answer so far: {so_far}

Continue with exactly the NEXT SENTENCE only (do not repeat what's already written):"""
    return llm.invoke(prompt).strip()


def assess_sentence_confidence(sentence: str, context: str, llm) -> dict:
    prompt = f"""Context: {context[:800]}
Sentence: {sentence}

Is this sentence well-supported by the context, or is it uncertain/possibly
made up? Answer CONFIDENT or UNCERTAIN, with a short reason.
Format:
Status: CONFIDENT/UNCERTAIN
Reason: ..."""

    response = llm.invoke(prompt).strip()
    is_confident = "CONFIDENT" in response.upper().split("\n")[0] and "UNCERTAIN" not in response.upper().split("\n")[0]
    reason = response.split("Reason:")[-1].strip() if "Reason:" in response else ""

    return {"confident": is_confident, "reason": reason}


def flare_generate(question: str, initial_context: str, hybrid_retriever, llm, max_sentences: int = 6) -> dict:
    context = initial_context
    answer_so_far = ""
    trace = []

    for i in range(max_sentences):
        sentence = generate_next_sentence(question, context, answer_so_far, llm)

        if not sentence or sentence.strip() in (".", ""):
            break

        confidence = assess_sentence_confidence(sentence, context, llm)

        if not confidence["confident"]:
            fresh_docs = hybrid_retriever.retrieve(sentence, k=3)
            fresh_context = "\n\n".join(d.page_content for d in fresh_docs)
            context = context + "\n\n" + fresh_context
            sentence = generate_next_sentence(question, context, answer_so_far, llm)

        answer_so_far += " " + sentence
        trace.append({
            "sentence_index": i + 1,
            "sentence": sentence,
            "was_uncertain": not confidence["confident"],
            "reason": confidence["reason"]
        })

        if any(sentence.strip().endswith(p) for p in [".", "!"]) and i >= 2 and "conclu" in sentence.lower():
            break

    with mlflow.start_run(run_name="flare_generation"):
        mlflow.log_metric("sentences_generated", len(trace))
        mlflow.log_metric("re_retrievals_triggered", sum(1 for t in trace if t["was_uncertain"]))

    return {"answer": answer_so_far.strip(), "trace": trace}
