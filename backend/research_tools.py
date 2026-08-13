import mlflow
from langchain_huggingface import HuggingFaceEndpoint
from typing import List


def get_llm(hf_api_key: str) -> HuggingFaceEndpoint:
    return HuggingFaceEndpoint(
        repo_id="google/flan-t5-base",
        huggingfacehub_api_token=hf_api_key,
        max_new_tokens=512,
        temperature=0.1,
    )


def summarize_paper(context: str, llm) -> str:
    prompt = f"""Summarize this research paper in a structured way.
Include: main objective, methodology, key findings, and conclusions.

Paper content:
{context[:2000]}

Structured Summary:"""
    return llm.invoke(prompt)


def compare_papers(contexts: list[str], query: str, llm) -> str:
    combined = "\n\n---PAPER BREAK---\n\n".join(
        [f"Paper {i+1}:\n{ctx[:800]}" for i, ctx in enumerate(contexts)]
    )
    prompt = f"""Compare these research papers on the topic: {query}

Papers:
{combined}

Comparison (similarities, differences, contradictions, consensus):"""
    return llm.invoke(prompt)


def generate_literature_review(contexts: list[str], topic: str, llm) -> str:
    combined = "\n\n".join(
        [f"Source {i+1}: {ctx[:600]}" for i, ctx in enumerate(contexts)]
    )
    prompt = f"""Write a short academic literature review on: {topic}

Based on these sources:
{combined}

Literature Review:"""
    return llm.invoke(prompt)


def extract_key_findings(context: str, llm) -> str:
    prompt = f"""Extract the key findings and contributions from this research paper.
Format as a numbered list.

Paper:
{context[:1500]}

Key Findings:"""
    return llm.invoke(prompt)


def generate_quiz(context: str, llm) -> str:
    prompt = f"""Generate 5 multiple choice questions based on this research paper.
Format: Question, then 4 options (A, B, C, D), then correct answer.

Paper:
{context[:1500]}

Quiz:"""
    return llm.invoke(prompt)


def build_citations(docs: List) -> list[dict]:
    citations = []
    for i, doc in enumerate(docs, start=1):
        citations.append({
            "citation_number": i,
            "paper":   doc.metadata.get("source", doc.metadata.get("paper_title", "Unknown")),
            "page":    doc.metadata.get("page", "N/A"),
            "excerpt": doc.page_content[:150].strip()
        })
    return citations


def answer_with_citations(answer: str, docs: List, llm) -> dict:
    citations = build_citations(docs)

    if not citations:
        return {"answer": answer, "citations": []}

    paper_list = [f"[{c['citation_number']}] {c['paper']}" for c in citations]

    prompt = f"""Add citation numbers like [1], [2], [3] into this answer,
placing each number right after the claim it supports.
Only use numbers from the source list below. Do not invent new numbers.

Answer:
{answer}

Available sources:
{chr(10).join(paper_list)}

Answer with citation numbers inserted:"""

    try:
        cited_answer = llm.invoke(prompt).strip()
    except Exception:
        cited_answer = answer

    with mlflow.start_run(run_name="citation_tracking"):
        mlflow.log_metric("num_citations", len(citations))

    return {
        "answer":    cited_answer,
        "citations": citations
    }


def build_answer_prompt(context: str, query: str, detail_level: str = "medium") -> str:
    length_instruction = {
        "short":    "Answer in 1-2 sentences. Be direct, no elaboration.",
        "medium":   "Answer in a short paragraph (3-5 sentences).",
        "detailed": "Answer in full detail, explaining reasoning and relevant context from the papers."
    }.get(detail_level, "Answer in a short paragraph (3-5 sentences).")

    return f"""You are a research assistant. Answer the question using only the research papers below.
If the answer is not in the papers, say "This information is not in the uploaded papers."

{length_instruction}

Research Papers Context:
{context}

Question: {query}
Answer:"""


def map_reduce_summarize(full_text: str, llm, chunk_size: int = 2000) -> str:
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        prompt = f"""Summarize this section of a research paper in 2-3 sentences.
Focus on any new information: objective, method, results, or conclusions mentioned.

Section {i+1}:
{chunk}

Summary:"""
        try:
            summary = llm.invoke(prompt).strip()
            chunk_summaries.append(summary)
        except Exception:
            continue

    combined = "\n".join(f"- {s}" for s in chunk_summaries)
    reduce_prompt = f"""Below are summaries of consecutive sections of a research paper.
Combine them into ONE structured summary covering: main objective, methodology,
key findings, and conclusions. Remove redundancy.

Section summaries:
{combined}

Final Structured Summary:"""

    final_summary = llm.invoke(reduce_prompt).strip()

    with mlflow.start_run(run_name="map_reduce_summarization"):
        mlflow.log_metric("num_chunks", len(chunks))
        mlflow.log_metric("text_length", len(full_text))

    return final_summary


def hierarchical_summarize(full_text: str, llm, chunk_size: int = 2000, max_levels: int = 2) -> str:
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]

    current_level = []
    for chunk in chunks:
        prompt = f"Summarize in 2-3 sentences:\n{chunk}\n\nSummary:"
        try:
            current_level.append(llm.invoke(prompt).strip())
        except Exception:
            continue

    level = 1
    while len("\n".join(current_level)) > chunk_size and level < max_levels:
        batches = [current_level[i:i + 5] for i in range(0, len(current_level), 5)]
        next_level = []
        for batch in batches:
            combined_text = "\n".join(batch)
            prompt = f"Combine these summaries into one shorter summary:\n{combined_text}\n\nCombined Summary:"
            try:
                next_level.append(llm.invoke(prompt).strip())
            except Exception:
                continue
        current_level = next_level
        level += 1

    final_prompt = f"""Combine these into one final structured summary
(objective, methodology, key findings, conclusions):
{chr(10).join(current_level)}

Final Summary:"""
    final = llm.invoke(final_prompt).strip()

    with mlflow.start_run(run_name="hierarchical_summarization"):
        mlflow.log_metric("levels_used", level)
        mlflow.log_metric("text_length", len(full_text))

    return final
