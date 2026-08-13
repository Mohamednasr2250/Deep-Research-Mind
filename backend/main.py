"""
main.py — ResearchMind FastAPI Entry Point (fully consolidated)

This assembles every endpoint added across all build phases into one
running application:
  Phase 1        -> citations, chat memory, MRR/NDCG eval endpoint
  Phase 2        -> dedup, per-user scoping, guardrails, semantic cache,
                     streaming, delete/reindex, query routing, adaptive RAG
  Chunking phase -> table/figure-aware ingestion in /upload
  Generation     -> /ask/advanced (multi-hop, agentic, speculative, CRAG)
  Observability  -> full request tracing + output guardrails wired into /ask
  Advanced       -> GraphRAG, active-learning stats, continuous eval, FLARE
"""

from dotenv import load_dotenv
load_dotenv()

import os
import time
import hashlib
import tempfile
import mlflow

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from prometheus_client import Counter, Histogram, generate_latest

from hybrid_retriever import HybridRetriever
from reranker import rerank_and_reorder
from hallucination import detect_hallucination
from feedback import log_feedback, get_feedback_stats
from research_tools import (
    get_llm, summarize_paper, compare_papers, generate_literature_review,
    extract_key_findings, generate_quiz, answer_with_citations,
    build_answer_prompt, map_reduce_summarize, hierarchical_summarize
)
from chat_memory import chat_memory_manager
from router import route_query, needs_retrieval, handle_general_query
from guardrails import sanitize_context, build_safe_prompt_prefix, validate_output
from semantic_cache import SemanticCache
from evaluator import evaluate_rag, evaluate_retrieval, evaluate_retrieval_batch
from contextual_chunker import ingest_with_structure
from query_transformer import multi_hop_answer, agentic_rag_loop
from speculative_rag import speculative_rag_answer
from crag import corrective_rag_retrieve
from tracing import RequestTrace, get_trace, get_recent_traces
from flare import flare_generate
from active_learning import mine_hard_negative, get_mining_stats, prepare_training_pairs
from continuous_eval import run_scheduled_evaluation, get_drift_trend
from graph_rag import (
    extract_from_documents, build_graph, detect_communities,
    build_community_summaries, answer_relationship_query
)


# ── App ────────────────────────────────────────────────────
app = FastAPI(title="ResearchMind — Intelligent Research Paper Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus ─────────────────────────────────────────────
REQUEST_COUNT   = Counter("api_requests_total",            "Total requests",  ["endpoint"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Latency",         ["endpoint"])
UPLOAD_COUNT    = Counter("papers_uploaded_total",         "Papers uploaded")
QUESTION_COUNT  = Counter("questions_asked_total",         "Questions asked")
LLM_LATENCY     = Histogram("llm_response_latency_seconds","LLM latency")
HALLUCINATION   = Counter("hallucinations_detected_total", "Hallucinations")
FEEDBACK_POS    = Counter("feedback_positive_total",       "Positive feedback")
FEEDBACK_NEG    = Counter("feedback_negative_total",       "Negative feedback")

# ── MLflow ─────────────────────────────────────────────────
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("researchmind")

# ── Pinecone ───────────────────────────────────────────────
pc             = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "researchmind")

if PINECONE_INDEX not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# ── Embeddings + Vector Store ──────────────────────────────
embeddings   = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index_name=PINECONE_INDEX, embedding=embeddings)

# ── LLM ─────────────────────────────────────────────────────
llm = get_llm(os.environ.get("HF_API_KEY"))

# ── Global state ────────────────────────────────────────────
all_docs = []
hybrid_retriever = None
seen_chunk_hashes = set()
semantic_cache = SemanticCache(embeddings_model=embeddings)
paper_graph = None
recent_docs_by_query = {}   # short-lived cache: query -> docs used, for active-learning mining


def _hash_chunk(text: str) -> str:
    return hashlib.md5(text.strip().encode()).hexdigest()


# ── Schemas ──────────────────────────────────────────────────

class Question(BaseModel):
    query: str
    session_id: Optional[str] = None
    detail_level: Optional[str] = "medium"
    user_id: Optional[str] = "default"
    stream: Optional[bool] = False


class FeedbackRequest(BaseModel):
    query:   str
    answer:  str
    rating:  bool
    comment: Optional[str] = ""


class RetrievalEvalRequest(BaseModel):
    retrieved_docs: list[str]
    relevant_docs:  list[str]
    k: int = 5


class CompareRequest(BaseModel):
    topic: str


class AdvancedAskRequest(BaseModel):
    query: str
    mode: str = "standard"   # standard / multi_hop / agentic / speculative / crag
    session_id: Optional[str] = None


# ── Basic endpoints ───────────────────────────────────────────

@app.get("/")
def home():
    REQUEST_COUNT.labels(endpoint="/").inc()
    return {
        "status": "ResearchMind is running 🚀",
        "features": [
            "Multi-paper upload (with table/figure extraction)",
            "Hybrid search (BM25 + Semantic + RRF)",
            "Reranking + lost-in-the-middle mitigation",
            "Query transformation (rewrite/HyDE/multi-query/multi-hop)",
            "Adaptive / Agentic / Corrective / Speculative RAG",
            "Citation tracking, conversational memory",
            "RAGAS + MRR/NDCG evaluation, continuous drift monitoring",
            "Input/output guardrails, full request tracing",
            "Semantic caching, GraphRAG, active learning"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "services": {"pinecone": "connected", "mlflow": "connected", "llm": "loaded"}
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return PlainTextResponse(generate_latest())


# ── Upload (dedup + per-user scoping + table/figure extraction) ──

@app.post("/upload")
async def upload_paper(file: UploadFile = File(...), user_id: str = "default", extract_tables: bool = True):
    global all_docs, hybrid_retriever

    start = time.time()
    REQUEST_COUNT.labels(endpoint="/upload").inc()

    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT supported")

    suffix = ".pdf" if file.filename.endswith(".pdf") else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        new_docs = []
        skipped  = 0

        if suffix == ".pdf" and extract_tables:
            structured = ingest_with_structure(tmp_path, file.filename, llm=llm)

            if structured["structured"] and structured["narrative_text"]:
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                narrative_docs = splitter.create_documents(
                    [structured["narrative_text"]],
                    metadatas=[{"source": file.filename, "content_type": "narrative"}]
                )
                candidate_docs = narrative_docs + structured["table_docs"] + structured["figure_docs"]
            else:
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                candidate_docs = splitter.split_documents(documents)
        else:
            loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            candidate_docs = splitter.split_documents(documents)

        for doc in candidate_docs:
            doc.metadata.setdefault("user_id", user_id)
            doc.metadata.setdefault("paper_title", file.filename)
            doc.metadata.setdefault("embedding_model", "all-MiniLM-L6-v2")

            chunk_hash = _hash_chunk(doc.page_content)
            if chunk_hash in seen_chunk_hashes:
                skipped += 1
                continue
            seen_chunk_hashes.add(chunk_hash)
            new_docs.append(doc)

        if new_docs:
            vector_store.add_documents(new_docs)
            all_docs.extend(new_docs)
            hybrid_retriever = HybridRetriever(vector_store, all_docs)

        duration = time.time() - start
        table_count  = sum(1 for d in new_docs if d.metadata.get("content_type") == "table")
        figure_count = sum(1 for d in new_docs if d.metadata.get("content_type") == "figure_caption")

        with mlflow.start_run(run_name=f"upload_{file.filename}"):
            mlflow.log_param("filename", file.filename)
            mlflow.log_param("user_id", user_id)
            mlflow.log_metric("num_chunks_added", len(new_docs))
            mlflow.log_metric("num_chunks_skipped_dupe", skipped)
            mlflow.log_metric("tables_found", table_count)
            mlflow.log_metric("figures_found", figure_count)
            mlflow.log_metric("ingest_time", round(duration, 3))

        UPLOAD_COUNT.inc()
        REQUEST_LATENCY.labels(endpoint="/upload").observe(duration)

        return {
            "message": f"✅ Uploaded '{file.filename}'",
            "num_chunks_added": len(new_docs),
            "num_chunks_skipped": skipped,
            "tables_extracted": table_count,
            "figures_extracted": figure_count,
            "total_papers": len(set(d.metadata.get("paper_title", "") for d in all_docs)),
            "time_taken": f"{duration:.2f}s"
        }
    finally:
        os.unlink(tmp_path)


@app.delete("/papers/{filename}")
def delete_paper(filename: str):
    global all_docs, hybrid_retriever

    before = len(all_docs)
    all_docs = [d for d in all_docs if d.metadata.get("paper_title") != filename]
    removed  = before - len(all_docs)

    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No chunks found for '{filename}'")

    hybrid_retriever = HybridRetriever(vector_store, all_docs) if all_docs else None

    try:
        pc.Index(PINECONE_INDEX).delete(filter={"paper_title": filename})
    except Exception as e:
        print(f"Pinecone delete warning: {e}")

    with mlflow.start_run(run_name=f"delete_{filename}"):
        mlflow.log_param("filename", filename)
        mlflow.log_metric("chunks_removed", removed)

    return {"message": f"✅ Removed '{filename}'", "chunks_removed": removed}


# ── Main /ask (routing + cache + adaptive + retrieval + rerank +
#    guardrails + citations + output-guardrail + hallucination + trace) ──

@app.post("/ask")
def ask_question(question: Question):
    global hybrid_retriever

    trace = RequestTrace(question.query)
    REQUEST_COUNT.labels(endpoint="/ask").inc()
    QUESTION_COUNT.inc()

    session_id = question.session_id or "default"
    filters    = {"user_id": question.user_id}

    trace.start_step("query_routing")
    category = route_query(question.query, llm)
    trace.end_step({"category": category})

    if category == "general":
        trace.start_step("general_response")
        answer = handle_general_query(question.query, llm)
        trace.end_step()
        record = trace.finish(final_answer=answer)
        return {"query": question.query, "category": category, "answer": answer, "trace_id": record["trace_id"]}

    if not hybrid_retriever:
        raise HTTPException(status_code=400, detail="No papers uploaded yet.")

    trace.start_step("semantic_cache_check")
    cached = semantic_cache.get(question.query, filters=filters)
    trace.end_step({"cache_hit": cached is not None})

    if cached:
        record = trace.finish(final_answer=cached["answer"])
        return {**cached, "category": category, "trace_id": record["trace_id"]}

    trace.start_step("adaptive_rag_check")
    retrieve_needed = needs_retrieval(question.query, llm)
    trace.end_step({"needs_retrieval": retrieve_needed})

    if not retrieve_needed:
        answer = llm.invoke(question.query)
        record = trace.finish(final_answer=answer)
        return {"query": question.query, "category": category, "answer": answer,
                "retrieval_used": False, "trace_id": record["trace_id"]}

    trace.start_step("contextual_query_resolution")
    contextual_query = chat_memory_manager.build_contextual_query(session_id, question.query)
    trace.end_step()

    trace.start_step("hybrid_retrieval")
    docs = hybrid_retriever.retrieve(contextual_query, k=10)
    trace.end_step({"docs_retrieved": len(docs)})

    if not docs:
        raise HTTPException(status_code=404, detail="No relevant content found.")

    trace.start_step("reranking")
    docs = rerank_and_reorder(question.query, docs, top_k=5)
    trace.end_step({"docs_after_rerank": len(docs)})

    trace.start_step("input_guardrail")
    docs = sanitize_context(docs)
    trace.end_step()

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt  = build_safe_prompt_prefix() + build_answer_prompt(context, question.query, question.detail_level)

    if question.stream:
        def token_stream():
            for chunk in llm.stream(prompt):
                yield chunk
        return StreamingResponse(token_stream(), media_type="text/plain")

    trace.start_step("generation")
    answer = llm.invoke(prompt)
    trace.end_step()

    trace.start_step("citation_tracking")
    result = answer_with_citations(answer, docs, llm)
    trace.end_step({"num_citations": len(result["citations"])})

    trace.start_step("output_guardrail")
    validated = validate_output(result["answer"])
    trace.end_step(validated["flags"])

    trace.start_step("hallucination_check")
    hallucination = detect_hallucination(validated["safe_answer"], context)
    trace.end_step({"is_hallucination": hallucination["is_hallucination"]})

    if hallucination["is_hallucination"]:
        HALLUCINATION.inc()

    chat_memory_manager.add_exchange(session_id, question.query, validated["safe_answer"])
    semantic_cache.set(question.query, validated["safe_answer"], result["citations"], filters=filters)
    recent_docs_by_query[question.query] = docs   # for active-learning mining on feedback

    record = trace.finish(final_answer=validated["safe_answer"], flags=validated["flags"])

    return {
        "query": question.query,
        "category": category,
        "session_id": session_id,
        "answer": validated["safe_answer"],
        "citations": result["citations"],
        "hallucination": hallucination,
        "guardrail_flags": validated["flags"],
        "retrieval_used": True,
        "trace_id": record["trace_id"]
    }


# ── Advanced generation strategies ───────────────────────────

@app.post("/ask/advanced")
def ask_advanced(request: AdvancedAskRequest):
    global hybrid_retriever
    if not hybrid_retriever:
        raise HTTPException(status_code=400, detail="No papers uploaded yet.")

    if request.mode == "multi_hop":
        result = multi_hop_answer(request.query, llm, hybrid_retriever)
        return {"mode": "multi_hop", **result}

    elif request.mode == "agentic":
        result = agentic_rag_loop(request.query, llm, hybrid_retriever)
        return {"mode": "agentic", **result}

    elif request.mode == "speculative":
        docs = hybrid_retriever.retrieve(request.query, k=5)
        context = "\n\n".join(d.page_content for d in docs)
        result = speculative_rag_answer(request.query, context, llm)
        return {"mode": "speculative", **result}

    elif request.mode == "crag":
        result = corrective_rag_retrieve(request.query, hybrid_retriever, llm)
        if result["docs"]:
            context = "\n\n".join(d.page_content for d in result["docs"])
            answer = llm.invoke(f"Context:\n{context}\n\nQuestion: {request.query}\nAnswer:")
        else:
            answer = "Could not find sufficiently relevant information, even after query reformulation."
        return {"mode": "crag", "answer": answer, "trace": result["trace"], "fallback_used": result["fallback_used"]}

    else:
        docs = hybrid_retriever.retrieve(request.query, k=5)
        context = "\n\n".join(d.page_content for d in docs)
        answer = llm.invoke(f"Context:\n{context}\n\nQuestion: {request.query}\nAnswer:")
        return {"mode": "standard", "answer": answer}


@app.post("/ask/flare")
def ask_with_flare(question: Question):
    if not hybrid_retriever:
        raise HTTPException(status_code=400, detail="No papers uploaded yet.")
    initial_docs = hybrid_retriever.retrieve(question.query, k=5)
    initial_context = "\n\n".join(d.page_content for d in initial_docs)
    result = flare_generate(question.query, initial_context, hybrid_retriever, llm)
    return {"query": question.query, **result}


# ── Summarize / compare / literature review ──────────────────

@app.post("/summarize")
async def summarize(file: UploadFile = File(...), method: str = "map_reduce"):
    REQUEST_COUNT.labels(endpoint="/summarize").inc()

    suffix = ".pdf" if file.filename.endswith(".pdf") else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        loader    = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path)
        documents = loader.load()
        full_text = " ".join([doc.page_content for doc in documents])

        if method == "hierarchical":
            summary = hierarchical_summarize(full_text, llm)
        elif method == "map_reduce":
            summary = map_reduce_summarize(full_text, llm)
        else:
            summary = summarize_paper(full_text, llm)

        findings = extract_key_findings(full_text, llm)
        quiz     = generate_quiz(full_text, llm)

        with mlflow.start_run(run_name=f"summarize_{file.filename}"):
            mlflow.log_param("filename", file.filename)
            mlflow.log_param("method", method)
            mlflow.log_metric("text_length", len(full_text))

        return {"filename": file.filename, "summary": summary, "findings": findings, "quiz": quiz}
    finally:
        os.unlink(tmp_path)


@app.post("/compare")
def compare(request: CompareRequest):
    global hybrid_retriever
    REQUEST_COUNT.labels(endpoint="/compare").inc()

    if not hybrid_retriever:
        raise HTTPException(status_code=400, detail="Upload at least 2 papers first.")

    docs     = hybrid_retriever.retrieve_diverse(request.topic, k=6, max_per_paper=2)
    contexts = [doc.page_content for doc in docs]
    comparison = compare_papers(contexts, request.topic, llm)

    with mlflow.start_run(run_name="paper_comparison"):
        mlflow.log_param("topic", request.topic)
        mlflow.log_metric("papers_used", len(contexts))

    return {"topic": request.topic, "comparison": comparison, "papers_used": len(contexts)}


@app.post("/literature-review")
def literature_review(request: CompareRequest):
    global hybrid_retriever
    REQUEST_COUNT.labels(endpoint="/literature-review").inc()

    if not hybrid_retriever:
        raise HTTPException(status_code=400, detail="Upload papers first.")

    docs     = hybrid_retriever.retrieve_diverse(request.topic, k=6, max_per_paper=2)
    contexts = [doc.page_content for doc in docs]
    review   = generate_literature_review(contexts, request.topic, llm)

    with mlflow.start_run(run_name="literature_review"):
        mlflow.log_param("topic", request.topic)
        mlflow.log_metric("sources_used", len(contexts))

    return {"topic": request.topic, "review": review, "sources": len(contexts)}


# ── Feedback ───────────────────────────────────────────────

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    REQUEST_COUNT.labels(endpoint="/feedback").inc()

    entry = log_feedback(
        query=feedback.query, answer=feedback.answer,
        rating=feedback.rating, comment=feedback.comment
    )

    if feedback.rating:
        FEEDBACK_POS.inc()
    else:
        FEEDBACK_NEG.inc()
        docs_used = recent_docs_by_query.get(feedback.query, [])
        if docs_used:
            mine_hard_negative(feedback.query, feedback.answer, docs_used, feedback.rating)

    return {"message": "✅ Feedback recorded", "entry": entry}


@app.get("/feedback/stats")
def feedback_stats():
    REQUEST_COUNT.labels(endpoint="/feedback/stats").inc()
    return get_feedback_stats()


# ── Evaluation ────────────────────────────────────────────

@app.post("/evaluate")
def evaluate_rag_quality(questions: list[str], answers: list[str], contexts: list[list[str]], ground_truths: list[str]):
    REQUEST_COUNT.labels(endpoint="/evaluate").inc()
    scores = evaluate_rag(questions, answers, contexts, ground_truths)
    return {
        "ragas_scores": scores,
        "interpretation": {
            "faithfulness": "How grounded is the answer in the context",
            "answer_relevancy": "How relevant is the answer to the question",
            "context_precision": "How useful are the retrieved chunks",
            "context_recall": "Did we retrieve all needed information"
        }
    }


@app.post("/evaluate/retrieval")
def evaluate_retrieval_endpoint(request: RetrievalEvalRequest):
    REQUEST_COUNT.labels(endpoint="/evaluate/retrieval").inc()
    scores = evaluate_retrieval(request.retrieved_docs, request.relevant_docs, k=request.k)
    return {
        "scores": scores,
        "interpretation": {
            "mrr": "How quickly the first relevant document was found (rank-based)",
            "ndcg": "How good the entire ranked list is, not just the first hit"
        }
    }


@app.post("/eval/run")
def trigger_continuous_eval():
    if not hybrid_retriever:
        raise HTTPException(status_code=400, detail="No papers uploaded yet.")
    return run_scheduled_evaluation(hybrid_retriever, llm)


@app.get("/eval/drift/{metric}")
def eval_drift(metric: str = "faithfulness"):
    return {"metric": metric, "trend": get_drift_trend(metric)}


# ── Sessions / cache / traces ─────────────────────────────

@app.get("/session/{session_id}")
def get_session_history(session_id: str):
    return {"session_id": session_id, "history": chat_memory_manager.get_history(session_id)}


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    chat_memory_manager.clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}


@app.get("/cache/stats")
def cache_stats():
    return semantic_cache.stats()


@app.delete("/cache")
def clear_cache():
    semantic_cache.clear()
    return {"message": "Semantic cache cleared"}


@app.get("/trace/{trace_id}")
def view_trace(trace_id: str):
    trace = get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


@app.get("/traces")
def list_traces(limit: int = 20):
    return {"traces": get_recent_traces(limit)}


# ── Active learning ────────────────────────────────────────

@app.get("/active-learning/stats")
def active_learning_stats():
    return get_mining_stats()


@app.get("/active-learning/training-data")
def get_training_data():
    triples = prepare_training_pairs()
    return {"num_triples": len(triples), "sample": triples[:3]}


# ── GraphRAG ────────────────────────────────────────────────

@app.post("/graph/build")
def build_paper_graph():
    global paper_graph, all_docs
    if not all_docs:
        raise HTTPException(status_code=400, detail="No papers uploaded yet.")

    triples = extract_from_documents(all_docs, llm)
    paper_graph = build_graph(triples)
    communities = detect_communities(paper_graph)
    summaries = build_community_summaries(paper_graph, communities, llm)

    return {
        "num_entities": paper_graph.number_of_nodes(),
        "num_relationships": paper_graph.number_of_edges(),
        "num_communities": len(summaries),
        "community_summaries": {k: v["summary"] for k, v in summaries.items()}
    }


@app.get("/graph/relationship")
def graph_relationship_query(query: str):
    global paper_graph
    if paper_graph is None:
        raise HTTPException(status_code=400, detail="Graph not built yet. Call /graph/build first.")
    answer = answer_relationship_query(query, paper_graph, llm)
    return {"query": query, "answer": answer}
