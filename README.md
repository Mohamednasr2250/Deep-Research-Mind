# ResearchMind — Intelligent Research Paper Analysis Platform

A production-grade multi-paper RAG platform: upload research papers, ask cross-paper questions,
generate summaries and literature reviews, and get answers that are retrieved, reranked, graded,
cited, and continuously evaluated — not just generated.

## Problem

Researchers drown in volume (reading dozens of papers manually), can't trust naive AI summaries
(no grounding, no citations), lose conversational context between questions, and have no way to
know if answer quality degrades over time as more papers are added. ResearchMind addresses all
four with a full retrieval → generation → evaluation → observability pipeline.

## Architecture

```
Upload → Ingest (chunk / table+figure extraction) → Hybrid Retrieval (BM25 + Semantic + RRF)
       → Rerank (cross-encoder + lost-in-the-middle reorder) → Input Guardrail
       → Generate (adaptive / agentic / corrective / speculative) → Output Guardrail
       → Citation Tracking → Hallucination Check → Semantic Cache → Request Trace
```

## Core Capabilities

### Retrieval
Hybrid search (BM25 + semantic, merged via Reciprocal Rank Fusion), cross-encoder reranking with
lost-in-the-middle mitigation, query rewriting / HyDE / multi-query / multi-hop decomposition,
parent document retriever, contextual compression, metadata filtering, dynamic top-k, source
diversity enforcement, retrieval confidence scoring, embedding-model versioning stub.

### Chunking / Ingestion
Fixed, semantic, and contextual chunking; map-reduce and hierarchical summarization for
full-length papers; table/figure-aware ingestion via `unstructured`.

### Generation
Adaptive RAG (skip retrieval when unnecessary), Agentic RAG (iterative retrieve-decide loop),
Corrective RAG / CRAG (chunk grading + reformulation fallback), Speculative RAG (draft-then-verify),
FLARE (sentence-level generation with confidence-triggered re-retrieval), inline citation tracking,
answer length control, conversational memory.

### Evaluation & Trust
RAGAS (faithfulness, relevancy, precision, recall), MRR + NDCG retrieval ranking metrics,
NLI-based hallucination detection, continuous evaluation pipeline (nightly drift tracking +
regression alerts), input/output guardrails (prompt-injection defense, PII redaction, system-leak
detection), full request-level tracing.

### Production
Semantic caching, streaming responses, chunk deduplication, per-user document scoping,
delete/re-index endpoints, MLflow tracking, Prometheus/Grafana monitoring, Docker/Kubernetes,
GitHub Actions CI/CD.

### Advanced / Research-Grade (scoped for portfolio demo, not production scale)
- **GraphRAG** — LLM entity/relationship extraction + Louvain community detection (NetworkX,
  in-memory) + community summarization, enabling "how does X relate to Y" via graph traversal.
- **Active Learning** — automated hard-negative mining from negative feedback; fine-tuning is a
  manually-triggered script (`active_learning.run_finetune_script`), not continuous retraining.
- **RAFT** — automated training-data generation with real distractor chunks; fine-tuning uses a
  small local model (`flan-t5-small` + LoRA) with before/after RAGAS comparison.

## Honest Scope Notes

- Generation quality is capped by `flan-t5-base` (a small hosted model) — the architecture is
  production-grade, the LLM itself is not state-of-the-art.
- Active Learning and RAFT fine-tuning are offline scripts, run manually, not automated
  production training pipelines.
- Table/figure extraction handles structure and captions, not visual understanding of image
  content (no vision model in the loop).
- GraphRAG runs at portfolio scale (NetworkX in-memory); real GraphRAG at scale needs a graph
  database and more robust community detection.
- `fixed_eval_set.json` ships with only 2 example entries — curate a real 20-30 example set before
  relying on the continuous evaluation pipeline's drift numbers.

## Project Structure

```
ResearchMind/
├── backend/                 # FastAPI app — see backend/ for all 21 modules
├── kubernetes/              # deployment, service, HPA manifests
├── monitoring/              # Prometheus scrape config
├── .github/workflows/       # CI/CD + nightly evaluation
├── docker-compose.yml       # backend + MLflow + Prometheus + Grafana
├── .env.example
└── .gitignore
```

## How to Run

### Option A — Docker Compose (recommended, runs everything together)
```bash
cp .env.example .env   # fill in real PINECONE_API_KEY and HF_API_KEY
docker-compose up --build
```
- API docs: http://localhost:8000/docs
- MLflow:   http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana:  http://localhost:3000 (admin/admin)

### Option B — Local, manual
```bash
cd backend
python -m venv venv && source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
cp ../.env.example .env   # fill in real keys

mlflow ui --port 5000                      # terminal 1
uvicorn main:app --reload --port 8000      # terminal 2
```

### Kubernetes (production)
```bash
kubectl create secret generic app-secrets \
  --from-literal=HF_API_KEY=your_key \
  --from-literal=PINECONE_API_KEY=your_key
kubectl apply -f kubernetes/
```

## Future Work (explicitly descoped)
- Research Frontier Detection (identify unsolved problems across a paper set)
- Research Advisor (idea orientation + novelty check + publication roadmap)
- Paper Discovery (auto-find top papers via Semantic Scholar API)
- Cross-lingual RAG, online A/B evaluation
