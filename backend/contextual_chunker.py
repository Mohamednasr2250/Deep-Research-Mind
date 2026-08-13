from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import mlflow


def fixed_chunk(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def semantic_chunk(documents: List[Document], embeddings: HuggingFaceEmbeddings) -> List[Document]:
    try:
        from langchain_experimental.text_splitter import SemanticChunker

        splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
        )
        chunks = splitter.split_documents(documents)

        with mlflow.start_run(run_name="semantic_chunking"):
            mlflow.log_metric("input_docs", len(documents))
            mlflow.log_metric("output_chunks", len(chunks))
            mlflow.log_param("method", "semantic_percentile")

        return chunks
    except ImportError:
        print("langchain_experimental not installed — falling back to fixed chunking")
        return fixed_chunk(documents)


def add_contextual_context(chunks: List[Document], full_document_text: str, llm) -> List[Document]:
    contextualized_chunks = []
    doc_preview = full_document_text[:3000]

    for i, chunk in enumerate(chunks):
        prompt = f"""Here is a document:
{doc_preview}

Here is a specific chunk from that document:
{chunk.page_content[:500]}

In 1-2 sentences, explain where this chunk fits in the overall document
and what topic it covers. Be concise.

Context:"""
        try:
            context_summary = llm.invoke(prompt).strip()
        except Exception:
            context_summary = f"Chunk {i+1} from document."

        contextualized_content = f"{context_summary}\n\n{chunk.page_content}"

        new_doc = Document(
            page_content=contextualized_content,
            metadata={
                **chunk.metadata,
                "has_context": True,
                "chunk_index": i,
                "original_chunk": chunk.page_content[:100]
            }
        )
        contextualized_chunks.append(new_doc)

    with mlflow.start_run(run_name="contextual_retrieval"):
        mlflow.log_metric("chunks_contextualized", len(contextualized_chunks))
        mlflow.log_param("method", "llm_context_prepend")

    return contextualized_chunks


def chunk_documents(
    documents: List[Document],
    method: str = "fixed",
    embeddings: HuggingFaceEmbeddings = None,
    llm=None,
    full_text: str = ""
) -> List[Document]:
    if method == "semantic" and embeddings:
        return semantic_chunk(documents, embeddings)
    elif method == "contextual" and llm:
        base_chunks = fixed_chunk(documents)
        return add_contextual_context(base_chunks, full_text, llm)
    else:
        return fixed_chunk(documents)


# ── Table/Figure-Aware Ingestion ───────────────────────────

def extract_structured_elements(file_path: str) -> dict:
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError:
        print("`unstructured` not installed — falling back to text-only ingestion")
        return {"tables": [], "figures": [], "narrative": [], "available": False}

    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True
    )

    tables, figures, narrative = [], [], []

    for el in elements:
        el_type = type(el).__name__

        if el_type == "Table":
            tables.append({
                "text": str(el),
                "html": getattr(el.metadata, "text_as_html", None),
                "page": getattr(el.metadata, "page_number", None)
            })
        elif el_type in ("Image", "Figure"):
            figures.append({
                "text": str(el),
                "page": getattr(el.metadata, "page_number", None)
            })
        else:
            narrative.append({
                "text": str(el),
                "page": getattr(el.metadata, "page_number", None)
            })

    return {"tables": tables, "figures": figures, "narrative": narrative, "available": True}


def tables_to_documents(tables: List[dict], source_filename: str, llm=None) -> List[Document]:
    docs = []
    for i, table in enumerate(tables):
        content = table["html"] or table["text"]

        description = ""
        if llm:
            prompt = f"""Describe in one sentence what this table shows (its topic and columns),
so it can be found via a natural-language search query.

Table content:
{table['text'][:800]}

One-sentence description:"""
            try:
                description = llm.invoke(prompt).strip()
            except Exception:
                description = ""

        page_content = f"{description}\n\n{content}" if description else content

        docs.append(Document(
            page_content=page_content,
            metadata={
                "source": source_filename,
                "content_type": "table",
                "page": table.get("page"),
                "table_index": i
            }
        ))

    with mlflow.start_run(run_name="table_extraction"):
        mlflow.log_metric("tables_extracted", len(docs))
        mlflow.log_param("source", source_filename)

    return docs


def figures_to_documents(figures: List[dict], source_filename: str) -> List[Document]:
    docs = []
    for i, fig in enumerate(figures):
        if not fig["text"].strip():
            continue

        docs.append(Document(
            page_content=f"[Figure caption]: {fig['text']}",
            metadata={
                "source": source_filename,
                "content_type": "figure_caption",
                "page": fig.get("page"),
                "figure_index": i
            }
        ))

    with mlflow.start_run(run_name="figure_extraction"):
        mlflow.log_metric("figures_extracted", len(docs))
        mlflow.log_param("source", source_filename)

    return docs


def ingest_with_structure(file_path: str, source_filename: str, llm=None) -> dict:
    elements = extract_structured_elements(file_path)

    if not elements["available"]:
        return {"narrative_text": None, "table_docs": [], "figure_docs": [], "structured": False}

    narrative_text = "\n\n".join(n["text"] for n in elements["narrative"])
    table_docs  = tables_to_documents(elements["tables"], source_filename, llm=llm)
    figure_docs = figures_to_documents(elements["figures"], source_filename)

    return {
        "narrative_text": narrative_text,
        "table_docs": table_docs,
        "figure_docs": figure_docs,
        "structured": True
    }
