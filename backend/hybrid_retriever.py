from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.storage import InMemoryStore
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Optional, Dict
from collections import defaultdict
import mlflow


class HybridRetriever:
    def __init__(self, vector_store: PineconeVectorStore, docs: List[Document], merge_strategy: str = "rrf"):
        self.vector_store   = vector_store
        self.docs           = docs
        self.merge_strategy = merge_strategy

        self.bm25 = BM25Retriever.from_documents(docs, k=10)
        self.semantic = vector_store.as_retriever(search_kwargs={"k": 10})

        self.weighted_retriever = EnsembleRetriever(
            retrievers=[self.bm25, self.semantic],
            weights=[0.4, 0.6]
        )

    def _rrf_merge(self, ranked_lists: List[List[Document]], k_constant: int = 60) -> List[Document]:
        scores: Dict[str, float] = defaultdict(float)
        doc_lookup: Dict[str, Document] = {}

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list, start=1):
                key = doc.page_content[:150]
                scores[key] += 1.0 / (k_constant + rank)
                doc_lookup[key] = doc

        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [doc_lookup[k] for k in sorted_keys]

    def retrieve_rrf(self, query: str, k: int = 10) -> List[Document]:
        bm25_docs     = self.bm25.invoke(query)
        semantic_docs = self.semantic.invoke(query)
        merged = self._rrf_merge([bm25_docs, semantic_docs])
        return merged[:k]

    def _dynamic_k(self, query: str, base_k: int = 10) -> int:
        word_count = len(query.split())
        is_comparative = any(w in query.lower() for w in ["compare", "versus", "vs", "difference", "both"])

        if word_count <= 6 and not is_comparative:
            return max(3, base_k - 5)
        elif is_comparative or word_count > 20:
            return base_k + 5
        return base_k

    def retrieve(self, query: str, k: int = 10, use_dynamic_k: bool = True) -> List[Document]:
        effective_k = self._dynamic_k(query, k) if use_dynamic_k else k

        if self.merge_strategy == "rrf":
            docs = self.retrieve_rrf(query, k=effective_k)
        else:
            docs = self.weighted_retriever.invoke(query)[:effective_k]

        with mlflow.start_run(run_name="hybrid_retrieval"):
            mlflow.log_param("merge_strategy", self.merge_strategy)
            mlflow.log_metric("effective_k", effective_k)
            mlflow.log_metric("docs_returned", len(docs))

        return docs

    def retrieve_diverse(self, query: str, k: int = 6, max_per_paper: int = 2) -> List[Document]:
        candidates = self.retrieve(query, k=k * 3, use_dynamic_k=False)

        per_paper_count: Dict[str, int] = defaultdict(int)
        diverse_docs = []

        for doc in candidates:
            paper = doc.metadata.get("paper_title", doc.metadata.get("source", "unknown"))
            if per_paper_count[paper] < max_per_paper:
                diverse_docs.append(doc)
                per_paper_count[paper] += 1
            if len(diverse_docs) >= k:
                break

        with mlflow.start_run(run_name="diverse_retrieval"):
            mlflow.log_metric("unique_papers", len(per_paper_count))
            mlflow.log_metric("docs_returned", len(diverse_docs))

        return diverse_docs

    def retrieve_with_confidence(self, query: str, k: int = 5) -> dict:
        scored = self.vector_store.similarity_search_with_score(query, k=k)
        docs   = [doc for doc, _ in scored]
        scores = [score for _, score in scored]

        avg_confidence = round(1 - (sum(scores) / len(scores)), 3) if scores else 0.0

        return {
            "docs":            docs,
            "confidence":      max(0.0, min(1.0, avg_confidence)),
            "top_score":       round(1 - scores[0], 3) if scores else 0.0
        }

    def retrieve_with_metadata_filter(self, query: str, filter_dict: Optional[Dict] = None, k: int = 10) -> List[Document]:
        if not filter_dict:
            return self.retrieve(query, k)

        filtered_docs = self.vector_store.similarity_search(query, k=k, filter=filter_dict)

        bm25_filtered = [
            doc for doc in self.docs
            if all(doc.metadata.get(key) == val for key, val in filter_dict.items())
        ]

        if bm25_filtered:
            bm25_retriever = BM25Retriever.from_documents(bm25_filtered, k=k)
            bm25_docs      = bm25_retriever.invoke(query)
            merged = self._rrf_merge([filtered_docs, bm25_docs])
            return merged[:k]

        return filtered_docs

    def retrieve_with_scores(self, query: str, k: int = 5) -> list:
        return self.vector_store.similarity_search_with_score(query, k=k)


class ParentRetriever:
    def __init__(self, vector_store: PineconeVectorStore):
        self.vector_store = vector_store
        self.docstore      = InMemoryStore()
        self.child_splitter  = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=self.docstore,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter
        )

    def add_documents(self, docs: List[Document]):
        self.retriever.add_documents(docs)

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        docs = self.retriever.invoke(query)
        with mlflow.start_run(run_name="parent_doc_retrieval"):
            mlflow.log_metric("docs_returned", len(docs))
        return docs[:k]


class CompressionRetriever:
    def __init__(self, base_retriever, llm):
        compressor = LLMChainExtractor.from_llm(llm)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        docs = self.retriever.invoke(query)
        with mlflow.start_run(run_name="compression_retrieval"):
            mlflow.log_metric("compressed_docs", len(docs))
        return docs[:k]
