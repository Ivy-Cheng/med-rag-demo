from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from .chunking import chunk_documents
from .document_loader import load_documents
from .evaluation import evaluate_generation, self_similarity_eval
from .generator import build_generator
from .hybrid_retriever import HybridRetriever
from .reranker import HeuristicReranker


class MedRAGPipeline:
    """End-to-end mini Med-RAG pipeline for portfolio demonstration."""

    def __init__(
        self,
        data_dir: str | Path,
        chunk_size: int = 90,
        chunk_overlap: int = 20,
        generator_backend: str = "template",
        generator_model: str = "deepseek-r1:7b",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.generator = build_generator(generator_backend, model=generator_model)

        self.documents = load_documents(self.data_dir)
        self.chunks = chunk_documents(
            self.documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.retriever = HybridRetriever(self.chunks)
        self.reranker = HeuristicReranker()

    def run(self, query: str, top_k: int = 4) -> dict[str, Any]:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        candidates = self.retriever.search(query, top_k=top_k * 2, candidate_k=top_k * 3)
        timings["retrieval"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        selected = self.reranker.rerank(query, candidates, top_k=top_k)
        timings["reranking"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        answer = self.generator.generate(query, selected)
        timings["generation"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        generation_eval = evaluate_generation(answer, selected)
        retrieval_eval = self_similarity_eval(self.chunks, self.retriever, sample_k=min(5, len(self.chunks)))
        timings["evaluation"] = time.perf_counter() - t0

        timings["total"] = sum(timings.values())

        return {
            "query": query,
            "answer": answer,
            "sources": [_format_source(hit) for hit in selected],
            "retrieval_debug": [_format_debug(hit) for hit in selected],
            "evaluation": {
                "generation": generation_eval,
                "retrieval_proxy": retrieval_eval,
            },
            "metrics": {
                "num_documents": len(self.documents),
                "num_chunks": len(self.chunks),
                "top_k": top_k,
                "timings_seconds": {k: round(v, 4) for k, v in timings.items()},
                "answer_chars": len(answer),
            },
        }


def _format_source(hit) -> dict[str, Any]:
    metadata = hit.chunk.metadata
    return {
        "chunk_id": hit.chunk.chunk_id,
        "doc_id": hit.chunk.doc_id,
        "title": metadata.get("title"),
        "journal": metadata.get("journal"),
        "year": metadata.get("year"),
        "score": round(hit.score, 4),
    }


def _format_debug(hit) -> dict[str, Any]:
    return {
        "chunk_id": hit.chunk.chunk_id,
        "score": round(hit.score, 4),
        "score_vector": round(hit.score_vector, 4),
        "score_bm25": round(hit.score_bm25, 4),
        "debug": hit.debug or {},
    }

