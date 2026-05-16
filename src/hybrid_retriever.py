from __future__ import annotations

from .bm25_retriever import BM25Retriever
from .chunking import Chunk, ScoredChunk
from .vector_retriever import VectorRetriever


class HybridRetriever:
    """Hybrid retrieval using reciprocal rank fusion (RRF)."""

    def __init__(self, chunks: list[Chunk], rrf_k: int = 60) -> None:
        self.vector = VectorRetriever(chunks)
        self.bm25 = BM25Retriever(chunks)
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5, candidate_k: int = 8) -> list[ScoredChunk]:
        vector_hits = self.vector.search(query, top_k=candidate_k)
        bm25_hits = self.bm25.search(query, top_k=candidate_k)

        fused: dict[str, ScoredChunk] = {}

        def add_hits(hits: list[ScoredChunk], channel: str) -> None:
            for rank, hit in enumerate(hits, start=1):
                cid = hit.chunk.chunk_id
                if cid not in fused:
                    fused[cid] = ScoredChunk(
                        chunk=hit.chunk,
                        score=0.0,
                        score_vector=0.0,
                        score_bm25=0.0,
                        debug={"channels": []},
                    )
                fused_hit = fused[cid]
                fused_hit.score += 1.0 / (self.rrf_k + rank)
                if channel == "vector":
                    fused_hit.score_vector = hit.score_vector
                if channel == "bm25":
                    fused_hit.score_bm25 = hit.score_bm25
                fused_hit.debug = fused_hit.debug or {"channels": []}
                fused_hit.debug["channels"].append({"name": channel, "rank": rank, "raw_score": hit.score})

        add_hits(vector_hits, "vector")
        add_hits(bm25_hits, "bm25")

        return sorted(fused.values(), key=lambda x: x.score, reverse=True)[:top_k]

