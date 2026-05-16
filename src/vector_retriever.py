from __future__ import annotations

from .chunking import Chunk, ScoredChunk
from .embeddings import TfidfEmbedder, cosine


class VectorRetriever:
    """Semantic retrieval over local TF-IDF vectors."""

    def __init__(self, chunks: list[Chunk], embedder: TfidfEmbedder | None = None) -> None:
        self.chunks = chunks
        self.embedder = embedder or TfidfEmbedder()
        self.embedder.fit([chunk.text for chunk in chunks])
        self.chunk_vectors = self.embedder.transform([chunk.text for chunk in chunks])

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        q_vec = self.embedder.transform_one(query)
        scored = [
            ScoredChunk(chunk=chunk, score=score, score_vector=score)
            for chunk, vec in zip(self.chunks, self.chunk_vectors)
            if (score := cosine(q_vec, vec)) > 0
        ]
        return sorted(scored, key=lambda x: x.score, reverse=True)[:top_k]

