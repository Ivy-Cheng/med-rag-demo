from __future__ import annotations

from .chunking import ScoredChunk, tokenize


class HeuristicReranker:
    """Small reranker that exposes the scoring trade-offs in a readable way."""

    def __init__(
        self,
        relevance_weight: float = 0.55,
        coverage_weight: float = 0.25,
        title_weight: float = 0.15,
        recency_weight: float = 0.05,
        current_year: int = 2026,
    ) -> None:
        self.relevance_weight = relevance_weight
        self.coverage_weight = coverage_weight
        self.title_weight = title_weight
        self.recency_weight = recency_weight
        self.current_year = current_year

    def rerank(self, query: str, hits: list[ScoredChunk], top_k: int = 5) -> list[ScoredChunk]:
        if not hits:
            return []

        q_terms = set(tokenize(query))
        max_base = max(h.score for h in hits) or 1.0
        reranked: list[ScoredChunk] = []

        for hit in hits:
            text_terms = set(tokenize(hit.chunk.text))
            title_terms = set(tokenize(str(hit.chunk.metadata.get("title", ""))))
            coverage = len(q_terms & text_terms) / len(q_terms) if q_terms else 0.0
            title_coverage = len(q_terms & title_terms) / len(q_terms) if q_terms else 0.0
            relevance = hit.score / max_base
            recency = self._recency_score(hit)
            final = (
                self.relevance_weight * relevance
                + self.coverage_weight * coverage
                + self.title_weight * title_coverage
                + self.recency_weight * recency
            )
            hit.score_rerank = final
            hit.score = final
            hit.debug = hit.debug or {}
            hit.debug["reranker"] = {
                "normalized_retrieval_score": round(relevance, 4),
                "query_term_coverage": round(coverage, 4),
                "title_term_coverage": round(title_coverage, 4),
                "recency_score": round(recency, 4),
            }
            reranked.append(hit)

        return sorted(reranked, key=lambda x: x.score, reverse=True)[:top_k]

    def _recency_score(self, hit: ScoredChunk) -> float:
        year = hit.chunk.metadata.get("year")
        try:
            age = max(0, self.current_year - int(year))
        except Exception:
            return 0.5
        return max(0.0, 1.0 - age / 20.0)

