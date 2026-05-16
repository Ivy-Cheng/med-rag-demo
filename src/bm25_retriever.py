from __future__ import annotations

from collections import Counter
import math

from .chunking import Chunk, ScoredChunk, tokenize


class BM25Retriever:
    """Dependency-free BM25 lexical retriever."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_freqs = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.doc_lengths = [sum(tf.values()) for tf in self.term_freqs]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.doc_freq = Counter()
        for tf in self.term_freqs:
            self.doc_freq.update(tf.keys())

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        query_terms = tokenize(query)
        scored: list[ScoredChunk] = []

        for chunk, tf, dl in zip(self.chunks, self.term_freqs, self.doc_lengths):
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                df = self.doc_freq.get(term, 0)
                idf = math.log(1 + (len(self.chunks) - df + 0.5) / (df + 0.5))
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                score += idf * (freq * (self.k1 + 1)) / denom

            if score > 0:
                scored.append(ScoredChunk(chunk=chunk, score=score, score_bm25=score))

        return sorted(scored, key=lambda x: x.score, reverse=True)[:top_k]

