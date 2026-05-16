from __future__ import annotations

from collections import Counter
import math

from .chunking import tokenize


class TfidfEmbedder:
    """Small dependency-free TF-IDF embedder for portfolio demos."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        doc_freq: Counter[str] = Counter()
        for text in texts:
            doc_freq.update(set(tokenize(text)))

        n_docs = max(1, len(texts))
        terms = sorted(doc_freq)
        self.vocab = {term: i for i, term in enumerate(terms)}
        self.idf = {
            term: math.log((1 + n_docs) / (1 + df)) + 1.0
            for term, df in doc_freq.items()
        }
        return self

    def transform_one(self, text: str) -> list[float]:
        if not self.vocab:
            raise RuntimeError("TfidfEmbedder must be fitted before transform")

        counts = Counter(tokenize(text))
        total = sum(counts.values()) or 1
        vector = [0.0] * len(self.vocab)

        for term, count in counts.items():
            idx = self.vocab.get(term)
            if idx is None:
                continue
            tf = count / total
            vector[idx] = tf * self.idf.get(term, 1.0)

        return _l2_normalize(vector)

    def transform(self, texts: list[str]) -> list[list[float]]:
        return [self.transform_one(text) for text in texts]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]

