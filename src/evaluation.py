from __future__ import annotations

from collections import Counter
import re
from typing import Any

from .chunking import Chunk, ScoredChunk, split_sentences


_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.:-]+::c\d+)\]")


def extract_citations(answer: str) -> list[str]:
    return _CITATION_RE.findall(answer or "")


def citation_coverage(answer: str, selected: list[ScoredChunk]) -> dict[str, Any]:
    citations = extract_citations(answer)
    allowed = {hit.chunk.chunk_id for hit in selected}
    supported = [c for c in citations if c in allowed]
    unsupported = [c for c in citations if c not in allowed]
    return {
        "citations_found": citations,
        "supported_citations": supported,
        "unsupported_citations": unsupported,
        "coverage": len(supported) / len(citations) if citations else 0.0,
        "passed": bool(citations) and not unsupported,
    }


def structure_check(answer: str) -> dict[str, Any]:
    required = ["Answer", "Evidence", "Limitations"]
    present = {name: (name.lower() in (answer or "").lower()) for name in required}
    return {
        "required_sections": present,
        "passed": all(present.values()),
        "answer_chars": len(answer or ""),
        "answer_sentences": len(split_sentences(answer or "")),
    }


def source_diversity(selected: list[ScoredChunk]) -> dict[str, Any]:
    doc_ids = [hit.chunk.doc_id for hit in selected]
    journals = [str(hit.chunk.metadata.get("journal", "unknown")) for hit in selected]
    years = [str(hit.chunk.metadata.get("year", "unknown")) for hit in selected]
    return {
        "num_selected_chunks": len(selected),
        "unique_docs": len(set(doc_ids)),
        "doc_distribution": dict(Counter(doc_ids)),
        "journal_distribution": dict(Counter(journals)),
        "year_distribution": dict(Counter(years)),
    }


def evaluate_generation(answer: str, selected: list[ScoredChunk]) -> dict[str, Any]:
    return {
        "citation_coverage": citation_coverage(answer, selected),
        "structure": structure_check(answer),
        "source_diversity": source_diversity(selected),
    }


def self_similarity_eval(
    chunks: list[Chunk],
    retriever,
    sample_k: int = 5,
    top_k: int = 3,
) -> dict[str, Any]:
    """Proxy retrieval evaluation when no manually labeled queries exist."""
    cases = []
    for chunk in chunks[:sample_k]:
        query = _first_sentence_or_text(chunk.text)
        hits = retriever.search(query, top_k=top_k)
        hit_ids = [h.chunk.chunk_id for h in hits]
        rank = hit_ids.index(chunk.chunk_id) + 1 if chunk.chunk_id in hit_ids else None
        cases.append(
            {
                "query_from_chunk": chunk.chunk_id,
                "top_ids": hit_ids,
                "self_rank": rank,
                "pass_at_1": rank == 1,
                "pass_at_3": rank is not None and rank <= 3,
                "reciprocal_rank": 1.0 / rank if rank else 0.0,
            }
        )

    n = len(cases) or 1
    return {
        "sample_k": len(cases),
        "top_k": top_k,
        "pass_at_1": sum(c["pass_at_1"] for c in cases) / n,
        "pass_at_3": sum(c["pass_at_3"] for c in cases) / n,
        "mrr": sum(c["reciprocal_rank"] for c in cases) / n,
        "cases": cases,
    }


def _first_sentence_or_text(text: str) -> str:
    sentences = split_sentences(text)
    return sentences[0] if sentences else text[:200]

