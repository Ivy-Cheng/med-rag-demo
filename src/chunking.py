from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    score_vector: float = 0.0
    score_bm25: float = 0.0
    score_rerank: float = 0.0
    debug: dict[str, Any] | None = None


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 90,
    chunk_overlap: int = 20,
) -> list[Chunk]:
    """Word-window chunking for small raw-text demo documents."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    stride = chunk_size - chunk_overlap

    for doc in documents:
        words = doc.text.split()
        if not words:
            continue

        for start in range(0, len(words), stride):
            window = words[start : start + chunk_size]
            if not window:
                continue

            chunk_index = len([c for c in chunks if c.doc_id == doc.doc_id])
            metadata = {
                **doc.metadata,
                "title": doc.title,
                "chunk_index": chunk_index,
                "word_start": start,
                "word_end": start + len(window),
            }
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::c{chunk_index}",
                    doc_id=doc.doc_id,
                    text=" ".join(window),
                    metadata=metadata,
                )
            )

            if start + chunk_size >= len(words):
                break

    return chunks

