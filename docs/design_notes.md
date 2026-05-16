# Design Notes

## Why This Demo Exists

This folder is a compact version of a fuller Med-RAG project. 

The system intentionally uses a tiny corpus and dependency-free defaults. That makes the demo runnable in a clean Python environment while still showing the same architectural decisions used in larger RAG systems.

## Pipeline Overview

1. Document parsing
   - Raw `.txt` files are loaded from `data/sample_documents`.
   - Header metadata such as title, year, and journal is parsed into document metadata.

2. Chunking
   - Documents are split by word windows with overlap.
   - Each chunk receives a stable ID: `doc_id::cN`.
   - Overlap reduces the chance that useful context is split across two chunks.

3. Embedding and indexing
   - The default embedder is local TF-IDF.
   - It is not meant to be a production embedding model; it is a transparent substitute that keeps the demo runnable.

4. Retrieval
   - BM25 handles exact terms such as diabetes, inhaled corticosteroids, hypertension, or sodium.
   - Vector retrieval handles softer lexical variation.
   - Hybrid retrieval uses reciprocal rank fusion to combine both channels without score calibration.

5. Reranking
   - The reranker combines normalized retrieval score, query-term coverage, and recency.
   - This mirrors a production reranking stage where a cross-encoder or task-specific scoring model would be used.

6. Generation
   - The default generator is grounded and extractive.
   - It writes only from selected chunks and cites chunk IDs.
   - If evidence is missing, it says the evidence is insufficient.
   - Optional Ollama support is included for local LLM generation.

7. Evaluation
   - The demo includes evaluation even without labeled gold answers.
   - It checks retrieval self-similarity, citation validity, answer structure, source diversity, and runtime.

## Hallucination Mitigation

The demo uses multiple simple controls:

- The generator receives only reranked evidence chunks.
- Output citations must match retrieved `chunk_id` values.
- The evaluation layer flags unsupported citations.
- The answer template includes limitations and safety notes.
- If no evidence is retrieved, the generator refuses to answer.

These controls are intentionally visible because they are often more important in applied RAG work than the choice of LLM alone.

## Evaluation Strategy for Raw Text Only

Since it lacks gold query-answer pairs, the demo uses proxy evaluation:

- Self-similarity retrieval: use a chunk sentence as a query and check whether the original chunk returns in top-k.
- MRR: reward systems that rank the source chunk higher.
- Citation coverage: answer citations must be selected evidence chunks.
- Structure checks: answer should contain Answer, Evidence, and Limitations/Safety.
- Source diversity: inspect how many source documents contribute to an answer.
- Latency: track retrieval, reranking, generation, and evaluation timings.

## What Would Change in Production

- Use a real vector database such as Chroma.
- Use biomedical embeddings or BGE-style sentence embeddings.
- Use a cross-encoder reranker.
- Add document-level metadata filters such as date and journal.
- Add human-labeled query-document pairs for Recall@k, MRR, and NDCG.
- Add LLM-as-judge, RAGAS, or TruLens for faithfulness and answer relevance.

