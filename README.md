# Medical RAG Demo

This repository is a public demonstration version of a medical-domain retrieval-augmented generation pipeline.

The original project explored biomedical literature question answering using document preprocessing, chunking, dense retrieval, BM25 retrieval, hybrid retrieval, reranking, and source-grounded answer generation.

## Motivation

Large language models can hallucinate when answering domain-specific medical questions. Retrieval-augmented generation helps ground answers in relevant source documents and improves transparency.

## My Contributions

- Implemented document loading and preprocessing.
- Designed chunking strategies for long biomedical documents.
- Built dense embedding retrieval and BM25 keyword retrieval.
- Combined dense and sparse retrievers using hybrid retrieval.
- Added reranking to improve top-k relevance.
- Designed a source-grounded QA pipeline.
- Tested the pipeline using biomedical literature examples.

## Repository Scope

This is a simplified public demo. The original project-specific data, larger document collection, and private experiment files are not included.

## Planned Demo Structure

```text
med-rag-demo/
├── README.md
├── requirements.txt
├── data/
│   └── sample_documents/
├── src/
│   ├── document_loader.py
│   ├── chunking.py
│   ├── bm25_retriever.py
│   ├── vector_retriever.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   └── qa_pipeline.py
├── examples/
│   └── run_demo.py
└── notebooks/
    └── demo_rag_pipeline.ipynb
