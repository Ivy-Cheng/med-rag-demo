#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.qa_pipeline import MedRAGPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the portfolio Med-RAG demo.")
    parser.add_argument(
        "--query",
        type=str,
        default="How can type 2 diabetes be prevented?",
        help="Medical question to answer from the sample corpus.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(ROOT / "data" / "sample_documents"),
        help="Directory containing raw .txt documents.",
    )
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=90)
    parser.add_argument("--chunk_overlap", type=int, default=20)
    parser.add_argument(
        "--generator_backend",
        choices=["template", "ollama"],
        default="template",
        help="template is deterministic and requires no external service.",
    )
    parser.add_argument("--generator_model", type=str, default="deepseek-r1:7b")
    parser.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "outputs" / "demo_result.json"),
    )
    args = parser.parse_args()

    pipeline = MedRAGPipeline(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        generator_backend=args.generator_backend,
        generator_model=args.generator_model,
    )
    result = pipeline.run(args.query, top_k=args.top_k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Answer ===\n")
    print(result["answer"])
    print("\n=== Sources ===")
    for source in result["sources"]:
        print(f"- {source['chunk_id']} | {source['title']} | score={source['score']}")

    print("\n=== Evaluation Summary ===")
    gen_eval = result["evaluation"]["generation"]
    ret_eval = result["evaluation"]["retrieval_proxy"]
    print("citation_coverage:", gen_eval["citation_coverage"]["coverage"])
    print("citations_passed:", gen_eval["citation_coverage"]["passed"])
    print("structure_passed:", gen_eval["structure"]["passed"])
    print("retrieval_pass@3:", ret_eval["pass_at_3"])
    print("retrieval_mrr:", ret_eval["mrr"])
    print("saved:", out_path)


if __name__ == "__main__":
    main()

