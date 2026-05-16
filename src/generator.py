from __future__ import annotations

import json
import urllib.request

from .chunking import ScoredChunk, split_sentences, tokenize


class GroundedAnswerGenerator:
    """Default generator that only writes from retrieved evidence."""

    def generate(self, query: str, evidence: list[ScoredChunk]) -> str:
        if not evidence:
            return (
                "## Answer\n"
                "Insufficient evidence was retrieved to answer the question safely.\n\n"
                "## Evidence\n"
                "- No supporting chunks were selected.\n\n"
                "## Limitations and Safety\n"
                "This demo only answers from retrieved text and should not be used as medical advice."
            )

        evidence = _keep_strong_evidence(evidence)
        q_terms = set(tokenize(query))
        bullets: list[str] = []
        for hit in evidence[:3]:
            sentence = _best_evidence_sentence(hit.chunk.text, q_terms)
            bullets.append(
                f"- {sentence} [{hit.chunk.chunk_id}]"
            )

        short = _make_short_answer(query, bullets)
        return (
            "## Answer\n"
            f"{short}\n\n"
            "## Evidence\n"
            + "\n".join(bullets)
            + "\n\n"
            "## Limitations and Safety\n"
            "The answer is grounded only in the retrieved sample documents. It is a technical RAG demo, not medical advice."
        )


class OllamaGenerator:
    """Optional local LLM generator for users who have Ollama running."""

    def __init__(self, model: str = "deepseek-r1:7b", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, query: str, evidence: list[ScoredChunk]) -> str:
        context = "\n\n".join(
            f"[{hit.chunk.chunk_id}] {hit.chunk.text}" for hit in evidence
        )
        prompt = f"""You are a careful medical RAG assistant.
Only answer using the provided context. Cite chunk IDs in square brackets.
If the context is insufficient, say so.

Question:
{query}

Context:
{context}

Return sections: Answer, Evidence, Limitations and Safety.
"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 700},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body.get("response", "").strip()


def build_generator(backend: str = "template", model: str = "deepseek-r1:7b"):
    if backend == "template":
        return GroundedAnswerGenerator()
    if backend == "ollama":
        return OllamaGenerator(model=model)
    raise ValueError(f"Unsupported generator backend: {backend}")


def _best_evidence_sentence(text: str, query_terms: set[str]) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return text[:260].strip()

    def score(sentence: str) -> int:
        return len(query_terms & set(tokenize(sentence)))

    best = max(sentences, key=score)
    return best[:320].strip()


def _keep_strong_evidence(evidence: list[ScoredChunk]) -> list[ScoredChunk]:
    """Keep near-top chunks so weak cross-topic matches do not enter the answer."""
    if not evidence:
        return []
    threshold = evidence[0].score * 0.80
    kept = [hit for hit in evidence if hit.score >= threshold]
    return kept or evidence[:1]


def _make_short_answer(query: str, bullets: list[str]) -> str:
    if not bullets:
        return "Insufficient evidence was retrieved."
    topic = query.rstrip("?")
    return (
        f"For the question '{topic}', the retrieved evidence suggests the answer should be limited "
        "to the cited findings below."
    )

