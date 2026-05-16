from __future__ import annotations

from pathlib import Path
import re

from .chunking import Document


def _slug(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    metadata: dict[str, str] = {}
    body_lines: list[str] = []
    in_header = True

    for line in raw.splitlines():
        stripped = line.strip()
        if in_header and stripped.startswith("#"):
            key_value = stripped.lstrip("#").strip()
            if ":" in key_value:
                key, value = key_value.split(":", 1)
                metadata[key.strip().lower()] = value.strip()
            continue
        if in_header and not stripped:
            in_header = False
            continue
        in_header = False
        body_lines.append(line)

    return metadata, "\n".join(body_lines).strip()


def load_documents(data_dir: str | Path) -> list[Document]:
    """Load small raw-text documents with optional '# Key: value' headers."""
    root = Path(data_dir)
    documents: list[Document] = []

    for path in sorted(root.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        metadata, body = _parse_front_matter(raw)
        doc_id = metadata.get("doc_id") or _slug(path)
        title = metadata.get("title") or path.stem.replace("_", " ").title()

        documents.append(
            Document(
                doc_id=doc_id,
                title=title,
                text=body,
                metadata={
                    "doc_id": doc_id,
                    "title": title,
                    "source_file": path.name,
                    "journal": metadata.get("journal", "sample medical notes"),
                    "year": int(metadata["year"]) if metadata.get("year", "").isdigit() else metadata.get("year", ""),
                    "source": metadata.get("source", "demo corpus"),
                },
            )
        )

    if not documents:
        raise FileNotFoundError(f"No .txt documents found in {root}")

    return documents

