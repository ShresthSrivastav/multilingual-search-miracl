"""Small, dependency-light semantic retrieval engine for the MIRACL demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterable, Sequence

import numpy as np


SUPPORTED_LANGUAGES = {"en": "English", "hi": "Hindi", "es": "Spanish", "ar": "Arabic"}


@dataclass(frozen=True)
class Document:
    docid: str
    title: str
    text: str
    language: str


def load_documents(path: str | Path) -> list[Document]:
    """Load and validate the JSONL corpus used by the demo."""
    documents: list[Document] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                document = Document(
                    docid=str(item["docid"]),
                    title=str(item["title"]),
                    text=str(item["text"]),
                    language=str(item["language"]),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid corpus record on line {line_number}") from exc
            if document.language not in SUPPORTED_LANGUAGES:
                raise ValueError(f"Unsupported language {document.language!r} on line {line_number}")
            documents.append(document)
    return documents


def rank_embeddings(
    query_embedding: Sequence[float] | np.ndarray,
    document_embeddings: Sequence[Sequence[float]] | np.ndarray,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Return document indexes and cosine scores in descending order."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    query = np.asarray(query_embedding, dtype=np.float32)
    documents = np.asarray(document_embeddings, dtype=np.float32)
    if query.ndim != 1 or documents.ndim != 2 or documents.shape[1] != query.shape[0]:
        raise ValueError("query and document embeddings have incompatible shapes")
    if not len(documents):
        return []

    query_norm = np.linalg.norm(query)
    document_norms = np.linalg.norm(documents, axis=1)
    if query_norm == 0 or np.any(document_norms == 0):
        raise ValueError("embeddings must not contain zero vectors")
    scores = (documents @ query) / (document_norms * query_norm)
    indexes = np.argsort(-scores, kind="stable")[:top_k]
    return [(int(index), float(scores[index])) for index in indexes]


def filter_documents(documents: Sequence[Document], language: str) -> list[Document]:
    if language == "all":
        return list(documents)
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language {language!r}")
    return [document for document in documents if document.language == language]


def recall_at_k(results: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return float(bool(set(results[:k]) & relevant))


def reciprocal_rank(results: Sequence[str], relevant: set[str], k: int) -> float:
    for rank, docid in enumerate(results[:k], start=1):
        if docid in relevant:
            return 1.0 / rank
    return 0.0

