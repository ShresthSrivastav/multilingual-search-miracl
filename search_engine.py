"""Small, dependency-light semantic retrieval engine for the MIRACL demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import json
import math
import re
from typing import Iterable, Sequence

import numpy as np


SUPPORTED_LANGUAGES = {"en": "English", "hi": "Hindi", "es": "Spanish", "ar": "Arabic"}
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?\u0964\u0965؟])\s+|\n+")


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


def keyword_rank(query: str, documents: Sequence[Document], top_k: int = 5) -> list[tuple[int, float]]:
    """Rank documents with a small TF-IDF cosine baseline, without extra dependencies."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    query_terms = Counter(TOKEN_PATTERN.findall(query.casefold()))
    if not query_terms or not documents:
        return []

    document_terms = [Counter(TOKEN_PATTERN.findall(f"{document.title} {document.text}".casefold())) for document in documents]
    document_frequency = Counter(term for terms in document_terms for term in terms)
    total_documents = len(document_terms)

    def weight(term: str, frequency: int) -> float:
        inverse_document_frequency = math.log((1 + total_documents) / (1 + document_frequency[term])) + 1.0
        return (1.0 + math.log(frequency)) * inverse_document_frequency

    query_vector = {term: weight(term, frequency) for term, frequency in query_terms.items()}
    query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
    scores: list[tuple[int, float]] = []
    for index, terms in enumerate(document_terms):
        document_vector = {term: weight(term, frequency) for term, frequency in terms.items() if term in query_vector}
        if not document_vector:
            continue
        dot_product = sum(query_vector[term] * value for term, value in document_vector.items())
        document_norm = math.sqrt(sum(value * value for value in document_vector.values()))
        scores.append((index, dot_product / (query_norm * document_norm)))
    return sorted(scores, key=lambda item: (-item[1], item[0]))[:top_k]


def diversify_ranked(
    ranked: Sequence[tuple[int, float]], documents: Sequence[Document], top_k: int
) -> list[tuple[int, float]]:
    """Prefer different article titles before returning more passages from one article."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    selected: list[tuple[int, float]] = []
    seen_titles: set[str] = set()
    for require_new_title in (True, False):
        for index, score in ranked:
            is_new_title = documents[index].title not in seen_titles
            if require_new_title and not is_new_title:
                continue
            selected.append((index, score))
            seen_titles.add(documents[index].title)
            if len(selected) == top_k:
                return selected
    return selected


def extractive_answer(query: str, text: str, max_sentences: int = 2) -> str:
    """Select supporting sentences from a passage; never invents text."""
    if max_sentences < 1:
        raise ValueError("max_sentences must be at least 1")
    sentences = [sentence.strip() for sentence in SENTENCE_PATTERN.split(text) if sentence.strip()]
    if not sentences:
        return ""
    query_terms = set(TOKEN_PATTERN.findall(query.casefold()))
    if not query_terms:
        return " ".join(sentences[:max_sentences])
    scored = []
    for index, sentence in enumerate(sentences):
        sentence_terms = set(TOKEN_PATTERN.findall(sentence.casefold()))
        overlap = len(query_terms & sentence_terms)
        scored.append((overlap, -index, sentence))
    selected = sorted(scored, reverse=True)[:max_sentences]
    return " ".join(sentence for _, _, sentence in sorted(selected, key=lambda item: -item[1]))


def recall_at_k(results: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return float(bool(set(results[:k]) & relevant))


def reciprocal_rank(results: Sequence[str], relevant: set[str], k: int) -> float:
    for rank, docid in enumerate(results[:k], start=1):
        if docid in relevant:
            return 1.0 / rank
    return 0.0
