"""Evaluate the local corpus against official MIRACL development topics/qrels."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_engine import filter_documents, load_documents, rank_embeddings, recall_at_k, reciprocal_rank


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(corpus_path: Path, topics_path: Path, qrels_path: Path, model_name: str, max_queries: int = 100) -> dict[str, float]:
    documents = load_documents(corpus_path)
    topics = read_jsonl(topics_path)
    qrels: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in read_jsonl(qrels_path):
        qrels[(item["language"], item["qid"])].add(item["docid"])
    document_ids = {document.docid for document in documents}
    eligible = [item for item in topics if qrels[(item["language"], item["qid"])] & document_ids]
    if not eligible:
        return {"topics": float(len(topics)), "covered_topics": 0.0, "coverage": 0.0, "recall_at_5": 0.0, "mrr_at_10": 0.0}

    if max_queries > 0:
        eligible = eligible[:max_queries]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        [f"passage: {document.title}. {document.text}" for document in documents],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    document_indexes = {document.docid: index for index, document in enumerate(documents)}
    candidate_cache = {}
    for language in {item["language"] for item in eligible}:
        candidates = filter_documents(documents, language)
        candidate_cache[language] = (candidates, np.asarray([embeddings[document_indexes[document.docid]] for document in candidates]))
    query_embeddings = model.encode(
        [f"query: {item['query']}" for item in eligible],
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    recalls: list[float] = []
    ranks: list[float] = []
    for item, query_embedding in zip(eligible, query_embeddings):
        candidates, candidate_embeddings = candidate_cache[item["language"]]
        ranked = rank_embeddings(query_embedding, candidate_embeddings, top_k=10)
        ranked_docids = [candidates[index].docid for index, _ in ranked]
        relevant = qrels[(item["language"], item["qid"])]
        recalls.append(recall_at_k(ranked_docids, relevant, 5))
        ranks.append(reciprocal_rank(ranked_docids, relevant, 10))
    return {
        "topics": float(len(topics)),
        "covered_topics": float(len(eligible)),
        "coverage": float(len(eligible) / len(topics)) if topics else 0.0,
        "recall_at_5": float(np.mean(recalls)),
        "mrr_at_10": float(np.mean(ranks)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/local_corpus.jsonl"))
    parser.add_argument("--topics", type=Path, default=Path("data/miracl_dev_topics.jsonl"))
    parser.add_argument("--qrels", type=Path, default=Path("data/miracl_dev_qrels.jsonl"))
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--max-queries", type=int, default=100, help="Queries to evaluate; use 0 for all eligible queries")
    args = parser.parse_args()
    metrics = evaluate(args.corpus, args.topics, args.qrels, args.model, args.max_queries)
    print(f"Official topics: {int(metrics['topics'])}")
    print(f"Covered by local corpus: {int(metrics['covered_topics'])} ({metrics['coverage']:.1%})")
    print(f"Recall@5: {metrics['recall_at_5']:.3f}")
    print(f"MRR@10: {metrics['mrr_at_10']:.3f}")


if __name__ == "__main__":
    main()
