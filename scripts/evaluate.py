"""Evaluate retrieval against a small JSONL query/qrels file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

# Allow `python scripts/evaluate.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_engine import filter_documents, keyword_rank, load_documents, rank_embeddings, recall_at_k, reciprocal_rank


def evaluate(corpus_path: Path, queries_path: Path, model_name: str) -> dict[str, float]:
    documents = load_documents(corpus_path)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        [f"passage: {document.title}. {document.text}" for document in documents],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    semantic_recalls: list[float] = []
    semantic_ranks: list[float] = []
    keyword_recalls: list[float] = []
    keyword_ranks: list[float] = []
    with queries_path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            candidates = filter_documents(documents, item.get("language", "all"))
            candidate_indexes = [documents.index(document) for document in candidates]
            candidate_embeddings = np.asarray([embeddings[index] for index in candidate_indexes])
            query_embedding = model.encode([f"query: {item['query']}"], normalize_embeddings=True, show_progress_bar=False)[0]
            ranked = rank_embeddings(query_embedding, candidate_embeddings, top_k=10)
            ranked_docids = [candidates[index].docid for index, _ in ranked]
            relevant = set(item["relevant_docids"])
            semantic_recalls.append(recall_at_k(ranked_docids, relevant, 5))
            semantic_ranks.append(reciprocal_rank(ranked_docids, relevant, 10))
            keyword_results = keyword_rank(item["query"], candidates, top_k=10)
            keyword_docids = [candidates[index].docid for index, _ in keyword_results]
            keyword_recalls.append(recall_at_k(keyword_docids, relevant, 5))
            keyword_ranks.append(reciprocal_rank(keyword_docids, relevant, 10))
    if not semantic_recalls:
        raise ValueError("No evaluation queries found")
    return {
        "queries": float(len(semantic_recalls)),
        "semantic_recall_at_5": float(np.mean(semantic_recalls)),
        "semantic_mrr_at_10": float(np.mean(semantic_ranks)),
        "keyword_recall_at_5": float(np.mean(keyword_recalls)),
        "keyword_mrr_at_10": float(np.mean(keyword_ranks)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/sample_corpus.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/eval_queries.jsonl"))
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    args = parser.parse_args()
    metrics = evaluate(args.corpus, args.queries, args.model)
    print(f"Evaluation queries: {int(metrics['queries'])}")
    print(f"Semantic Recall@5: {metrics['semantic_recall_at_5']:.3f}")
    print(f"Semantic MRR@10: {metrics['semantic_mrr_at_10']:.3f}")
    print(f"Keyword Recall@5: {metrics['keyword_recall_at_5']:.3f}")
    print(f"Keyword MRR@10: {metrics['keyword_mrr_at_10']:.3f}")


if __name__ == "__main__":
    main()
