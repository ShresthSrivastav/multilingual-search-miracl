"""Evaluate retrieval against a small JSONL query/qrels file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

# Allow `python scripts/evaluate.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_engine import filter_documents, hybrid_rank, keyword_rank, load_documents, rank_embeddings, recall_at_k, reciprocal_rank


def evaluate(corpus_path: Path, queries_path: Path, model_name: str, reranker_name: str) -> list[dict[str, float | str]]:
    documents = load_documents(corpus_path)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        [f"passage: {document.title}. {document.text}" for document in documents],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    reranker = CrossEncoder(reranker_name)
    metric_rows = {name: {"recall": [], "mrr": []} for name in ("Semantic E5", "Keyword baseline", "Hybrid RRF", "Hybrid + reranker")}
    with queries_path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            candidates = filter_documents(documents, item.get("language", "all"))
            candidate_indexes = [documents.index(document) for document in candidates]
            candidate_embeddings = np.asarray([embeddings[index] for index in candidate_indexes])
            query_embedding = model.encode([f"query: {item['query']}"], normalize_embeddings=True, show_progress_bar=False)[0]
            semantic = rank_embeddings(query_embedding, candidate_embeddings, top_k=min(50, len(candidates)))
            lexical = keyword_rank(item["query"], candidates, top_k=50)
            hybrid = hybrid_rank(semantic, lexical, top_k=20)
            reranked_scores = reranker.predict(
                [(item["query"], f"{candidates[index].title}. {candidates[index].text}") for index, _ in hybrid],
                show_progress_bar=False,
            )
            reranked = sorted(((index, float(score)) for (index, _), score in zip(hybrid, reranked_scores)), key=lambda result: -result[1])
            relevant = set(item["relevant_docids"])
            for name, ranking in (("Semantic E5", semantic), ("Keyword baseline", lexical), ("Hybrid RRF", hybrid), ("Hybrid + reranker", reranked)):
                docids = [candidates[index].docid for index, _ in ranking]
                metric_rows[name]["recall"].append(recall_at_k(docids, relevant, 5))
                metric_rows[name]["mrr"].append(reciprocal_rank(docids, relevant, 10))
    if not metric_rows["Semantic E5"]["recall"]:
        raise ValueError("No evaluation queries found")
    return [{
        "Method": name,
        "Recall@5": float(np.mean(values["recall"])),
        "MRR@10": float(np.mean(values["mrr"])),
        "Queries": len(values["recall"]),
    } for name, values in metric_rows.items()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/sample_corpus.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/eval_queries.jsonl"))
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--reranker", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    args = parser.parse_args()
    for row in evaluate(args.corpus, args.queries, args.model, args.reranker):
        print(f"{row['Method']}: Recall@5={row['Recall@5']:.3f}, MRR@10={row['MRR@10']:.3f} (n={row['Queries']})")


if __name__ == "__main__":
    main()
