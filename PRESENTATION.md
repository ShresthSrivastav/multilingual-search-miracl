# Presentation Outline: Multilingual Search Using MIRACL

## Slide 1 - Title

Multilingual Search Using MIRACL

## Slide 2 - Problem

Keyword matching misses paraphrases and does not naturally bridge language differences. This project compares lexical and multilingual semantic retrieval.

## Slide 3 - Dataset

MIRACL is a multilingual Wikipedia passage retrieval dataset. The app uses a small balanced cloud sample and can build a local index from all 18 language shards.

## Slide 4 - Objectives

- Search in multiple languages
- Compare semantic, keyword, hybrid, and reranked methods
- Evaluate retrieval with standard ranking metrics
- Show inspectable evidence for every result

## Slide 5 - Architecture

Query -> E5 embedding and SQLite FTS -> rank fusion -> optional cross-encoder -> highlighted passages

## Slide 6 - Methods

Explain E5 query/passage prefixes and cosine similarity, SQLite full-text candidates, reciprocal-rank fusion, and multilingual reranking.

## Slide 7 - Full corpus index

The builder batches vectors into memory-mapped files and stores passage text for retrieval. Exact vector search has bounded RAM usage but still scans the vectors.

## Slide 8 - Application

Show Search, Compare methods, and Analysis tabs; demonstrate highlighted query evidence and language distribution charts.

## Slide 9 - Evaluation and limitations

Show Recall@5 and MRR@10. Explain why the four-query classroom sample is only a pipeline demo; propose official MIRACL dev topics for stronger evidence.

## Slide 10 - Conclusion and future scope

Summarize retrieval comparison and grounded evidence. Future work: all-query evaluation, approximate indexing if latency measurements justify it, and stronger per-language analysis.

## Viva questions

1. What is multilingual semantic search?
2. What is MIRACL and what does a passage contain?
3. Why does E5 use `query:` and `passage:` prefixes?
4. How does cosine similarity rank passages?
5. What does SQLite FTS5 do in this project?
6. How does reciprocal-rank fusion combine rankings?
7. What is the role of the cross-encoder reranker?
8. What does Recall@5 measure?
9. What does MRR@10 measure?
10. Why store a full index locally?
11. What tradeoff remains with exact vector search?
12. How does highlighting help verify retrieved evidence?
13. Why is a four-query evaluation insufficient as a benchmark?
14. How can the app be evaluated fairly across languages?
15. Does the app generate answers or retrieve source text?
