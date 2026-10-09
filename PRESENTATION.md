# Presentation Outline: Multilingual Search Using MIRACL

## Slide 1 — Title

Multilingual Search Using MIRACL

## Slide 2 — Problem Statement

Keyword search often struggles with different languages, scripts, and paraphrased queries. The project explores semantic retrieval across multiple languages.

## Slide 3 — What Is MIRACL?

MIRACL is a multilingual information-retrieval dataset containing Wikipedia passages and relevance data across many languages.

## Slide 4 — Project Objectives

- Build a working multilingual search demo
- Search English, Hindi, Spanish, and Arabic passages
- Rank results using semantic similarity
- Measure retrieval quality

## Slide 5 — System Architecture

User query → multilingual embedding model → cosine similarity → ranked MIRACL passages

## Slide 6 — Technology Stack

Python, Streamlit, Sentence Transformers, multilingual E5, NumPy, JSONL, Hugging Face dataset API.

## Slide 7 — Search Workflow

1. Read the selected language corpus.
2. Encode passages with the `passage:` prefix.
3. Encode the user query with the `query:` prefix.
4. Compute similarity scores.
5. Display the top-k passages.

## Slide 8 — User Interface

Show the language selector, example queries, top-k slider, score, document ID, title, and passage text.

## Slide 9 — Baseline Comparison and Limitations

Compare multilingual E5 semantic search against the TF-IDF-style keyword baseline using Recall@5 and MRR@10. Discuss small sample size, model download requirement, and linear NumPy ranking.

## Slide 10 — Grounded Answers, Conclusion, and Future Work

The app extracts an answer candidate only from the best retrieved passage. Future work includes full-corpus indexing, more languages, reranking, and optional generative answer synthesis with citations.

## Viva Questions

1. What is multilingual semantic search?
2. What is MIRACL used for?
3. Why use embeddings instead of exact keyword matching?
4. Why are `query:` and `passage:` prefixes used?
5. What is cosine similarity?
6. Why are embeddings normalized?
7. What does top-k mean?
8. What is Recall@5?
9. What is MRR@10?
10. Why is the full MIRACL corpus not bundled?
11. How would you scale this system?
12. What is the difference between retrieval and answer generation?
13. Why is a keyword baseline useful in an information-retrieval project?
14. What would happen if the corpus were expanded to millions of passages?
15. How does the extractive answer layer avoid hallucination?
16. Why should corpus coverage be reported during evaluation?
