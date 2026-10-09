# Multilingual Search Using MIRACL

## Abstract

This project implements a multilingual semantic search system using a sample of the MIRACL retrieval corpus. Users can submit queries in English, Hindi, Spanish, or Arabic and receive passages ranked by semantic similarity. Unlike keyword search, the system represents both queries and passages as multilingual embeddings, allowing it to match meaning across languages and wording variations.

## 1. Objectives

1. Build a working multilingual search interface.
2. Use a standard multilingual embedding model for semantic retrieval.
3. Demonstrate ranking with cosine similarity.
4. Evaluate retrieval with Recall@5 and MRR@10.
5. Provide a reproducible and explainable college-level implementation.
6. Compare semantic retrieval with a traditional keyword baseline.
7. Produce a grounded answer candidate using only retrieved evidence.

## 2. Dataset

MIRACL is a multilingual information-retrieval dataset built from Wikipedia passages. This project uses a small sample from the English, Hindi, Spanish, and Arabic corpus configurations. Each record contains a document ID, title, passage text, and language code.

The full MIRACL corpus is not bundled because it is too large for a normal classroom laptop. The sampler can retrieve a larger sample when required.

## 3. Methodology

### Preprocessing

Each passage is represented as:

```text
passage: <title>. <passage text>
```

The `intfloat/multilingual-e5-small` model encodes these strings into normalized vectors. User queries are represented as:

```text
query: <user query>
```

### Retrieval

For every query, the application computes the dot product between the normalized query vector and normalized passage vectors. With normalized vectors, the dot product is equivalent to cosine similarity. Results are sorted from highest to lowest score.

The project also includes a dependency-free TF-IDF-style keyword baseline. It uses Unicode-aware tokenization, term frequency, inverse document frequency, and cosine similarity. This provides a simple traditional baseline for comparison in the report and presentation.

### Interface

The Streamlit interface provides a language filter, top-k control, search-method selector, example queries, and ranked result cards. Expensive model loading and corpus encoding are cached.

The interface displays an extractive answer candidate from the highest-ranked passage. The answer is made from source sentences only, with a visible document ID, so the system does not present unsupported generated text as fact.

## 4. System Design

```text
User query
    ↓
Streamlit interface
    ↓
E5 query embedding
    ↓
NumPy cosine ranking
    ↓
Ranked MIRACL passages
```

## 5. Evaluation

The demonstration evaluation set contains one query per supported language. The relevant document IDs are stored in `data/eval_queries.jsonl`. The evaluation compares multilingual E5 semantic search against the keyword baseline.

Run:

```bash
python scripts/evaluate.py
```

Record the generated values here before submission:

| Method | Recall@5 | MRR@10 |
|---|---:|---:|
| Multilingual E5 semantic search | 0.750 | 0.750 |
| Keyword baseline | 0.750 | 0.625 |

The evaluation sample is intended to demonstrate the metric pipeline, not to represent a statistically complete MIRACL benchmark.

For a stronger experiment, run `scripts/download_miracl_dev.py` and `scripts/evaluate_official.py`. The official evaluator reports how many development queries are covered by the selected local corpus before calculating retrieval metrics. This prevents a small sample from being incorrectly presented as a full benchmark.

Example bounded run on the included 4,000-passage local corpus: 31 of 4,693 official topics were covered (0.7%), with Recall@5 of 0.935 and MRR@10 of 0.754. The low coverage is expected because the full MIRACL corpus is not bundled.

## 6. Limitations

- The bundled corpus is small and does not represent the full MIRACL distribution.
- NumPy ranking is linear in the number of passages.
- Retrieval quality depends on the pretrained embedding model.
- The system retrieves passages but does not generate answers or citations.
- The model requires an initial online download.
- The comparison set is intentionally small and is not a full MIRACL benchmark.
- Extractive answers are sentence selections, not independent natural-language reasoning.

## 7. Future Scope

- Index the complete MIRACL corpus with FAISS or a vector database.
- Add all available MIRACL languages.
- Compare BM25, multilingual E5, and a reranked hybrid system.
- Add query translation and language identification.
- Add answer generation with retrieved passages as context.
- Index the complete MIRACL corpus and evaluate on the official development queries and qrels.

## 8. Conclusion

The project demonstrates the complete pipeline of a multilingual semantic search system: data loading, multilingual representation, similarity-based ranking, web presentation, and metric evaluation. Its small design makes the core method easy to understand while leaving clear paths for scaling.
