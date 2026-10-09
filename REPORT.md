# Multilingual Search Using MIRACL

## Abstract

This project compares semantic, lexical, hybrid, and reranked retrieval over the MIRACL multilingual passage collection. A small balanced sample supports the hosted demonstration, while a local builder creates disk-backed vectors and full-text indexes for the complete downloaded collection. Results include passage evidence and source identifiers.

## Objectives

1. Build a multilingual passage search application.
2. Compare multilingual semantic retrieval with lexical retrieval.
3. Combine the ranking methods and test a multilingual reranker.
4. Report Recall@5 and MRR@10.
5. Make a local full-corpus index usable without loading the entire corpus into RAM.
6. Highlight query terms in the retrieved source evidence.

## Dataset

MIRACL consists of Wikipedia passages and retrieval topics across 18 languages. The cloud app uses a small balanced sample; the local downloader can retrieve all available corpus shards. Each passage has a document ID, title, text, and language code.

## Methodology

### Semantic retrieval

The `intfloat/multilingual-e5-small` encoder embeds passages as `passage: <title>. <text>` and queries as `query: <query>`. The vectors are normalized; dot product therefore gives cosine similarity.

### Lexical and hybrid retrieval

The bundled JSONL demo uses a lightweight TF-IDF keyword baseline; the full local index uses SQLite FTS5 for lexical candidates. Hybrid search combines semantic and lexical rankings with reciprocal-rank fusion. A multilingual cross-encoder optionally reranks the fused candidates. Its language coverage is smaller than the 18-language corpus, so reranker results should be checked per language.

### Full local index

The index builder processes compressed MIRACL shards in batches, writing float16 normalized vectors as memory-mapped NumPy files and passage text/metadata into SQLite FTS5. Query-time vector scoring is chunked to limit RAM. It remains an exact linear scan, so larger indexes can take longer to search.

### User interface and evidence

The Streamlit app includes Search, Compare methods, and Analysis tabs. Users can compare all four methods on the same query. Matching query terms are highlighted in source passages; the answer excerpt is selected from retrieved text and is not generated.

## System design

```text
Query
  -> multilingual E5 vector + SQLite FTS candidates
  -> semantic / lexical / hybrid ranking
  -> optional multilingual cross-encoder reranking
  -> ranked MIRACL passages with highlighted evidence
```

## Evaluation

The bundled classroom evaluation has four queries, one for each initial demo language. Run `python scripts/evaluate.py` to compare semantic E5, keyword, hybrid RRF, and hybrid plus reranker. The app displays saved results and charts in Analysis.

| Corpus | Method | Recall@5 | MRR@10 |
|---|---|---:|---:|
| Cloud sample | Semantic E5 | 0.750 | 0.750 |
| Cloud sample | Keyword baseline | 0.750 | 0.625 |
| Cloud sample | Hybrid RRF | 0.750 | 0.750 |
| Cloud sample | Hybrid + reranker | 0.750 | 0.750 |
| 4,000-passage local sample | Semantic E5 | 1.000 | 0.750 |
| 4,000-passage local sample | Keyword baseline | 0.500 | 0.188 |
| 4,000-passage local sample | Hybrid RRF | 0.750 | 0.417 |
| 4,000-passage local sample | Hybrid + reranker | 1.000 | 0.875 |

These four queries only demonstrate the metric pipeline; they are not a full benchmark.

For a stronger result, download official development topics/qrels and run `python scripts/evaluate_official.py`. Report corpus coverage with Recall@5 and MRR@10; the supplied official evaluator currently measures semantic retrieval.

## Limitations

- The bundled four-query metric sample is not statistically representative.
- Automatic language identification uses scripts and simple word markers and can misclassify short queries.
- Exact vector scoring scales linearly with the number of passages.
- The cross-encoder requires a separate model download and adds CPU latency.
- The system retrieves evidence and excerpts sentences; it does not generate or translate answers.

## Future work

- Compare the four methods on all official development queries.
- Add approximate nearest-neighbor search if measured full-index latency requires it.
- Add quality analysis by language and query type.
- Improve language identification and evaluate cross-language retrieval separately.

## Conclusion

The project demonstrates an end-to-end multilingual retrieval workflow, from MIRACL ingestion and indexing to ranked, inspectable evidence. The small hosted sample keeps the demo practical, while the local index makes larger experiments possible.
