# Multilingual MIRACL Search

A college-level multilingual information retrieval project using Streamlit, multilingual E5 embeddings, SQLite full-text search, and MIRACL. The GitHub demo uses a small corpus; locally, the full downloaded corpus can be indexed and searched without loading all passage text into RAM.

## Features

- Semantic E5, keyword, hybrid rank fusion, and multilingual cross-encoder reranking
- Same-query method comparison with latency and ranked evidence
- Query term highlighting in retrieved passages
- Analysis charts for corpus language distribution and Recall@5/MRR@10
- Disk-backed precomputed vectors and SQLite FTS5 for the full local collection
- Support for all 18 MIRACL languages when present in the corpus/index
- Corpus sampler, official MIRACL evaluator, and unit tests

## Setup

Python 3.12 is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The embedding model downloads from Hugging Face the first time semantic search is used. The reranker downloads only if selected. A network connection is required for first use.

## Local sample

When available, local startup automatically uses the ignored 4,000-passage four-language corpus; Streamlit Cloud falls back to the small checked-in sample.

```powershell
python -m streamlit run app.py
```

Set `MIRACL_CORPUS_PATH` only when you want to choose a different corpus file. The checked-in `data/sample_corpus.jsonl` remains the Streamlit Cloud fallback.
The app uses `data/local_corpus.jsonl` by default when present. An existing index is used only when you explicitly set `MIRACL_INDEX_PATH`, so an unrelated or incomplete index cannot replace the 4,000-passage local corpus by accident.

## Optional full MIRACL local index

The 4,000-passage local corpus is the intended setup for this college demo; you do not need to download or index the full corpus. The following is an optional extension for machines with substantial free disk space and time.

Download all 18 language shards (about 16 GB in this workspace):

```powershell
python scripts/download_full_corpus.py
```

Build the index. This is a long-running, disk-intensive first-time job for the full corpus:

```powershell
python scripts/build_index.py --source data/miracl_full --output data/miracl_index
```

Then run the app against that index:

```powershell
python -m streamlit run app.py
```

After building an index, opt into it by setting `$env:MIRACL_INDEX_PATH = "data/miracl_index"` before starting Streamlit. Embedding the full corpus can take days depending on hardware; the builder reports progress and saves completed language shards so it can resume. Streamlit Cloud continues to use its bundled small sample.

The builder batches normalized E5 vectors into memory-mapped files and stores searchable passage text and metadata in SQLite FTS5. It saves a manifest after each shard, so it can resume after an interruption. Similarity is exact and chunked, so RAM use is bounded but query time still grows with corpus size. For the 4,000-row local sample, use:

```powershell
python scripts/build_index.py --source data/local_corpus.jsonl --output data/local_sample_index --languages en hi es ar
$env:MIRACL_INDEX_PATH = "data/local_sample_index"
python -m streamlit run app.py
```

Large corpus and index files are excluded from Git. Streamlit Cloud continues to use the small bundled sample.

## Refresh the sample

```powershell
python scripts/download_sample.py --profile cloud --languages en hi es ar --per-language 50
python scripts/download_sample.py --profile local --languages en hi es ar --per-language 1000
```

## Evaluation

```powershell
python scripts/evaluate.py
python scripts/download_miracl_dev.py
python scripts/evaluate_official.py --max-queries 100
```

The Analysis tab shows saved classroom metrics and charts immediately. The bundled set has four queries, so treat it as a pipeline demonstration. The evaluation script compares all four methods. The official evaluator reports corpus coverage alongside Recall@5 and MRR@10.

## Architecture

1. `multilingual-e5-small` embeds queries and passages with the required `query:` and `passage:` prefixes.
2. Semantic retrieval ranks precomputed vectors; the local index uses SQLite FTS5 lexical candidates, while the small JSONL demo uses a lightweight TF-IDF baseline.
3. Hybrid search combines rankings with reciprocal-rank fusion.
4. An optional multilingual cross-encoder reranks the candidate passages.
5. Streamlit shows ranked evidence and highlights query terms in the source passages.

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## Troubleshooting

- **Model download fails:** check internet access and retry.
- **No results:** choose the correct language or use All languages.
- **Full corpus is not active:** set `MIRACL_INDEX_PATH` to the generated index directory.
- **Reranker first run is slow:** it downloads a separate multilingual model; other methods do not need it.
- **MIRACL API rate limit:** wait and retry the sampler.

## References

- MIRACL corpus: https://huggingface.co/datasets/miracl/miracl-corpus
- MIRACL project: https://github.com/project-miracl/miracl
- E5 model: https://huggingface.co/intfloat/multilingual-e5-small
- Cross-encoder: https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
