# Multilingual MIRACL Search

A college-level semantic search project built with Python, Streamlit, NumPy, and a multilingual sentence-transformer. It searches a small MIRACL corpus sample in English, Hindi, Spanish, and Arabic.

## Features

- Cross-language semantic retrieval using `intfloat/multilingual-e5-small`
- Language filtering and configurable top-k results
- Ranked results with cosine similarity scores
- Keyword baseline for an explainable comparison
- Reproducible sampler for the official MIRACL corpus
- Recall@5 and MRR@10 evaluation script
- Plain unit tests for the ranking and filtering logic

## Project structure

```text
.
├── app.py
├── search_engine.py
├── data/
│   ├── sample_corpus.jsonl
│   └── eval_queries.jsonl
├── scripts/
│   ├── download_sample.py
│   └── evaluate.py
├── tests/test_search_engine.py
├── REPORT.md
├── PRESENTATION.md
└── requirements.txt
```

## Setup

Python 3.12 is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

The embedding model is downloaded from Hugging Face on first use. An internet connection is required the first time the app or evaluation script is run.

## Run the application

```bash
python -m streamlit run app.py
```

If the `streamlit` command is already mapped to the active virtual environment, `streamlit run app.py` works as well.

Try the provided example queries or enter a query in any of the four supported languages. The model encodes queries with the `query: ` prefix and passages with the `passage: ` prefix because this is the format used during E5 training.

## Refresh the MIRACL sample

The checked-in corpus is intentionally small. To download a new sample from the official MIRACL corpus:

```bash
python scripts/download_sample.py --languages en hi es ar --per-language 50
```

The script uses the Hugging Face dataset-row API, so it does not need to download the complete multi-gigabyte corpus. The generated JSONL file replaces `data/sample_corpus.jsonl`.

## Evaluate retrieval

```bash
python scripts/evaluate.py
```

The evaluation file contains four demonstration queries and their relevant document IDs. The script compares semantic retrieval with the keyword baseline and prints Recall@5, MRR@10, and the number of evaluation queries.

## Run tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Architecture

1. The corpus loader reads validated JSONL records.
2. The E5 model converts each passage into a normalized vector, or the keyword baseline builds TF-IDF-style token vectors.
3. A query is converted into a normalized query vector or token vector.
4. The selected ranker scores and sorts candidate passages.
5. Streamlit displays the highest-scoring results.

For a large production corpus, replace the in-memory NumPy ranking step with an approximate nearest-neighbor index such as FAISS or a vector database.

## Troubleshooting

- **Model download fails:** check internet access and retry; Hugging Face caches the model after a successful download.
- **Streamlit is not found:** activate the virtual environment and run `pip install -r requirements.txt`.
- **No results appear:** enter a non-empty query or choose an example query.
- **MIRACL API rate limit:** wait and retry the sampler, or use the included sample corpus.

## Data and references

- MIRACL corpus: https://huggingface.co/datasets/miracl/miracl-corpus
- MIRACL project: https://github.com/project-miracl/miracl
- Multilingual E5 model: https://huggingface.co/intfloat/multilingual-e5-small
