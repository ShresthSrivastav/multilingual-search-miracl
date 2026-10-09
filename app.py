"""Streamlit UI for the multilingual MIRACL semantic search project."""

from pathlib import Path
import json
import os

import streamlit as st
from sentence_transformers import SentenceTransformer

from search_engine import (
    SUPPORTED_LANGUAGES,
    Document,
    diversify_ranked,
    detect_language,
    extractive_answer,
    filter_documents,
    keyword_rank,
    load_documents,
    rank_embeddings,
    recall_at_k,
    reciprocal_rank,
)


ROOT = Path(__file__).parent
configured_corpus = Path(os.getenv("MIRACL_CORPUS_PATH", "data/sample_corpus.jsonl"))
CORPUS_PATH = configured_corpus if configured_corpus.is_absolute() else ROOT / configured_corpus
MODEL_NAME = "intfloat/multilingual-e5-small"


@st.cache_resource(show_spinner="Loading multilingual embedding model...")
def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@st.cache_data(show_spinner="Encoding the corpus...")
def encode_documents(_model: SentenceTransformer, documents: tuple[Document, ...]):
    passages = [f"passage: {document.title}. {document.text}" for document in documents]
    return _model.encode(passages, normalize_embeddings=True, show_progress_bar=False)


def search(model: SentenceTransformer, documents: list[Document], query: str, top_k: int):
    if not query.strip() or not documents:
        return []
    embeddings = encode_documents(model, tuple(documents))
    query_embedding = model.encode([f"query: {query.strip()}"], normalize_embeddings=True, show_progress_bar=False)[0]
    ranked = rank_embeddings(query_embedding, embeddings, min(len(documents), top_k * 3))
    return [(documents[index], score) for index, score in diversify_ranked(ranked, documents, top_k)]


def keyword_search(documents: list[Document], query: str, top_k: int):
    if not query.strip() or not documents:
        return []
    ranked = keyword_rank(query, documents, min(len(documents), top_k * 3))
    return [(documents[index], score) for index, score in diversify_ranked(ranked, documents, top_k)]


def corpus_examples(documents: list[Document]) -> dict[str, str]:
    """Use topics that actually exist in the selected corpus profile."""
    examples: dict[str, str] = {}
    for code, name in SUPPORTED_LANGUAGES.items():
        for document in documents:
            if document.language == code:
                examples[f"{name}: {document.title}"] = document.title
                break
    return examples


@st.cache_data(show_spinner="Running demo evaluation...")
def demo_evaluation(_model: SentenceTransformer, documents: tuple[Document, ...]) -> list[dict[str, float | str]]:
    queries_path = ROOT / "data" / "eval_queries.jsonl"
    if not queries_path.exists():
        return []
    with queries_path.open(encoding="utf-8") as handle:
        queries = [json.loads(line) for line in handle if line.strip()]
    embeddings = encode_documents(_model, documents)
    semantic_recalls: list[float] = []
    semantic_ranks: list[float] = []
    keyword_recalls: list[float] = []
    keyword_ranks: list[float] = []
    for item in queries:
        candidates = filter_documents(documents, item.get("language", "all"))
        if not candidates:
            continue
        indexes = [documents.index(document) for document in candidates]
        query_embedding = _model.encode([f"query: {item['query']}"], normalize_embeddings=True, show_progress_bar=False)[0]
        ranked = rank_embeddings(query_embedding, embeddings[indexes], top_k=min(10, len(candidates)))
        relevant = set(item["relevant_docids"])
        semantic_ids = [candidates[index].docid for index, _ in ranked]
        semantic_recalls.append(recall_at_k(semantic_ids, relevant, 5))
        semantic_ranks.append(reciprocal_rank(semantic_ids, relevant, 10))
        keyword_ids = [candidates[index].docid for index, _ in keyword_rank(item["query"], candidates, top_k=10)]
        keyword_recalls.append(recall_at_k(keyword_ids, relevant, 5))
        keyword_ranks.append(reciprocal_rank(keyword_ids, relevant, 10))
    if not semantic_recalls:
        return []
    return [
        {"Method": "Semantic E5", "Recall@5": sum(semantic_recalls) / len(semantic_recalls), "MRR@10": sum(semantic_ranks) / len(semantic_ranks)},
        {"Method": "Keyword baseline", "Recall@5": sum(keyword_recalls) / len(keyword_recalls), "MRR@10": sum(keyword_ranks) / len(keyword_ranks)},
    ]


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


st.set_page_config(page_title="MIRACL Multilingual Search", layout="wide")
st.title("Multilingual MIRACL Search")
st.caption("Semantic retrieval over MIRACL passages using multilingual-e5-small. Results are ranked evidence, not unsupported generated answers.")

try:
    all_documents = load_documents(CORPUS_PATH)
except (OSError, ValueError) as exc:
    st.error(f"Could not load the corpus: {exc}")
    st.stop()

with st.sidebar:
    st.header("Search settings")
    language_options = {"Query language (automatic)": "auto", "All languages": "all", **{name: code for code, name in SUPPORTED_LANGUAGES.items()}}
    selected_label = st.selectbox("Language", list(language_options))
    selected_language_mode = language_options[selected_label]
    top_k = st.slider("Results to show", min_value=1, max_value=10, value=5)
    search_method = st.selectbox("Search method", ["Semantic (multilingual E5)", "Keyword baseline"])
    st.divider()
    st.markdown("**Corpus**")
    st.write(f"{len(all_documents):,} passages ({CORPUS_PATH.name})")
    st.write("English | Hindi | Spanish | Arabic")
    st.markdown("[Refresh the sample](https://huggingface.co/datasets/miracl/miracl-corpus)")

search_tab, analysis_tab = st.tabs(["Search", "Analysis"])

with search_tab:
    examples = corpus_examples(all_documents)
    selected_example = st.selectbox("Try an example query", ["Choose an example...", *examples])
    st.caption("Examples come from topics present in the loaded corpus.")
    default_query = "" if selected_example == "Choose an example..." else examples[selected_example]
    query = st.text_input("Enter a search query", value=default_query, placeholder="Ask in English, Hindi, Spanish, or Arabic...")
    detected_language = detect_language(query) if query.strip() else "en"
    effective_language = detected_language if selected_language_mode == "auto" else selected_language_mode
    documents = filter_documents(all_documents, effective_language)
    if selected_language_mode == "auto" and query.strip():
        st.caption(f"Detected language: {SUPPORTED_LANGUAGES[detected_language]}. Searching that language only. Choose All languages for cross-language search.")

    if query.strip():
        with st.spinner("Searching..."):
            if search_method == "Semantic (multilingual E5)":
                results = search(load_model(), documents, query, top_k)
            else:
                results = keyword_search(documents, query, top_k)
        st.subheader(f"Top {len(results)} results")
        if not results:
            st.info("No matching documents were found. Try All languages or the larger local corpus.")
        else:
            answer_document, answer_score = results[0]
            with st.expander("Grounded answer candidate", expanded=True):
                st.info(extractive_answer(query, answer_document.text))
                st.caption(f"Source: {answer_document.title} | {answer_document.docid} | score {answer_score:.4f}")
        for rank, (document, score) in enumerate(results, start=1):
            st.markdown(f"### {rank}. {document.title}")
            score_name = "cosine score" if search_method == "Semantic (multilingual E5)" else "TF-IDF score"
            st.caption(f"{SUPPORTED_LANGUAGES[document.language]} | {document.docid} | {score_name} {score:.4f}")
            st.write(document.text)
            st.divider()
    else:
        st.info("Enter a query or choose an example to begin.")

with analysis_tab:
    st.subheader("Corpus analysis")
    unique_articles = len({(document.language, document.title) for document in all_documents})
    metric_columns = st.columns(3)
    metric_columns[0].metric("Passages", f"{len(all_documents):,}")
    metric_columns[1].metric("Articles", f"{unique_articles:,}")
    metric_columns[2].metric("Languages", len({document.language for document in all_documents}))
    language_rows = []
    for code, name in SUPPORTED_LANGUAGES.items():
        language_documents = [document for document in all_documents if document.language == code]
        language_rows.append({"Language": name, "Passages": len(language_documents), "Articles": len({document.title for document in language_documents})})
    st.table(language_rows)

    st.subheader("Evaluation")
    st.write("The demo evaluation compares semantic E5 retrieval with the keyword baseline on the bundled query set.")
    if st.button("Run demo evaluation"):
        evaluation = demo_evaluation(load_model(), tuple(all_documents))
        if evaluation:
            st.table(evaluation)
        else:
            st.info("No evaluation queries are available.")
    else:
        st.info("Click Run demo evaluation to calculate Recall@5 and MRR@10.")

    topics_path = ROOT / "data" / "miracl_dev_topics.jsonl"
    qrels_path = ROOT / "data" / "miracl_dev_qrels.jsonl"
    st.subheader("Official MIRACL evaluation files")
    st.write(f"Development topics: {count_jsonl(topics_path):,}")
    st.write(f"Positive qrels: {count_jsonl(qrels_path):,}")
    st.code("python scripts/download_miracl_dev.py\npython scripts/evaluate_official.py --max-queries 100")

    st.subheader("Runtime")
    st.write(f"Embedding model: {MODEL_NAME}")
    st.write("Inference backend: Sentence Transformers / PyTorch")
    st.write("Ollama: not used by this project")
