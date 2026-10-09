"""Streamlit UI for the multilingual MIRACL semantic search project."""

from pathlib import Path
import os

import streamlit as st
from sentence_transformers import SentenceTransformer

from search_engine import SUPPORTED_LANGUAGES, Document, filter_documents, keyword_rank, load_documents, rank_embeddings


ROOT = Path(__file__).parent
configured_corpus = Path(os.getenv("MIRACL_CORPUS_PATH", "data/sample_corpus.jsonl"))
CORPUS_PATH = configured_corpus if configured_corpus.is_absolute() else ROOT / configured_corpus
MODEL_NAME = "intfloat/multilingual-e5-small"


@st.cache_resource(show_spinner="Loading multilingual embedding model…")
def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@st.cache_data(show_spinner="Encoding the demo corpus…")
def encode_documents(_model: SentenceTransformer, documents: tuple[Document, ...]):
    passages = [f"passage: {document.title}. {document.text}" for document in documents]
    return _model.encode(passages, normalize_embeddings=True, show_progress_bar=False)


def search(model: SentenceTransformer, documents: list[Document], query: str, top_k: int):
    if not query.strip():
        return []
    embeddings = encode_documents(model, tuple(documents))
    query_embedding = model.encode([f"query: {query.strip()}"], normalize_embeddings=True, show_progress_bar=False)[0]
    return [(documents[index], score) for index, score in rank_embeddings(query_embedding, embeddings, top_k)]


def keyword_search(documents: list[Document], query: str, top_k: int):
    if not query.strip():
        return []
    return [(documents[index], score) for index, score in keyword_rank(query, documents, top_k)]


def corpus_examples(documents: list[Document]) -> dict[str, str]:
    """Use topics that actually exist in the selected corpus profile."""
    examples: dict[str, str] = {}
    for code, name in SUPPORTED_LANGUAGES.items():
        for document in documents:
            if document.language == code:
                examples[f"{name}: {document.title}"] = document.title
                break
    return examples


st.set_page_config(page_title="MIRACL Multilingual Search", page_icon="🌍", layout="wide")
st.title("🌍 Multilingual MIRACL Search")
st.caption("Semantic search over a MIRACL-derived corpus using multilingual-e5-small. Results are ranked passages, not generated answers.")

try:
    all_documents = load_documents(CORPUS_PATH)
except (OSError, ValueError) as exc:
    st.error(f"Could not load the sample corpus: {exc}")
    st.stop()

with st.sidebar:
    st.header("Search settings")
    language_options = {"All languages": "all", **{name: code for code, name in SUPPORTED_LANGUAGES.items()}}
    selected_label = st.selectbox("Language", list(language_options))
    selected_language = language_options[selected_label]
    top_k = st.slider("Results to show", min_value=1, max_value=10, value=5)
    search_method = st.selectbox("Search method", ["Semantic (multilingual E5)", "Keyword baseline"])
    st.divider()
    st.markdown("**Corpus**")
    st.write(f"{len(all_documents):,} passages ({CORPUS_PATH.name})")
    st.write("English · Hindi · Spanish · Arabic")
    st.markdown("[Refresh the sample](https://huggingface.co/datasets/miracl/miracl-corpus)")

documents = filter_documents(all_documents, selected_language)
examples = corpus_examples(all_documents)
selected_example = st.selectbox("Try an example query", ["Choose an example…", *examples])
st.caption("Examples are taken from topics present in the loaded corpus, so they always have a matching document.")
default_query = "" if selected_example == "Choose an example…" else examples[selected_example]
query = st.text_input("Enter a search query", value=default_query, placeholder="Ask in any supported language…")

if query.strip():
    with st.spinner("Searching…"):
        if search_method == "Semantic (multilingual E5)":
            results = search(load_model(), documents, query, top_k)
        else:
            results = keyword_search(documents, query, top_k)
    st.subheader(f"Top {len(results)} results")
    if not results:
        st.info("No matching documents were found.")
    for rank, (document, score) in enumerate(results, start=1):
        st.markdown(f"### {rank}. {document.title}")
        score_name = "cosine score" if search_method == "Semantic (multilingual E5)" else "TF-IDF score"
        st.caption(f"{SUPPORTED_LANGUAGES[document.language]} · {document.docid} · {score_name} {score:.4f}")
        st.write(document.text)
        st.divider()
else:
    st.info("Enter a query or choose an example to begin.")
