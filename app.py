"""Streamlit interface for multilingual search over MIRACL."""

from html import escape
import json
import os
from pathlib import Path
import re
import time

import streamlit as st
from sentence_transformers import CrossEncoder, SentenceTransformer

from search_engine import (
    SUPPORTED_LANGUAGES,
    Document,
    detect_language,
    diversify_ranked,
    extractive_answer,
    filter_documents,
    hybrid_rank,
    keyword_rank,
    load_documents,
    rank_embeddings,
)
from vector_index import LocalIndex


ROOT = Path(__file__).parent
MODEL_NAME = "intfloat/multilingual-e5-small"
RERANKER_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CORPUS_PATH = Path(os.getenv("MIRACL_CORPUS_PATH", ROOT / "data" / "sample_corpus.jsonl"))
if not CORPUS_PATH.is_absolute():
    CORPUS_PATH = ROOT / CORPUS_PATH
INDEX_PATH = Path(os.getenv("MIRACL_INDEX_PATH", ROOT / "data" / "miracl_index"))
if not INDEX_PATH.is_absolute():
    INDEX_PATH = ROOT / INDEX_PATH


@st.cache_resource(show_spinner="Loading multilingual embedding model...")
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner="Loading local MIRACL index...")
def load_index(path: str):
    return LocalIndex(path)


@st.cache_resource(show_spinner="Loading multilingual reranker...")
def load_reranker():
    return CrossEncoder(RERANKER_NAME)


@st.cache_data(show_spinner="Encoding corpus passages...")
def encode_documents(_model, documents: tuple[Document, ...]):
    return _model.encode(
        [f"passage: {doc.title}. {doc.text}" for doc in documents],
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def diversified(results: list[tuple[Document, float]], top_k: int):
    ranked = [(i, score) for i, (_, score) in enumerate(results)]
    chosen = diversify_ranked(ranked, [doc for doc, _ in results], min(top_k, len(results)))
    return [(results[i][0], score) for i, score in chosen]


def fuse(semantic, lexical, limit: int):
    documents = []
    positions = {}
    rankings = []
    for ranking in (semantic, lexical):
        ranked = []
        for doc, score in ranking:
            key = (doc.language, doc.docid)
            if key not in positions:
                positions[key] = len(documents)
                documents.append(doc)
            ranked.append((positions[key], score))
        rankings.append(ranked)
    return [(documents[i], score) for i, score in hybrid_rank(*rankings, top_k=limit)]


def retrieve(model, documents, index, query: str, language: str, method: str, top_k: int):
    if not query.strip():
        return []
    pool_size = max(30, top_k * 6)
    semantic = []
    if method != "Keyword baseline":
        query_vector = model.encode([f"query: {query.strip()}"], normalize_embeddings=True, show_progress_bar=False)[0]
        if index:
            semantic = index.vector_search(query_vector, language, pool_size)
        else:
            candidates = filter_documents(documents, language)
            vectors = encode_documents(model, tuple(candidates))
            ranked = rank_embeddings(query_vector, vectors, min(pool_size, len(candidates)))
            semantic = [(candidates[i], score) for i, score in ranked]
    lexical = []
    if method in ("Keyword baseline", "Hybrid (semantic + keyword)", "Hybrid + reranker"):
        if index:
            lexical = [(index.document_by_global_index(i), score) for i, score in index.keyword_search(query, language, pool_size)]
        else:
            candidates = filter_documents(documents, language)
            lexical = [(candidates[i], score) for i, score in keyword_rank(query, candidates, pool_size)]
    if method == "Keyword baseline":
        results = lexical
    elif method == "Semantic (multilingual E5)":
        results = semantic
    else:
        results = fuse(semantic, lexical, pool_size)
    if method == "Hybrid + reranker" and results:
        reranker = load_reranker()
        scores = reranker.predict([(query, f"{doc.title}. {doc.text}") for doc, _ in results], show_progress_bar=False)
        results = sorted(((doc, float(score)) for (doc, _), score in zip(results, scores)), key=lambda item: -item[1])
    return diversified(results, top_k)


def corpus_examples(documents: list[Document]):
    examples = {}
    for doc in documents:
        examples.setdefault(f"{SUPPORTED_LANGUAGES[doc.language]}: {doc.title}", doc.title)
    return examples


def highlight(text: str, query: str) -> str:
    terms = list(dict.fromkeys(re.findall(r"\w+", query, flags=re.UNICODE)))
    escaped = escape(text)
    if not terms:
        return escaped
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True)) + r")(?!\w)", re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", escaped)


def render_results(results, query: str, method: str):
    if not results:
        st.info("No matching passages found. Try All languages or a larger corpus.")
        return
    doc, score = results[0]
    with st.expander("Evidence based answer excerpt", expanded=True):
        st.markdown(highlight(extractive_answer(query, doc.text), query), unsafe_allow_html=True)
        st.caption(f"Source: {doc.title} · {doc.language.upper()} · {doc.docid} · score {score:.4f}")
    for rank, (doc, score) in enumerate(results, 1):
        st.markdown(f"#### {rank}. {doc.title}")
        st.caption(f"{SUPPORTED_LANGUAGES[doc.language]} · {doc.docid} · score {score:.4f}")
        st.markdown(highlight(doc.text, query), unsafe_allow_html=True)


def read_snapshot():
    path = ROOT / "data" / "evaluation_snapshot.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"profiles": {}}


def jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


st.set_page_config(page_title="MIRACL Multilingual Search", layout="wide")
st.title("Multilingual MIRACL Search")
st.caption("Search MIRACL passages using multilingual E5, hybrid retrieval, and an optional multilingual reranker.")

use_index = bool(str(INDEX_PATH)) and (INDEX_PATH / "manifest.json").exists()
index = load_index(str(INDEX_PATH)) if use_index else None
try:
    all_documents = [] if index else load_documents(CORPUS_PATH)
except (OSError, ValueError) as exc:
    st.error(f"Could not load corpus: {exc}")
    st.stop()

available = index.languages if index else {
    code: sum(doc.language == code for doc in all_documents)
    for code in dict.fromkeys(doc.language for doc in all_documents)
}
language_options = {"Query language (automatic)": "auto", "All languages": "all"}
language_options.update({SUPPORTED_LANGUAGES[code]: code for code in available})
corpus_name = f"Indexed MIRACL ({index.count:,} passages)" if index else f"{len(all_documents):,} passages ({CORPUS_PATH.name})"

with st.sidebar:
    st.header("Search settings")
    language_mode = language_options[st.selectbox("Language", list(language_options))]
    top_k = st.slider("Results to show", 1, 10, 5)
    method = st.selectbox("Search method", ["Semantic (multilingual E5)", "Hybrid (semantic + keyword)", "Hybrid + reranker", "Keyword baseline"])
    st.divider()
    st.markdown("**Active corpus**")
    st.write(corpus_name)
    st.caption("The full indexed corpus stays on this computer; Streamlit Cloud uses its small sample.")

search_tab, compare_tab, analysis_tab = st.tabs(["Search", "Compare methods", "Analysis"])

with search_tab:
    examples = corpus_examples(all_documents) if all_documents else {}
    example = st.selectbox("Try a corpus title", ["Choose an example...", *examples])
    query = st.text_input("Search in your language", value=examples.get(example, ""), key="query_input", placeholder="Ask in English, Hindi, Spanish, or Arabic...")
    detected = detect_language(query) if query.strip() else "en"
    active_language = detected if language_mode == "auto" else language_mode
    if language_mode == "auto" and query.strip():
        if active_language not in available:
            active_language = "all"
            st.caption(f"Detected: {SUPPORTED_LANGUAGES.get(detected, detected)}, which is not in this corpus. Searching all available languages.")
        else:
            st.caption(f"Detected: {SUPPORTED_LANGUAGES.get(detected, detected)} · searching that language. Select All languages for cross-language results.")
    if query.strip():
        started = time.perf_counter()
        results = retrieve(load_model(), all_documents, index, query, active_language, method, top_k)
        elapsed = (time.perf_counter() - started) * 1000
        st.caption(f"{len(results)} results · {elapsed:.0f} ms · {method}")
        render_results(results, query, method)
    else:
        st.info("Enter a query or choose an example to begin.")

with compare_tab:
    compare_query = st.text_input("Query to compare", value=st.session_state.get("query_input", ""), key="compare_query")
    compare_language = st.selectbox("Compare within", ["Automatic", "All languages", *[SUPPORTED_LANGUAGES[c] for c in available]], key="compare_language")
    compare_code = detect_language(compare_query) if compare_language == "Automatic" else ("all" if compare_language == "All languages" else next(code for code in available if SUPPORTED_LANGUAGES[code] == compare_language))
    if compare_code not in available:
        compare_code = "all"
    if compare_query.strip() and st.button("Compare all methods"):
        pool_size = max(30, top_k * 6)
        model = load_model()
        vector = model.encode([f"query: {compare_query.strip()}"], normalize_embeddings=True, show_progress_bar=False)[0]
        candidates = []
        passage_vectors = None
        if not index:
            candidates = filter_documents(all_documents, compare_code)
            passage_vectors = encode_documents(model, tuple(candidates))
        started = time.perf_counter()
        if index:
            semantic = index.vector_search(vector, compare_code, pool_size)
        else:
            semantic = [(candidates[i], score) for i, score in rank_embeddings(vector, passage_vectors, min(pool_size, len(candidates)))]
        semantic_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        if index:
            lexical = [(index.document_by_global_index(i), score) for i, score in index.keyword_search(compare_query, compare_code, pool_size)]
        else:
            candidates = filter_documents(all_documents, compare_code)
            lexical = [(candidates[i], score) for i, score in keyword_rank(compare_query, candidates, pool_size)]
        lexical_ms = (time.perf_counter() - started) * 1000
        hybrid = fuse(semantic, lexical, pool_size)
        reranker = load_reranker()
        started = time.perf_counter()
        reranked_scores = reranker.predict([(compare_query, f"{doc.title}. {doc.text}") for doc, _ in hybrid], show_progress_bar=False)
        reranked = sorted(((doc, float(score)) for (doc, _), score in zip(hybrid, reranked_scores)), key=lambda row: -row[1])
        reranker_ms = (time.perf_counter() - started) * 1000
        comparison = [
            ("Semantic (multilingual E5)", diversified(semantic, top_k), semantic_ms),
            ("Hybrid (semantic + keyword)", diversified(hybrid, top_k), semantic_ms + lexical_ms),
            ("Hybrid + reranker", diversified(reranked, top_k), semantic_ms + lexical_ms + reranker_ms),
            ("Keyword baseline", diversified(lexical, top_k), lexical_ms),
        ]
        st.session_state["method_comparison"] = (compare_query, comparison)
    saved_comparison = st.session_state.get("method_comparison")
    if saved_comparison and saved_comparison[0] == compare_query:
        comparison = saved_comparison[1]
        st.caption("Timings exclude the shared query encoding and first model download/startup.")
        st.dataframe([{"Method": name, "Time (ms)": round(ms), "Top result": rows[0][0].title if rows else "No result", "Score": round(rows[0][1], 4) if rows else 0} for name, rows, ms in comparison], width="stretch", hide_index=True)
        for name, rows, ms in comparison:
            with st.expander(f"{name} · {ms:.0f} ms"):
                render_results(rows[:3], compare_query, name)

with analysis_tab:
    st.subheader("Corpus overview")
    total = index.count if index else len(all_documents)
    metrics = st.columns(3)
    metrics[0].metric("Passages", f"{total:,}")
    metrics[1].metric("Storage", "SQLite + mmap" if index else "JSONL sample")
    metrics[2].metric("Languages", len(available))
    language_rows = [{"Language": SUPPORTED_LANGUAGES[code], "Passages": count} for code, count in available.items()]
    st.bar_chart(language_rows, x="Language", y="Passages", horizontal=True)

    st.subheader("Search method comparison")
    st.write("Compare semantic, hybrid, reranked, and keyword results for the same query in the Compare methods tab.")
    profile = "local" if index or CORPUS_PATH.name == "local_corpus.jsonl" else "cloud"
    saved_rows = read_snapshot().get("profiles", {}).get(profile, [])
    if saved_rows:
        st.dataframe(saved_rows, width="stretch", hide_index=True)
        st.bar_chart(saved_rows, x="Method", y=["Recall@5", "MRR@10"])
        st.caption("Saved metrics use four demonstration queries. For a full-corpus index, these figures describe the 4,000-passage local sample, not the full index.")
    else:
        st.info("No saved evaluation is available for this corpus profile.")
    topics_path = ROOT / "data" / "miracl_dev_topics.jsonl"
    qrels_path = ROOT / "data" / "miracl_dev_qrels.jsonl"
    eval_columns = st.columns(2)
    eval_columns[0].metric("Official dev topics downloaded", f"{jsonl_count(topics_path):,}")
    eval_columns[1].metric("Positive relevance judgments", f"{jsonl_count(qrels_path):,}")
    st.caption("Download official files with `python scripts/download_miracl_dev.py`; evaluation commands are in README.")
    st.subheader("Index and runtime")
    st.write(f"Active corpus: `{INDEX_PATH if index else CORPUS_PATH}`")
    st.write(f"Embedding model: `{MODEL_NAME}` · NumPy exact similarity search")
    st.write(f"Reranker (loads only when selected): `{RERANKER_NAME}`")
    st.write("Local full index includes disk backed embeddings and SQLite full text search. Ollama is not used.")
