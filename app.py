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
LOCAL_SAMPLE_PATH = ROOT / "data" / "local_corpus.jsonl"
DEMO_SAMPLE_PATH = ROOT / "data" / "sample_corpus.jsonl"
DEFAULT_CORPUS_PATH = LOCAL_SAMPLE_PATH if LOCAL_SAMPLE_PATH.exists() else DEMO_SAMPLE_PATH
CORPUS_PATH = Path(os.getenv("MIRACL_CORPUS_PATH", DEFAULT_CORPUS_PATH))
if not CORPUS_PATH.is_absolute():
    CORPUS_PATH = ROOT / CORPUS_PATH
INDEX_PATH = Path(os.getenv("MIRACL_INDEX_PATH", ROOT / "data" / "miracl_index"))
if not INDEX_PATH.is_absolute():
    INDEX_PATH = ROOT / INDEX_PATH


@st.cache_resource(show_spinner="Loading multilingual embedding model...")
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner="Loading local MIRACL index...")
def load_index(path: str, manifest_modified_ns: int):
    del manifest_modified_ns  # Reload the cached index whenever a resumable build advances.
    return LocalIndex(path)


@st.cache_data(show_spinner="Loading MIRACL passages...")
def load_corpus(path: str, modified_ns: int):
    del modified_ns  # The modification time is a cache key for refreshed corpus files.
    return load_documents(Path(path))


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
    for rank, (doc, score) in enumerate(results, 1):
        title = escape(doc.title)
        language = escape(SUPPORTED_LANGUAGES.get(doc.language, doc.language))
        docid = escape(doc.docid)
        excerpt = highlight(doc.text, query)
        st.markdown(
            f"""<article class="result-card">
                <div class="result-topline"><span class="result-rank">{rank:02}</span>
                <span class="result-language">{language}</span>
                <span class="result-score">{score:.4f}</span></div>
                <h3>{title}</h3>
                <p class="result-passage">{excerpt}</p>
                <div class="result-source">MIRACL · {docid}</div>
            </article>""",
            unsafe_allow_html=True,
        )


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
st.markdown(
    """<style>
    :root { color-scheme:light; --ink:#182b3a; --muted:#526371; --line:#d6e0e5; --accent:#a94325; --paper:#fff; }
    html, body, [class], [data-testid="stAppViewContainer"] { color:var(--ink); }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background:#f3f6f8 !important; color:var(--ink) !important; }
    [data-testid="stHeader"] { background:rgba(243,246,248,.96); }
    .block-container { max-width:1280px; padding-top:2.1rem; padding-bottom:3.5rem; }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div { background:#eaf0f3 !important; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color:#203746 !important; }
    [data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"] { color:#405361; }
    h1,h2,h3,h4, label { color:var(--ink) !important; letter-spacing:-.015em; }
    h1 { font-size:2.35rem !important; font-weight:750 !important; }
    [data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.45rem; border-bottom:1px solid var(--line); }
    [data-testid="stTabs"] [data-baseweb="tab"] { padding:.8rem 1.05rem; color:#405361 !important; }
    [data-testid="stTabs"] [aria-selected="true"] { color:#8f351b !important; }
    [data-testid="stMetric"] { background:var(--paper) !important; border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; box-shadow:0 3px 14px #18324808; }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] { color:#203746 !important; }
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stNumberInput"] input, [data-baseweb="select"] > div { color:#182b3a !important; background:#fff !important; border-color:#aabac4 !important; }
    [data-testid="stTextInput"] input::placeholder, [data-testid="stTextArea"] textarea::placeholder { color:#667984 !important; opacity:1; }
    [role="listbox"], [role="option"] { color:#182b3a !important; background:#fff !important; }
    div.stButton > button { border-radius:10px; font-weight:650; color:#183243 !important; background:#fff !important; border-color:#aabac4 !important; }
    div.stButton > button[kind="primary"] { color:#fff !important; background:#9b4024 !important; border-color:#9b4024 !important; }
    [data-testid="stAlert"] { color:#203746 !important; background:#fff !important; }
    [data-testid="stExpander"] { background:#fff; border-color:var(--line); }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary span { color:#203746 !important; }
    [data-testid="stDataFrame"], [data-testid="stVegaLiteChart"] { background:var(--paper); border:1px solid var(--line); border-radius:14px; overflow:hidden; }
    .hero { background:linear-gradient(120deg,#17364a 0%,#245467 58%,#33756f 100%); color:white; padding:2rem 2.1rem; border-radius:20px; margin:.2rem 0 1.45rem; box-shadow:0 14px 32px #15384a20; }
    .hero-kicker { color:#d1e5e8 !important; font-size:.72rem; font-weight:750; letter-spacing:.13em; text-transform:uppercase; }
    .hero h1 { color:#fff !important; margin:.55rem 0 .35rem; font-size:2.15rem !important; }
    .hero p { color:#e4eef0 !important; margin:0; max-width:720px; font-size:1.02rem; }
    .hero-pill { display:inline-block; margin-top:1.1rem; padding:.35rem .68rem; border:1px solid #ffffff65; border-radius:99px; color:#fff !important; font-size:.78rem; }
    .result-card { background:var(--paper); border:1px solid var(--line); border-radius:15px; padding:1.15rem 1.3rem; margin:.65rem 0; box-shadow:0 3px 12px #17344708; }
    .result-topline { display:flex; align-items:center; gap:.6rem; margin-bottom:.45rem; }
    .result-rank { color:#b44f31; font-weight:800; font-size:.78rem; letter-spacing:.06em; }
    .result-language { color:#35625e; background:#e9f3ef; padding:.18rem .5rem; border-radius:99px; font-size:.72rem; font-weight:700; }
    .result-score { margin-left:auto; color:#455763; font:600 .75rem ui-monospace,SFMono-Regular,Consolas,monospace; }
    .result-card h3 { color:#182b3a !important; font-size:1.08rem; margin:.1rem 0 .55rem; }
    .result-passage { color:#293e4b !important; font-size:.96rem; line-height:1.75; margin:.25rem 0 .65rem; }
    .result-source { border-top:1px solid #e2e9ed; padding-top:.55rem; color:#526371 !important; font-size:.78rem; }
    mark { background:#fff0bd; color:#533b15; border-radius:3px; padding:0 .08rem; }
    @media (max-width:700px) { .block-container { padding:1rem 1rem 2rem; } .hero { padding:1.4rem; border-radius:15px; } .hero h1 { font-size:1.65rem !important; } }
    </style>""",
    unsafe_allow_html=True,
)

manifest_path = INDEX_PATH / "manifest.json"
use_index = bool(os.getenv("MIRACL_INDEX_PATH")) and manifest_path.exists()
index = load_index(str(INDEX_PATH), manifest_path.stat().st_mtime_ns) if use_index else None
try:
    all_documents = [] if index else load_corpus(str(CORPUS_PATH), CORPUS_PATH.stat().st_mtime_ns)
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
st.markdown(
    f"""<section class="hero">
        <div class="hero-kicker">Multilingual information retrieval · MIRACL</div>
        <h1>Find useful passages, across languages.</h1>
        <p>Search English, Hindi, Spanish, and Arabic with semantic, keyword, or hybrid retrieval—then inspect the source passages behind every result.</p>
        <span class="hero-pill">{escape(corpus_name)}</span>
    </section>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Search settings")
    st.caption("Tune how MIRACL ranks and filters passages.")
    language_mode = language_options[st.selectbox("Language", list(language_options))]
    top_k = st.slider("Results to show", 1, 10, 5)
    method = st.selectbox("Search method", ["Semantic (multilingual E5)", "Hybrid (semantic + keyword)", "Hybrid + reranker", "Keyword baseline"])
    st.divider()
    st.markdown("**Active corpus**")
    st.write(corpus_name)
    st.caption("This computer uses the 4,000-passage local corpus; Streamlit Cloud uses the bundled demo sample.")
search_tab, compare_tab, analysis_tab = st.tabs(["Search", "Compare methods", "Analysis"])

with search_tab:
    examples = {
        "English · What is anarchism?": "What is anarchism?",
        "हिन्दी · गणेश के नाम और पूजा की विधि क्या है?": "गणेश के नाम और पूजा की विधि क्या है?",
        "Español · ¿Cuál es el idioma oficial de Andorra?": "¿Cuál es el idioma oficial de Andorra?",
        "العربية · ما نسبة سطح الأرض التي يغطيها الماء؟": "ما نسبة سطح الأرض التي يغطيها الماء؟",
    }
    example = st.selectbox("Try a sample query", ["Choose an example...", *examples])
    if example in examples and st.session_state.get("_selected_example") != example:
        st.session_state["query_input"] = examples[example]
        st.session_state["_selected_example"] = example
    elif example == "Choose an example...":
        st.session_state["_selected_example"] = None
    query = st.text_input("Search in your language", key="query_input", placeholder="Ask in English, Hindi, Spanish, or Arabic...")
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
