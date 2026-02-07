from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

import streamlit as st
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from similarity_engine import compare_documents, load_document_from_bytes


DEFAULT_MODEL = os.getenv("MODEL_NAME", "tf-idf")
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "10"))
DEFAULT_MAX_SENTENCES = int(os.getenv("MAX_SENTENCES", "200"))
DEFAULT_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
BING_API_KEY = os.getenv("BING_API_KEY", "")
BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"


def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@st.cache_data(show_spinner=False)
def load_text_cached(filename: str, content_hash: str, content: bytes) -> str:
    return load_document_from_bytes(filename, content)


st.set_page_config(
    page_title="Plagiarify",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stToolbar"] {
        display: none;
    }
    a.stHeadingAnchor {
        display: none;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #4b4b4b;
        padding: 0;
        border-radius: 8px;
        background: #0f1117;
        width: 220px;
        max-width: 100%;
        aspect-ratio: 1 / 1;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]::before {
        content: "+";
        display: block;
        font-size: 64px;
        line-height: 64px;
        text-align: center;
        color: #8a8a8a;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] div,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] button {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Plagiarify")

st.subheader("📤 Upload Files")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Primary File (A)**")
    file_a = st.file_uploader(
        "Upload file A",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=False,
        help="Upload the primary file to compare against others",
        label_visibility="collapsed",
        key="file_a",
    )
with col_b:
    st.markdown("**Comparison Files (B)**")
    file_b = st.file_uploader(
        "Upload file(s) B",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        help="Upload one or more files to compare against A",
        label_visibility="collapsed",
        key="file_b",
    )

comparison_files = file_b if isinstance(file_b, list) else ([] if file_b is None else [file_b])
uploaded_files = [f for f in [file_a, *comparison_files] if f]

if file_a and comparison_files:
    st.success(f"✅ {1 + len(comparison_files)} file(s) uploaded successfully!")
    with st.expander("📋 View uploaded files"):
        st.write(f"- {file_a.name}")
        for f in comparison_files:
            st.write(f"- {f.name}")

model_name = DEFAULT_MODEL
max_sentences = DEFAULT_MAX_SENTENCES
batch_size = DEFAULT_BATCH_SIZE


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _tokenize_words(text: str) -> List[str]:
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9']+", text)]
    return [w for w in words if w not in ENGLISH_STOP_WORDS]


def _doc_stats(text: str) -> Dict[str, int]:
    words = _tokenize_words(text)
    sentences = _split_sentences(text)
    return {
        "word_count": len(words),
        "unique_words": len(set(words)),
        "sentence_count": len(sentences),
        "char_count": len(text),
    }


def _build_search_queries(text: str, max_queries: int = 3) -> List[str]:
    sentences = _split_sentences(text)
    candidates = [s for s in sentences if 8 <= len(s.split()) <= 20]
    candidates = sorted(candidates, key=lambda s: len(s), reverse=True)
    queries = []
    for s in candidates:
        cleaned = re.sub(r"\s+", " ", s).strip()
        if cleaned and cleaned not in queries:
            queries.append(cleaned)
        if len(queries) >= max_queries:
            break
    return queries


def _bing_search(
    query: str,
    api_key: str,
    count: int = 5,
    market: str = "en-US",
    safe_search: str = "Moderate",
) -> List[Dict[str, str]]:
    if not api_key:
        return []
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {
        "q": query,
        "count": count,
        "mkt": market,
        "safeSearch": safe_search,
    }
    try:
        resp = requests.get(BING_ENDPOINT, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return []

    results = []
    for item in data.get("webPages", {}).get("value", []):
        results.append({
            "name": item.get("name", ""),
            "url": item.get("url", ""),
            "snippet": item.get("snippet", ""),
        })
    return results


def _fetch_page_text(url: str, max_chars: int = 12000) -> str:
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except requests.RequestException:
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return text[:max_chars]


def _web_similarity_score(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf = vectorizer.fit_transform([text_a, text_b])
    return float(cosine_similarity(tfidf)[0, 1])


def _extract_phrases(text: str, top_n: int = 40) -> List[str]:
    tokens = _tokenize_words(text)
    if len(tokens) < 2:
        return []
    bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
    trigrams = [
        f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}"
        for i in range(len(tokens) - 2)
    ]
    phrases = bigrams + trigrams
    counts = Counter(phrases)
    return [p for p, _ in counts.most_common(top_n)]


def _semantic_phrase_matches(
    text_a: str,
    text_b: str,
    top_n: int = 8,
    min_score: float = 0.45,
) -> List[Dict[str, str]]:
    phrases_a = _extract_phrases(text_a)
    phrases_b = _extract_phrases(text_b)
    if not phrases_a or not phrases_b:
        return []

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    vectorizer.fit(phrases_a + phrases_b)
    mat_a = vectorizer.transform(phrases_a)
    mat_b = vectorizer.transform(phrases_b)
    sims = cosine_similarity(mat_a, mat_b)

    matches = []
    for i in range(sims.shape[0]):
        j = int(np.argmax(sims[i]))
        score = float(sims[i, j])
        if score >= min_score:
            matches.append({
                "phrase_a": phrases_a[i],
                "phrase_b": phrases_b[j],
                "score": score,
            })

    matches = sorted(matches, key=lambda m: m["score"], reverse=True)
    return matches[:top_n]


def _shared_terms(text_a: str, text_b: str, top_n: int = 12) -> List[Dict[str, float]]:
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([text_a, text_b]).toarray()
    shared = np.minimum(tfidf[0], tfidf[1])
    if shared.sum() == 0:
        return []
    top_idx = np.argsort(shared)[::-1][:top_n]
    terms = vectorizer.get_feature_names_out()
    return [
        {"term": terms[i], "score": float(shared[i])}
        for i in top_idx
        if shared[i] > 0
    ]


def _top_sentence_matches(
    text_a: str,
    text_b: str,
    top_n: int = 6,
    min_score: float = 0.2,
) -> List[Dict[str, str]]:
    sentences_a = _split_sentences(text_a)
    sentences_b = _split_sentences(text_b)
    if not sentences_a or not sentences_b:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(sentences_a + sentences_b)
    mat_a = vectorizer.transform(sentences_a)
    mat_b = vectorizer.transform(sentences_b)
    sims = cosine_similarity(mat_a, mat_b)

    matches = []
    for i in range(sims.shape[0]):
        j = int(np.argmax(sims[i]))
        score = float(sims[i, j])
        matches.append({
            "sentence_a": sentences_a[i],
            "sentence_b": sentences_b[j],
            "score": score,
        })

    matches = [m for m in matches if m["score"] >= min_score]
    matches = sorted(matches, key=lambda m: m["score"], reverse=True)
    return matches[:top_n]

st.divider()

if st.button(
    "🚀 Compare Documents",
    disabled=not (file_a and comparison_files),
    type="primary",
):
    if not file_a or not comparison_files:
        st.warning("⚠️ Please upload one primary file and at least one comparison file.")
        st.stop()

    documents: List[Tuple[str, str]] = []
    oversize_files = []
    for f in uploaded_files:
        content = f.read()
        if len(content) > MAX_FILE_MB * 1024 * 1024:
            oversize_files.append(f.name)
            continue
        content_hash = _hash_bytes(content)
        text = load_text_cached(f.name, content_hash, content)
        documents.append((f.name, text))

    if oversize_files:
        st.warning(
            "⚠️ Skipped oversized files: "
            + ", ".join(oversize_files)
            + f" (limit: {MAX_FILE_MB} MB)"
        )
        st.stop()

    with st.spinner("Computing similarity..."):
        results = compare_documents(
            documents,
            model_name=model_name,
            threshold=0.7,
            max_pairs=0,
        )

    if not results:
        st.warning("⚠️ Please upload one primary file and at least one comparison file.")
    else:
        primary_name = file_a.name
        primary_text = documents[0][1]
        pairs = [r for r in results if r.doc_a == primary_name or r.doc_b == primary_name]

        st.subheader("Similarity Scores (A vs others)")
        score_rows = []
        for r in pairs:
            other = r.doc_b if r.doc_a == primary_name else r.doc_a
            score_rows.append({
                "Document": other,
                "Similarity %": f"{r.similarity * 100:.2f}%",
            })
        st.dataframe(score_rows)

        if not pairs:
            st.stop()

        st.subheader("Web Source Check")
        check_web = st.checkbox("Check if primary file appears on the web")
        api_key = BING_API_KEY
        if not api_key:
            api_key = st.text_input("Bing API key", type="password", help="Set BING_API_KEY in Railway for production")

        if check_web:
            if not api_key:
                st.warning("Please provide a Bing API key to run web checks.")
            else:
                market = st.selectbox("Market", ["en-US", "en-GB", "en-IN"], index=0)
                safe_search = st.selectbox("SafeSearch", ["Off", "Moderate", "Strict"], index=1)
                queries = _build_search_queries(primary_text)
                if not queries:
                    st.info("Not enough content to form search queries.")
                else:
                    with st.spinner("Searching the web..."):
                        results_set = []
                        seen_urls = set()
                        for q in queries:
                            for item in _bing_search(q, api_key, market=market, safe_search=safe_search):
                                url = item.get("url", "")
                                if url and url not in seen_urls:
                                    seen_urls.add(url)
                                    results_set.append(item)

                        scored = []
                        for item in results_set[:8]:
                            page_text = _fetch_page_text(item.get("url", ""))
                            score = _web_similarity_score(primary_text[:20000], page_text)
                            scored.append({
                                "URL": item.get("url", ""),
                                "Title": item.get("name", ""),
                                "Similarity %": f"{score * 100:.2f}%",
                            })

                        if scored:
                            scored = sorted(scored, key=lambda r: float(r["Similarity %"][:-1]), reverse=True)
                            st.dataframe(scored)
                        else:
                            st.info("No web matches found.")

        selected_docs = st.multiselect(
            "View detailed reports for",
            [row["Document"] for row in score_rows],
            default=[row["Document"] for row in score_rows],
        )

        for selected_doc in selected_docs:
            selected_pair = next(
                r for r in pairs if (r.doc_b == selected_doc or r.doc_a == selected_doc)
            )
            text_a = primary_text
            other_index = [name for name, _ in documents].index(selected_doc)
            text_b = documents[other_index][1]
            stats_a = _doc_stats(text_a)
            stats_b = _doc_stats(text_b)
            words_a = set(_tokenize_words(text_a))
            words_b = set(_tokenize_words(text_b))
            shared_words = words_a & words_b
            unique_a = words_a - words_b
            unique_b = words_b - words_a

            st.subheader(f"Detailed Report: {selected_doc}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Words (A)", f"{stats_a['word_count']}")
            c2.metric("Words (B)", f"{stats_b['word_count']}")
            c3.metric("Shared Words", f"{len(shared_words)}")
            c4.metric("Sentences (A/B)", f"{stats_a['sentence_count']}/{stats_b['sentence_count']}")

            st.markdown("**Document Length Comparison**")
            length_data = [
                {"doc": "A", "label": "Words", "value": stats_a["word_count"]},
                {"doc": "B", "label": "Words", "value": stats_b["word_count"]},
                {"doc": "A", "label": "Characters", "value": stats_a["char_count"]},
                {"doc": "B", "label": "Characters", "value": stats_b["char_count"]},
                {"doc": "A", "label": "Sentences", "value": stats_a["sentence_count"]},
                {"doc": "B", "label": "Sentences", "value": stats_b["sentence_count"]},
            ]
            st.vega_lite_chart(
                {
                    "data": {"values": length_data},
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": "label", "type": "nominal"},
                        "y": {"field": "value", "type": "quantitative"},
                        "color": {"field": "doc", "type": "nominal"},
                    },
                },
            )

            st.markdown("**Shared vs Unique Vocabulary**")
            vocab_data = [
                {"group": "Shared", "value": len(shared_words)},
                {"group": "Unique A", "value": len(unique_a)},
                {"group": "Unique B", "value": len(unique_b)},
            ]
            st.vega_lite_chart(
                {
                    "data": {"values": vocab_data},
                    "mark": {"type": "arc", "innerRadius": 40},
                    "encoding": {
                        "theta": {"field": "value", "type": "quantitative"},
                        "color": {"field": "group", "type": "nominal"},
                    },
                },
            )

            st.markdown("**Top Shared Terms (TF-IDF overlap)**")
            shared_terms = _shared_terms(text_a, text_b)
            if shared_terms:
                st.vega_lite_chart(
                    {
                        "data": {"values": shared_terms},
                        "mark": "bar",
                        "encoding": {
                            "x": {"field": "term", "type": "nominal", "sort": "-y"},
                            "y": {"field": "score", "type": "quantitative"},
                        },
                    },
                )
            else:
                st.info("No shared TF-IDF terms found.")

            st.markdown("**Most Similar Sentences**")
            matches = _top_sentence_matches(text_a, text_b)
            if matches:
                st.dataframe(
                    [
                        {
                            "Score": f"{m['score'] * 100:.2f}%",
                            "Sentence A": m["sentence_a"],
                            "Sentence B": m["sentence_b"],
                        }
                        for m in matches
                    ],
                )
            else:
                st.info("No sentence matches found.")

            st.markdown("**Similar Concepts Matched**")
            phrase_matches = _semantic_phrase_matches(text_a, text_b)
            if phrase_matches:
                st.dataframe(
                    [
                        {
                            "Score": f"{m['score'] * 100:.2f}%",
                            "Phrase A": m["phrase_a"],
                            "Phrase B": m["phrase_b"],
                        }
                        for m in phrase_matches
                    ],
                )
            else:
                st.info("No similar phrase matches found.")

st.divider()

# Footer
st.caption("🤖 Model: TF-IDF + Cosine Similarity")
