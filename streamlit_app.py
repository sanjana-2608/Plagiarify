from __future__ import annotations

import hashlib
import os
from typing import List, Tuple

import pandas as pd
import streamlit as st

from similarity_engine import (
    compare_documents,
    get_model,
    load_document_from_bytes,
    results_to_json,
)


DEFAULT_MODEL = os.getenv("MODEL_NAME", "paraphrase-MiniLM-L3-v2")
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "10"))
DEFAULT_MAX_SENTENCES = int(os.getenv("MAX_SENTENCES", "200"))
DEFAULT_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))


def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    model = get_model(model_name)
    model.encode(["warmup"], convert_to_numpy=True, normalize_embeddings=True)
    return model


@st.cache_data(show_spinner=False)
def load_text_cached(filename: str, content_hash: str, content: bytes) -> str:
    return load_document_from_bytes(filename, content)


st.set_page_config(
    page_title="Plagiarism Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔍 Plagiarism & Document Similarity Detector")
st.markdown("""
### Welcome to the Plagiarism Detection System

This tool uses advanced AI to:
- 📄 Compare multiple documents simultaneously
- 🔬 Detect semantic similarity (not just exact matches)
- 🎯 Identify paraphrased content
- 📊 Generate detailed similarity reports

**Instructions:**
1. Upload 2 or more documents (PDF, TXT, or DOCX)
2. Adjust the similarity threshold if needed
3. Click "Compare Documents" to analyze
""")

st.divider()

st.subheader("📤 Upload Documents")
uploaded_files = st.file_uploader(
    "Choose your documents to compare",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
    help="Upload at least 2 documents for comparison"
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} document(s) uploaded successfully!")
    with st.expander("📋 View uploaded files"):
        for f in uploaded_files:
            st.write(f"- {f.name}")

st.subheader("⚙️ Configuration")
col1, col2 = st.columns(2)
with col1:
    threshold = st.slider(
        "Sentence match threshold",
        0.5,
        0.95,
        0.7,
        0.05,
        help="Higher threshold = stricter matching (0.7 recommended)",
    )
with col2:
    max_pairs = st.slider(
        "Max sentence matches per pair",
        3,
        20,
        10,
        1,
        help="Maximum number of similar sentences to display",
    )

st.sidebar.header("⚡ Performance Settings")
model_name = st.sidebar.selectbox(
    "Embedding model",
    [
        "paraphrase-MiniLM-L3-v2",
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
    ],
    index=0,
    help="Faster models load quicker on Railway",
)
max_sentences = st.sidebar.slider(
    "Max sentences per document",
    50,
    500,
    DEFAULT_MAX_SENTENCES,
    25,
    help="Lower values make comparisons faster",
)
batch_size = st.sidebar.selectbox(
    "Embedding batch size",
    [16, 32, 64],
    index=1,
    help="Higher values can be faster but use more memory",
)

st.divider()

if st.button(
    "🚀 Compare Documents",
    disabled=not uploaded_files,
    use_container_width=True,
    type="primary",
):
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

    with st.spinner("Computing similarity..."):
        model = load_model(model_name)
        results = compare_documents(
            documents,
            model_name=model_name,
            model=model,
            threshold=threshold,
            max_pairs=max_pairs,
            max_sentences=max_sentences,
            batch_size=batch_size,
        )

    if not results:
        st.warning("⚠️ Please upload at least two documents.")
    else:
        st.success("✅ Analysis complete!")
        
        st.subheader("📊 Similarity Scores")
        table_rows = [
            {
                "Document A": r.doc_a,
                "Document B": r.doc_b,
                "Similarity %": round(r.similarity * 100, 2),
            }
            for r in results
        ]
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        st.subheader("📝 Top Matching Sentences")
        for r in results:
            with st.expander(f"{r.doc_a} ↔ {r.doc_b}"):
                if not r.top_matches:
                    st.write("No sentence-level matches above threshold.")
                else:
                    for i, (a, b, score) in enumerate(r.top_matches, 1):
                        st.markdown(f"**Match #{i} - Score:** {score * 100:.2f}%")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.info(f"📄 {r.doc_a}")
                            st.write(a)
                        with col_b:
                            st.info(f"📄 {r.doc_b}")
                            st.write(b)
                        st.divider()

        st.subheader("💾 Download Report")
        report_json = results_to_json(results)
        st.download_button(
            "📥 Download JSON Report",
            data=report_json,
            file_name="similarity_report.json",
            mime="application/json",
            use_container_width=True
        )

st.divider()

# Footer
st.caption(
    f"🤖 Model: {model_name} (SentenceTransformers) | Cosine similarity scoring"
)
st.caption("Made with ❤️ using Streamlit")
