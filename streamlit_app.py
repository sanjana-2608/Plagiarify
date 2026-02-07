from __future__ import annotations

import hashlib
import os
from typing import List, Tuple

import streamlit as st

from similarity_engine import compare_documents, load_document_from_bytes


DEFAULT_MODEL = os.getenv("MODEL_NAME", "tf-idf")
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "10"))
DEFAULT_MAX_SENTENCES = int(os.getenv("MAX_SENTENCES", "200"))
DEFAULT_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))


def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@st.cache_data(show_spinner=False)
def load_text_cached(filename: str, content_hash: str, content: bytes) -> str:
    return load_document_from_bytes(filename, content)


st.set_page_config(
    page_title="Plagiarism Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    a.stHeadingAnchor {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 Document Similarity Score")

st.subheader("📤 Upload Two Files")
uploaded_files = st.file_uploader(
    "Choose two files (TXT, PDF, or DOCX)",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True,
    help="Upload exactly 2 files for comparison",
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
    with st.expander("📋 View uploaded files"):
        for f in uploaded_files:
            st.write(f"- {f.name}")

model_name = DEFAULT_MODEL
max_sentences = DEFAULT_MAX_SENTENCES
batch_size = DEFAULT_BATCH_SIZE

st.divider()

if st.button(
    "🚀 Compare Documents",
    disabled=not uploaded_files,
    use_container_width=True,
    type="primary",
):
    if not uploaded_files or len(uploaded_files) != 2:
        st.warning("⚠️ Please upload exactly two files.")
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
        st.warning("⚠️ Please upload exactly two files.")
    else:
        r = results[0]
        st.metric("Similarity Score", f"{r.similarity * 100:.2f}%")

st.divider()

# Footer
st.caption("🤖 Model: all-MiniLM-L6-v2")
