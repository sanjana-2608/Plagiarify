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
    [data-testid="stToolbar"] {
        display: none;
    }
    a.stHeadingAnchor {
        display: none;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #4b4b4b;
        padding: 36px 12px;
        border-radius: 12px;
        background: #0f1117;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]::before {
        content: "+";
        display: block;
        font-size: 44px;
        line-height: 44px;
        text-align: center;
        color: #8a8a8a;
        margin-bottom: 10px;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] div,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] span {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 Document Similarity Score")

st.subheader("📤 Upload Two Files")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**File A**")
    file_a = st.file_uploader(
        "Upload file A",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=False,
        help="Upload the first file",
        label_visibility="collapsed",
        key="file_a",
    )
with col_b:
    st.markdown("**File B**")
    file_b = st.file_uploader(
        "Upload file B",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=False,
        help="Upload the second file",
        label_visibility="collapsed",
        key="file_b",
    )

uploaded_files = [f for f in (file_a, file_b) if f]

if file_a and file_b:
    st.success("✅ 2 files uploaded successfully!")
    with st.expander("📋 View uploaded files"):
        st.write(f"- {file_a.name}")
        st.write(f"- {file_b.name}")

model_name = DEFAULT_MODEL
max_sentences = DEFAULT_MAX_SENTENCES
batch_size = DEFAULT_BATCH_SIZE

st.divider()

if st.button(
    "🚀 Compare Documents",
    disabled=not (file_a and file_b),
    use_container_width=True,
    type="primary",
):
    if not file_a or not file_b:
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
