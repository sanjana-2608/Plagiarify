from __future__ import annotations

import io
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Tuple

import numpy as np
from PyPDF2 import PdfReader
from docx import Document
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk


MODEL_NAME = os.getenv("MODEL_NAME", "paraphrase-MiniLM-L3-v2")
DEFAULT_MAX_SENTENCES = int(os.getenv("MAX_SENTENCES", "200"))
DEFAULT_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))


@dataclass
class DocumentData:
    name: str
    text: str
    sentences: List[str]
    sentence_embeddings: np.ndarray
    document_embedding: np.ndarray


@dataclass
class MatchPair:
    doc_a: str
    doc_b: str
    similarity: float
    top_matches: List[Tuple[str, str, float]]


def _ensure_nltk():
    try:
        sent_tokenize("test.")
    except LookupError:
        nltk.download("punkt", quiet=True)


@lru_cache(maxsize=2)
def get_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _read_txt_bytes(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1", errors="ignore")


def _read_pdf_bytes(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    text_parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text_parts.append(text)
    return "\n".join(text_parts)


def _read_docx_bytes(content: bytes) -> str:
    doc = Document(io.BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs)


def load_document_from_bytes(filename: str, content: bytes) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".txt":
        return _read_txt_bytes(content)
    if ext == ".pdf":
        return _read_pdf_bytes(content)
    if ext in {".docx", ".doc"}:
        return _read_docx_bytes(content)
    raise ValueError(f"Unsupported file type: {ext}")


def load_document_from_path(path: str) -> Tuple[str, str]:
    name = os.path.basename(path)
    with open(path, "rb") as f:
        content = f.read()
    text = load_document_from_bytes(name, content)
    return name, text


def preprocess_text(text: str) -> str:
    cleaned = text.replace("\u00a0", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned


def split_sentences(text: str) -> List[str]:
    _ensure_nltk()
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]


def embed_sentences(
    model: SentenceTransformer,
    sentences: List[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    if not sentences:
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)
    return model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )


def build_document_data(
    model: SentenceTransformer,
    name: str,
    text: str,
    max_sentences: int | None = DEFAULT_MAX_SENTENCES,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> DocumentData:
    cleaned = preprocess_text(text)
    sentences = split_sentences(cleaned)
    if max_sentences:
        sentences = sentences[:max_sentences]
    sentence_embeddings = embed_sentences(model, sentences, batch_size=batch_size)

    if sentence_embeddings.size == 0:
        dim = model.get_sentence_embedding_dimension()
        document_embedding = np.zeros((dim,), dtype=np.float32)
    else:
        document_embedding = sentence_embeddings.mean(axis=0)
        norm = np.linalg.norm(document_embedding)
        if norm > 0:
            document_embedding = document_embedding / norm

    return DocumentData(
        name=name,
        text=cleaned,
        sentences=sentences,
        sentence_embeddings=sentence_embeddings,
        document_embedding=document_embedding,
    )


def _top_sentence_matches(
    doc_a: DocumentData,
    doc_b: DocumentData,
    threshold: float = 0.7,
    max_pairs: int = 10,
) -> List[Tuple[str, str, float]]:
    if doc_a.sentence_embeddings.size == 0 or doc_b.sentence_embeddings.size == 0:
        return []

    sims = cosine_similarity(doc_a.sentence_embeddings, doc_b.sentence_embeddings)
    matches = []
    for i in range(sims.shape[0]):
        for j in range(sims.shape[1]):
            score = float(sims[i, j])
            if score >= threshold:
                matches.append((doc_a.sentences[i], doc_b.sentences[j], score))

    matches.sort(key=lambda x: x[2], reverse=True)
    return matches[:max_pairs]


def compare_documents(
    documents: Iterable[Tuple[str, str]],
    model_name: str = MODEL_NAME,
    model: SentenceTransformer | None = None,
    threshold: float = 0.7,
    max_pairs: int = 10,
    max_sentences: int | None = DEFAULT_MAX_SENTENCES,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[MatchPair]:
    model = model or get_model(model_name)
    doc_data = [
        build_document_data(
            model, name, text, max_sentences=max_sentences, batch_size=batch_size
        )
        for name, text in documents
    ]

    results: List[MatchPair] = []
    for i in range(len(doc_data)):
        for j in range(i + 1, len(doc_data)):
            doc_a = doc_data[i]
            doc_b = doc_data[j]
            sim = float(
                cosine_similarity(
                    doc_a.document_embedding.reshape(1, -1),
                    doc_b.document_embedding.reshape(1, -1),
                )[0, 0]
            )
            top_matches = _top_sentence_matches(
                doc_a, doc_b, threshold=threshold, max_pairs=max_pairs
            )
            results.append(
                MatchPair(
                    doc_a=doc_a.name,
                    doc_b=doc_b.name,
                    similarity=sim,
                    top_matches=top_matches,
                )
            )

    return results


def results_to_json(results: List[MatchPair]) -> str:
    payload = []
    for r in results:
        payload.append(
            {
                "doc_a": r.doc_a,
                "doc_b": r.doc_b,
                "similarity": round(r.similarity * 100, 2),
                "top_matches": [
                    {"a": a, "b": b, "score": round(score * 100, 2)}
                    for a, b, score in r.top_matches
                ],
            }
        )
    return json.dumps(payload, indent=2)
