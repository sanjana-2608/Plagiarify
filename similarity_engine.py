from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = os.getenv("MODEL_NAME", "tf-idf")


@dataclass
@dataclass
class MatchPair:
    doc_a: str
    doc_b: str
    similarity: float
    top_matches: List[Tuple[str, str, float]]


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


def compare_documents(
    documents: Iterable[Tuple[str, str]],
    model_name: str = MODEL_NAME,
    threshold: float = 0.7,
    max_pairs: int = 10,
) -> List[MatchPair]:
    cleaned_docs = [(name, preprocess_text(text)) for name, text in documents]
    if len(cleaned_docs) < 2:
        return []

    names = [name for name, _ in cleaned_docs]
    texts = [text for _, text in cleaned_docs]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    sims = cosine_similarity(tfidf_matrix)

    results: List[MatchPair] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            results.append(
                MatchPair(
                    doc_a=names[i],
                    doc_b=names[j],
                    similarity=float(sims[i, j]),
                    top_matches=[],
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
