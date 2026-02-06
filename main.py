from __future__ import annotations

import argparse
from typing import List, Tuple

from similarity_engine import compare_documents, load_document_from_path, results_to_json


def load_documents(paths: List[str]) -> List[Tuple[str, str]]:
    documents: List[Tuple[str, str]] = []
    for path in paths:
        name, text = load_document_from_path(path)
        documents.append((name, text))
    return documents


def main():
    parser = argparse.ArgumentParser(description="Document similarity engine")
    parser.add_argument("files", nargs="+", help="Paths to PDF/TXT/DOCX files")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Sentence match threshold (0-1)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=10,
        help="Max top matching sentence pairs",
    )
    args = parser.parse_args()

    documents = load_documents(args.files)
    results = compare_documents(
        documents, threshold=args.threshold, max_pairs=args.max_pairs
    )
    print(results_to_json(results))


if __name__ == "__main__":
    main()
