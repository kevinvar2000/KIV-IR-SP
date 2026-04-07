"""Build and persist an inverted index from preprocessed JSONL documents."""

import hashlib
import json
from pathlib import Path

from app_config import INDEX_DIR, PREPROCESSED_DIR
from retrieval.tfidf import InvertedIndex


class WhitespacePreprocessor:
    """Pre-tokenized text adapter used for index build from normalized text."""

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Split normalized text by whitespace into tokens."""
        return [token for token in text.split() if token]


def load_preprocessed_docs(input_path: Path) -> dict[str, str]:
    """Load doc_id -> normalized_text mapping from preprocessing JSONL output."""
    if not input_path.exists():
        raise FileNotFoundError(f"Preprocessed docs file does not exist: {input_path}")

    documents: dict[str, str] = {}
    with input_path.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                continue

            doc_id = str(record.get("doc_id") or f"doc_{line_number}")
            normalized_text = str(record.get("normalized_text") or "")
            if not normalized_text.strip():
                continue
            documents[doc_id] = normalized_text

    return documents


def documents_fingerprint(documents: dict[str, str]) -> str:
    """Build a stable SHA-256 fingerprint for a document mapping."""
    hasher = hashlib.sha256()
    for doc_id in sorted(documents.keys()):
        hasher.update(doc_id.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(documents[doc_id].encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def main() -> int:
    """Run indexing with default paths and pipeline settings."""
    return run_indexing_stage()


def run_indexing_stage(
    *,
    pipeline: str = "baseline",
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> int:
    """Build and save an inverted index for preprocessed documents."""
    input_path = Path(input_path) if input_path else PREPROCESSED_DIR / f"docs_{pipeline}.jsonl"
    output_path = Path(output_path) if output_path else INDEX_DIR / f"inverted_index_{pipeline}.json"

    documents = load_preprocessed_docs(input_path)
    if not documents:
        raise ValueError(f"No normalized documents found in {input_path}")

    index = InvertedIndex()
    index.build(documents, WhitespacePreprocessor())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "pipeline": pipeline,
            "source_file": str(input_path.resolve()),
            "doc_count": len(documents),
            "documents_sha256": documents_fingerprint(documents),
        },
        "index": index.to_dict(),
    }
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False)

    print(f"[indexing] docs={len(documents)} terms={len(index.postings)}")
    print(f"[indexing] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
