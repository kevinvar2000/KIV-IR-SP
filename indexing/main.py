import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    from app_config import INDEX_DIR, PREPROCESSED_DIR
    from retrieval.tfidf import InvertedIndex
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app_config import INDEX_DIR, PREPROCESSED_DIR
    from retrieval.tfidf import InvertedIndex


class WhitespacePreprocessor:
    """Pre-tokenized text adapter used for index build from normalized text."""

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [token for token in text.split() if token]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a persisted inverted index from preprocessed docs.")
    parser.add_argument(
        "--pipeline",
        default="baseline",
        help="Pipeline name used to locate docs_<pipeline>.jsonl and annotate metadata.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input JSONL docs file. Default: data/preprocessed/docs_<pipeline>.jsonl",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output index JSON file. Default: data/index/inverted_index_<pipeline>.json",
    )
    return parser.parse_args()


def load_preprocessed_docs(input_path: Path) -> dict[str, str]:
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
    hasher = hashlib.sha256()
    for doc_id in sorted(documents.keys()):
        hasher.update(doc_id.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(documents[doc_id].encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def main() -> int:
    args = parse_args()
    return run_indexing_stage(
        pipeline=args.pipeline,
        input_path=args.input,
        output_path=args.output,
    )


def run_indexing_stage(
    *,
    pipeline: str = "baseline",
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> int:
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
