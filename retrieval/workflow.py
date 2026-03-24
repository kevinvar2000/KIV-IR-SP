import hashlib
import json
from pathlib import Path

try:
    from .dataset import CollectionParser, Preprocessor
    from .reporting import build_term_breakdown_rows, render_table
    from .tfidf import CosineScorer, InvertedIndex
except ImportError:
    from dataset import CollectionParser, Preprocessor
    from reporting import build_term_breakdown_rows, render_table
    from tfidf import CosineScorer, InvertedIndex


def _index_file_for_collection(file_path: str | Path) -> Path:
    source = Path(file_path).resolve()
    root = Path(__file__).resolve().parent.parent
    index_dir = root / "data" / "index"
    return index_dir / f"{source.stem}_inverted_index.json"


def _documents_fingerprint(documents: dict[str, str]) -> str:
    hasher = hashlib.sha256()
    for doc_id in sorted(documents.keys()):
        hasher.update(doc_id.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(documents[doc_id].encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def _try_load_cached_index(index_path: Path, source_path: Path, documents: dict[str, str]) -> InvertedIndex | None:
    if not index_path.exists():
        return None

    with index_path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    if not isinstance(payload, dict):
        return None

    meta = payload.get("meta")
    data = payload.get("index")
    if not isinstance(meta, dict) or not isinstance(data, dict):
        return None

    expected_count = len(documents)
    expected_fingerprint = _documents_fingerprint(documents)
    if int(meta.get("doc_count", -1)) != expected_count:
        return None
    if str(meta.get("documents_sha256", "")) != expected_fingerprint:
        return None
    if str(meta.get("source_file", "")) != str(source_path):
        return None

    return InvertedIndex.from_dict(data)


def _save_index_cache(index_path: Path, source_path: Path, documents: dict[str, str], index: InvertedIndex) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "source_file": str(source_path),
            "doc_count": len(documents),
            "documents_sha256": _documents_fingerprint(documents),
        },
        "index": index.to_dict(),
    }
    with index_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False)


def run_collection(file_path: str | Path) -> None:
    """Run the full pipeline for one file and print all reports."""

    parser = CollectionParser()
    preprocessor = Preprocessor()

    source_path = Path(file_path).resolve()
    collection = parser.parse(source_path)
    index_path = _index_file_for_collection(source_path)

    index = _try_load_cached_index(index_path, source_path, collection.documents)
    if index is None:
        index = InvertedIndex()
        index.build(collection.documents, preprocessor)
        _save_index_cache(index_path, source_path, collection.documents, index)
        print(f"[index] built and saved: {index_path}")
    else:
        print(f"[index] loaded from cache: {index_path}")

    scorer = CosineScorer(index, preprocessor)

    print(f"\n=== {collection.name} ===")
    print("\nTerm DF / IDF")
    term_rows = [
        [term, str(index.doc_freq[term]), f"{index.idf[term]:.6f}"]
        for term in sorted(index.postings.keys())
    ]
    print(render_table(["term", "DF", "IDF"], term_rows))

    print("\nDocument vectors")
    for doc_id in sorted(collection.documents.keys()):
        # Per-document vector details for manual inspection.
        print(f"\n{doc_id}: {collection.documents[doc_id]}")
        rows = build_term_breakdown_rows(
            index.doc_term_freqs[doc_id],
            index,
            index.doc_vectors[doc_id],
            index.doc_norms[doc_id],
        )
        print(render_table(["term", "TF", "weighted TF", "TF-IDF", "normed"], rows))
        print(f"||{doc_id}|| = {index.doc_norms[doc_id]:.6f}")

    for query_id, query_text in collection.queries.items():
        # Query vector and ranking against document vectors.
        query_tf, query_vector, query_norm = scorer.build_query_vector(query_text)
        ranked = scorer.search(query_text)

        print(f"\n{query_id}: {query_text}")
        q_rows = build_term_breakdown_rows(query_tf, index, query_vector, query_norm)
        print(render_table(["term", "TF", "weighted TF", "TF-IDF", "normed"], q_rows))
        print(f"||{query_id}|| = {query_norm:.6f}")

        print("Ranking")
        if not ranked:
            print("  No matching documents.")
            print("Best relevant document: N/A")

        for doc_id, score in ranked:
            print(f"  {doc_id}: {score:.6f}")

        if ranked:
            best_doc, best_score = ranked[0]
            print(f"Best relevant document: {best_doc} (score={best_score:.6f})")
