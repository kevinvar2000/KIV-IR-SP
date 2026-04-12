"""CLI runner for Boolean retrieval on collection-style dataset files."""

import argparse
from pathlib import Path

from .dataset import CollectionParser, Preprocessor
from .boolean import BooleanIndex, BooleanScorer


def resolve_input_files(input_files: list[str]) -> list[Path]:
    """Resolve existing input files or fallback to local test collections."""
    if input_files:
        files = [Path(file) for file in input_files]
    else:
        base = Path(__file__).resolve().parent
        files = [base / "test1.txt", base / "test2.txt"]

    return [path for path in files if path.exists()]


def load_queries_from_file(query_file: str | Path) -> list[tuple[str, str]]:
    """Load Boolean queries from plain text file, one query per non-empty line."""
    path = Path(query_file)
    queries: list[tuple[str, str]] = []

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        query_text = line.strip()
        if not query_text:
            continue
        queries.append((f"q{line_number}", query_text))

    return queries


def run_collection(file_path: str | Path, query_rows: list[tuple[str, str]] | None = None) -> None:
    """Run Boolean retrieval for all queries in one collection file."""
    parser = CollectionParser()
    preprocessor = Preprocessor()

    collection = parser.parse(file_path)
    index = BooleanIndex()
    index.build(collection.documents, preprocessor)
    scorer = BooleanScorer(index, preprocessor)

    print(f"\n=== {collection.name} (Boolean) ===")
    queries = query_rows if query_rows is not None else list(collection.queries.items())
    if not queries:
        print("No queries found. Provide q-lines in collection file or use --query-file.")
        return

    for query_id, query_text in queries:
        ranked = scorer.search(query_text)
        print(f"\n{query_id}: {query_text}")
        print("Ranking")
        if not ranked:
            print("  No matching documents.")
            print("Best relevant document: N/A")
            continue
        for doc_id, score in ranked:
            print(f"  {doc_id}: {score:.6f}")
        best_doc, best_score = ranked[0]
        print(f"Best relevant document: {best_doc} (score={best_score:.6f})")


def main() -> int:
    """CLI entry point for Boolean retrieval runner."""
    parser = argparse.ArgumentParser(description="Run Boolean search on one or more collection files.")
    parser.add_argument(
        "files",
        nargs="*",
        help="Input collection files. If omitted, tries retrieval/test1.txt and retrieval/test2.txt.",
    )
    parser.add_argument(
        "--query-file",
        help="Path to text file with one Boolean query per line.",
    )
    args = parser.parse_args()

    query_rows: list[tuple[str, str]] | None = None
    if args.query_file:
        query_path = Path(args.query_file)
        if not query_path.exists():
            print(f"Query file not found: {query_path}")
            return 1
        query_rows = load_queries_from_file(query_path)
        if not query_rows:
            print(f"No non-empty queries found in: {query_path}")
            return 1

    existing_files = resolve_input_files(args.files)
    if not existing_files:
        print("No input files found for Boolean run.")
        print("Provide files explicitly, e.g.: python retrieval/boolean_search.py retrieval/example.txt")
        return 0

    for file_path in existing_files:
        run_collection(file_path, query_rows=query_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
