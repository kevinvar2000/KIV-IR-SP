import argparse
from pathlib import Path

try:
    from .dataset import CollectionParser, Preprocessor
    from .boolean import BooleanIndex, BooleanScorer
except ImportError:
    from dataset import CollectionParser, Preprocessor
    from boolean import BooleanIndex, BooleanScorer


def resolve_input_files(input_files: list[str]) -> list[Path]:
    if input_files:
        files = [Path(file) for file in input_files]
    else:
        base = Path(__file__).resolve().parent
        files = [base / "test1.txt", base / "test2.txt"]

    return [path for path in files if path.exists()]


def run_collection(file_path: str | Path) -> None:
    parser = CollectionParser()
    preprocessor = Preprocessor()

    collection = parser.parse(file_path)
    index = BooleanIndex()
    index.build(collection.documents, preprocessor)
    scorer = BooleanScorer(index, preprocessor)

    print(f"\n=== {collection.name} (Boolean) ===")
    for query_id, query_text in collection.queries.items():
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
    parser = argparse.ArgumentParser(description="Run Boolean search on one or more collection files.")
    parser.add_argument(
        "files",
        nargs="*",
        help="Input collection files. If omitted, tries retrieval/test1.txt and retrieval/test2.txt.",
    )
    args = parser.parse_args()

    existing_files = resolve_input_files(args.files)
    if not existing_files:
        print("No input files found for Boolean run.")
        print("Provide files explicitly, e.g.: python retrieval/boolean_search.py retrieval/example.txt")
        return 0

    for file_path in existing_files:
        run_collection(file_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
