import argparse
from pathlib import Path

try:
    from .workflow import run_collection
except ImportError:
    from workflow import run_collection


def resolve_input_files(input_files: list[str]) -> list[Path]:
    if input_files:
        files = [Path(file) for file in input_files]
    else:
        base = Path(__file__).resolve().parent
        files = [base / "test1.txt", base / "test2.txt"]

    return [path for path in files if path.exists()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TF-IDF search on one or more collection files.")
    parser.add_argument(
        "files",
        nargs="*",
        help="Input collection files. If omitted, tries tfidf/test1.txt and tfidf/test2.txt.",
    )
    args = parser.parse_args()

    existing_files = resolve_input_files(args.files)
    if not existing_files:
        print("No input files found for TF-IDF run.")
        print("Provide files explicitly, e.g.: python tfidf/tfidf_search.py tfidf/example.txt")
        return 0

    for file_path in existing_files:
        run_collection(file_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
