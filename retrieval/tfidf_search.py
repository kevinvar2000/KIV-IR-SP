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


def list_data_files() -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    if not data_dir.exists():
        return []

    return sorted(path for path in data_dir.rglob("*.txt") if path.is_file())


def choose_files_from_data_folder() -> list[Path]:
    data_files = list_data_files()
    if not data_files:
        return []

    root = Path(__file__).resolve().parent.parent
    print("No input files provided.")
    print("Choose collection files from the data folder or type paths from project root.")
    print("Available files:")
    for idx, path in enumerate(data_files, start=1):
        print(f"  {idx}. {path.relative_to(root)}")

    raw = input(
        "Select file numbers (comma-separated) or enter root-relative paths (comma-separated): "
    ).strip()

    if not raw:
        return []

    parts = [part.strip() for part in raw.split(",") if part.strip()]
    selected: list[Path] = []
    for part in parts:
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(data_files):
                selected.append(data_files[idx - 1])
            continue

        candidate = root / part
        if candidate.exists() and candidate.is_file():
            selected.append(candidate)

    # Keep order while removing duplicates.
    unique_selected = list(dict.fromkeys(selected))
    return unique_selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TF-IDF search on one or more collection files.")
    parser.add_argument(
        "files",
        nargs="*",
        help=(
            "Input collection files (paths from project root). "
            "If omitted, you can choose files from data/ interactively; "
            "fallback is retrieval/test1.txt and retrieval/test2.txt."
        ),
    )
    args = parser.parse_args()

    existing_files = resolve_input_files(args.files)
    if not args.files:
        chosen_files = choose_files_from_data_folder()
        if chosen_files:
            existing_files = chosen_files

    if not existing_files:
        print("No input files found for TF-IDF run.")
        print(
            "Provide files explicitly from project root, "
            "e.g.: python retrieval/tfidf_search.py retrieval/test1.txt"
        )
        return 0

    for file_path in existing_files:
        run_collection(file_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
