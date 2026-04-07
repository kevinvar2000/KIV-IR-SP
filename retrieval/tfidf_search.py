"""CLI entry points for index-based and collection-file TF-IDF retrieval modes."""

import argparse
import json
from pathlib import Path

from app_config import INDEX_DIR
from preprocessing.config import PIPELINE_NAMES, build_pipelines
from preprocessing.language_config import normalize_language_code
from preprocessing.tokenizer import RegexMatchTokenizer

from .tfidf import CosineScorer, InvertedIndex
from .workflow import run_collection


def resolve_input_files(input_files: list[str]) -> list[Path]:
    """Resolve provided collection file paths or fallback examples."""
    if input_files:
        files = [Path(file) for file in input_files]
    else:
        base = Path(__file__).resolve().parent
        files = [base / "test1.txt", base / "test2.txt"]

    return [path for path in files if path.exists()]


def resolve_index_file(index_file: str | None, pipeline: str) -> Path:
    """Resolve index file path from explicit argument or pipeline default."""
    if index_file:
        return Path(index_file)
    return INDEX_DIR / f"inverted_index_{pipeline}.json"


def load_index_payload(index_path: Path) -> tuple[InvertedIndex, dict]:
    """Load serialized index and optional metadata payload from JSON file."""
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    with index_path.open("r", encoding="utf-8") as file_handle:
        loaded = json.load(file_handle)

    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid index format in {index_path}")

    if "index" in loaded and isinstance(loaded["index"], dict):
        index_data = loaded["index"]
        meta = loaded.get("meta", {}) if isinstance(loaded.get("meta", {}), dict) else {}
    else:
        index_data = loaded
        meta = {}

    return InvertedIndex.from_dict(index_data), meta


class PipelineQueryPreprocessor:
    """Apply the same preprocessing pipeline to queries as used for index generation."""

    @staticmethod
    def _resolve_pipeline_and_language(pipeline_name: str) -> tuple[str, str]:
        """Resolve pipeline base name and language from optional suffix."""
        normalized = pipeline_name.strip()
        if normalized in PIPELINE_NAMES:
            return normalized, "cs"

        for language_code in ("cs", "sk", "en"):
            suffix = f"_{language_code}"
            if normalized.endswith(suffix):
                candidate = normalized[: -len(suffix)]
                if candidate in PIPELINE_NAMES:
                    return candidate, normalize_language_code(language_code)

        raise ValueError(f"Unknown preprocessing pipeline for query mode: {pipeline_name}")

    def __init__(self, pipeline_name: str):
        """Create query preprocessor aligned with indexing pipeline settings."""
        resolved_pipeline, resolved_language = self._resolve_pipeline_and_language(pipeline_name)
        pipelines = build_pipelines(resolved_language)
        self.pipeline = pipelines[resolved_pipeline]
        self.tokenizer = RegexMatchTokenizer()

    def tokenize(self, text: str) -> list[str]:
        """Tokenize and preprocess query text into normalized terms."""
        tokens = self.tokenizer.tokenize(text)
        tokens = self.pipeline.preprocess(tokens, text)
        return [token.processed_form for token in tokens if token.processed_form]


def run_index_query_mode(index_path: Path, pipeline_name: str, query: str | None, top_k: int) -> int:
    """Run query-time retrieval directly against a persisted index file."""
    index, meta = load_index_payload(index_path)

    resolved_pipeline = str(meta.get("pipeline") or pipeline_name)
    query_preprocessor = PipelineQueryPreprocessor(resolved_pipeline)

    print(f"[retrieval] loaded index: {index_path}")
    print(f"[retrieval] pipeline: {resolved_pipeline}")
    print(f"[retrieval] docs={index.num_docs} terms={len(index.postings)}")

    if not query:
        print("[retrieval] no query provided. Use --query to run search against this index.")
        return 0

    scorer = CosineScorer(index, query_preprocessor)
    ranked = scorer.search(query)
    limited = ranked[: max(top_k, 1)]

    print(f"\nQuery: {query}")
    if not limited:
        print("No matching documents.")
        return 0

    print("Top results:")
    for rank, (doc_id, score) in enumerate(limited, start=1):
        print(f"{rank}. {doc_id} -> {score:.6f}")
    return 0


def _looks_like_collection_file(path: Path, max_lines: int = 30) -> bool:
    """Heuristically detect legacy collection files containing d*/q* lines."""
    if path.name.lower().startswith("vocab"):
        return False

    try:
        with path.open("r", encoding="utf-8") as file_handle:
            seen_non_empty = 0
            for line in file_handle:
                stripped = line.strip()
                if not stripped:
                    continue
                seen_non_empty += 1
                if ":" in stripped:
                    prefix = stripped.split(":", maxsplit=1)[0].strip().lower()
                    if prefix.startswith("d") or prefix.startswith("q"):
                        return True
                if seen_non_empty >= max_lines:
                    break
    except OSError:
        return False

    return False


def list_data_files() -> list[Path]:
    """List candidate legacy collection files under data directory."""
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    if not data_dir.exists():
        return []

    txt_files = sorted(path for path in data_dir.rglob("*.txt") if path.is_file())
    return [path for path in txt_files if _looks_like_collection_file(path)]


def choose_files_from_data_folder() -> list[Path]:
    """Interactively choose collection files from the data folder."""
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for retrieval runner."""
    parser = argparse.ArgumentParser(description="Run TF-IDF search on one or more collection files.")
    parser.add_argument(
        "files",
        nargs="*",
        help=(
            "Legacy mode input collection files using d*/q* lines. "
            "If provided, retrieval runs in collection-file compatibility mode."
        ),
    )
    parser.add_argument(
        "--index-file",
        default=None,
        help="Index JSON file path. Default: data/index/inverted_index_<pipeline>.json",
    )
    parser.add_argument(
        "--pipeline",
        default="baseline",
        help="Pipeline name for default index path and query preprocessing (when metadata absent).",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Query string for index-based retrieval mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top results shown in index-based query mode.",
    )
    return parser.parse_args()


def run_retrieval_stage(
    *,
    files: list[str] | None = None,
    index_file: str | None = None,
    pipeline: str = "baseline",
    query: str | None = None,
    top_k: int = 10,
) -> int:
    """Run retrieval in index-query mode or legacy collection compatibility mode."""
    files = files or []

    if not files:
        index_path = resolve_index_file(index_file, pipeline)
        return run_index_query_mode(index_path, pipeline, query, top_k)

    existing_files = resolve_input_files(files)

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


def main() -> int:
    """CLI entry point for TF-IDF retrieval runner."""
    args = parse_args()
    return run_retrieval_stage(
        files=args.files,
        index_file=args.index_file,
        pipeline=args.pipeline,
        query=args.query,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    raise SystemExit(main())
