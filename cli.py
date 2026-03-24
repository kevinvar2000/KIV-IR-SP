import argparse

from app_config import CRAWLER_DATA_FILE, PREPROCESSED_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run project pipeline stages from one CLI entry point.",
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="interactive",
        choices=["interactive", "all", "crawler", "preprocessing", "indexing", "retrieval", "tfidf"],
        help="Which stage to run. Default is interactive mode.",
    )
    parser.add_argument(
        "--retrieval-files",
        "--tfidf-files",
        dest="retrieval_files",
        nargs="*",
        default=None,
        help="Optional input files for retrieval stage (TF-IDF currently).",
    )
    parser.add_argument(
        "--text-key",
        default=None,
        help="Text key used for preprocessing (non-interactive mode).",
    )
    parser.add_argument(
        "--pipelines",
        nargs="*",
        default=["all"],
        help="Preprocessing pipelines for non-interactive mode.",
    )
    parser.add_argument(
        "--input-data",
        default=str(CRAWLER_DATA_FILE),
        help="Input data file for preprocessing in non-interactive mode.",
    )
    parser.add_argument(
        "--preprocess-output-dir",
        default=str(PREPROCESSED_DIR),
        help="Output directory for preprocessing artifacts.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Do not auto-build indexes after preprocessing.",
    )
    parser.add_argument(
        "--crawler-background",
        action="store_true",
        help="Run crawler in background (only for stage=crawler).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with next stages even if one fails (applies to stage=all).",
    )
    parser.add_argument(
        "--index-pipeline",
        default="baseline",
        help="Pipeline name used for indexing/retrieval index file naming.",
    )
    parser.add_argument(
        "--index-input",
        default=None,
        help="Optional preprocessed docs JSONL path for indexing stage.",
    )
    parser.add_argument(
        "--index-output",
        default=None,
        help="Optional output index JSON path for indexing stage.",
    )
    parser.add_argument(
        "--index-file",
        default=None,
        help="Optional index JSON file for retrieval stage.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional query text for retrieval stage.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K documents to show for retrieval query mode.",
    )
    return parser.parse_args()
