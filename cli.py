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
        choices=["interactive", "all", "crawler", "preprocessing", "retrieval", "tfidf"],
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
        "--crawler-background",
        action="store_true",
        help="Run crawler in background (only for stage=crawler).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with next stages even if one fails (applies to stage=all).",
    )
    return parser.parse_args()
