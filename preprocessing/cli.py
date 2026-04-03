import argparse

try:
    from .config import DEFAULT_INPUT, DEFAULT_OUTPUT_DIR, PIPELINE_NAMES
except ImportError:
    from config import DEFAULT_INPUT, DEFAULT_OUTPUT_DIR, PIPELINE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text preprocessing pipelines.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input JSON/JSONL file with documents.",
    )
    parser.add_argument(
        "--text-key",
        default=None,
        help="Record field to preprocess (e.g. article_text, title, text).",
    )
    parser.add_argument(
        "--pipelines",
        nargs="*",
        default=["all"],
        help="Pipelines to run. Allowed: baseline, stemming, lemmatization, stemming_no_diacritics, lemmatization_no_diacritics, all.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where vocab files will be written.",
    )
    parser.add_argument(
        "--language",
        default="cs",
        help="Document language for preprocessing. Allowed: cs/czech, sk/slovak, en/english.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N documents.",
    )
    parser.add_argument(
        "--no-compat-vocab",
        action="store_true",
        help="Do not write compatibility vocab.txt from baseline.",
    )
    parser.add_argument(
        "--list-text-keys",
        action="store_true",
        help="List detected text keys from input and exit.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Do not auto-build index files after preprocessing pipelines finish.",
    )
    return parser.parse_args()
