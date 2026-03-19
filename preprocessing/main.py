import argparse
import json
import time
from pathlib import Path
from typing import Iterable

from tokenizer import RegexMatchTokenizer, Tokenizer, TokenType
from preprocess import (
    PreprocessingPipeline,
    LowercasePreprocessor,
    LemmatizationPreprocessor,
    MinLengthPreprocessor,
    RemoveDiacriticsPreprocessor,
    RemoveTokenTypesPreprocessor,
    StemmingPreprocessor,
    StopwordPreprocessor,
)
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "crawler" / "crawled_pages.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "preprocessed"

PIPELINE_NAMES = [
    "baseline",
    "stemming",
    "lemmatization",
    "stemming_no_diacritics",
    "lemmatization_no_diacritics",
]

class Document:
    def __init__(self, text: str):
        self.text = text
        self.tokens = None
        self.vocab = None

    def tokenize(self, tokenizer: Tokenizer=None):
        tokenizer = tokenizer or RegexMatchTokenizer()
        self.tokens = tokenizer.tokenize(self.text)
        return self

    def preprocess(self, preprocessing_pipeline: PreprocessingPipeline):
        self.tokens = preprocessing_pipeline.preprocess(self.tokens, self.text)
        return self

def build_vocabulary(documents: Iterable[Document]):
    vocab = Counter()
    for doc in documents:
        vocab.update((token.processed_form for token in doc.tokens))
    return vocab

def write_weighted_vocab(vocab, file):
    for key, value in sorted(vocab.items(), key=lambda x: x[1], reverse=True):
        file.write(f"{key} {value}\n")


def load_records(input_path: Path) -> list[dict]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    def load_jsonl_records() -> list[dict]:
        records: list[dict] = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                records.append(json.loads(stripped))
        return records

    if input_path.suffix.lower() == ".jsonl":
        return load_jsonl_records()

    try:
        with input_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
    except json.JSONDecodeError:
        # Some datasets use JSONL content with .json extension.
        return load_jsonl_records()

    if isinstance(loaded, list):
        return loaded

    if isinstance(loaded, dict):
        return [loaded]

    raise ValueError(f"Unsupported JSON format in {input_path}")


def detect_text_keys(records: list[dict], sample_limit: int = 200) -> list[str]:
    key_score: dict[str, int] = {}
    for row in records[:sample_limit]:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if isinstance(value, str) and value.strip():
                key_score[key] = key_score.get(key, 0) + 1

    return [key for key, _ in sorted(key_score.items(), key=lambda x: (-x[1], x[0]))]


def normalize_docs(records: list[dict], text_key: str) -> list[dict]:
    docs: list[dict] = []
    for row in records:
        if not isinstance(row, dict):
            continue
        value = row.get(text_key)
        if isinstance(value, str) and value.strip():
            docs.append({"text": value})
    return docs


def parse_pipeline_selection(raw_selection: list[str]) -> list[str]:
    tokens: list[str] = []
    for item in raw_selection:
        tokens.extend(part.strip() for part in item.split(",") if part.strip())

    if not tokens or "all" in tokens:
        return PIPELINE_NAMES

    invalid = [name for name in tokens if name not in PIPELINE_NAMES]
    if invalid:
        raise ValueError(f"Unknown pipeline names: {', '.join(invalid)}")

    ordered_unique: list[str] = []
    for name in tokens:
        if name not in ordered_unique:
            ordered_unique.append(name)
    return ordered_unique


def build_pipelines() -> dict[str, PreprocessingPipeline]:
    return {
        "baseline": PreprocessingPipeline([*BASE_STEPS]),
        "stemming": PreprocessingPipeline([*BASE_STEPS, StemmingPreprocessor()]),
        "lemmatization": PreprocessingPipeline([*BASE_STEPS, LemmatizationPreprocessor()]),
        "stemming_no_diacritics": PreprocessingPipeline([
            *BASE_STEPS, StemmingPreprocessor(), RemoveDiacriticsPreprocessor(),
        ]),
        "lemmatization_no_diacritics": PreprocessingPipeline([
            *BASE_STEPS, LemmatizationPreprocessor(), RemoveDiacriticsPreprocessor(),
        ]),
    }


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
    return parser.parse_args()


def process_pipeline(
    pipeline_name: str,
    pipeline: PreprocessingPipeline,
    raw_docs: list[dict],
    tokenizer: Tokenizer,
    progress_every: int = 1000,
) -> Counter:
    total_docs = len(raw_docs)
    print(f"[{pipeline_name}] start | docs={total_docs}")

    t0 = time.perf_counter()
    documents: list[Document] = []

    # Tokenize + preprocess in one pass to reduce peak memory and show progress.
    for i, raw in enumerate(raw_docs, start=1):
        doc = Document(raw["text"]).tokenize(tokenizer).preprocess(pipeline)
        documents.append(doc)
        if i % progress_every == 0 or i == total_docs:
            elapsed = time.perf_counter() - t0
            print(f"[{pipeline_name}] processed {i}/{total_docs} docs in {elapsed:.1f}s")

    t_vocab_start = time.perf_counter()
    vocab = build_vocabulary(documents)
    t_vocab = time.perf_counter() - t_vocab_start
    print(f"[{pipeline_name}] vocab built | terms={len(vocab)} | {t_vocab:.1f}s")

    return vocab


# Shared base steps used by every pipeline
BASE_STEPS = [
    LowercasePreprocessor(),
    RemoveTokenTypesPreprocessor({TokenType.PUNCT, TokenType.TAG, TokenType.URL}),
    StopwordPreprocessor(),
    MinLengthPreprocessor(min_length=2),
]

if __name__ == '__main__':
    args = parse_args()
    run_start = time.perf_counter()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    available_keys = detect_text_keys(records)

    if args.list_text_keys:
        if available_keys:
            print("Detected text-like keys:")
            for key in available_keys:
                print(f"- {key}")
        else:
            print("No text-like keys detected.")
        raise SystemExit(0)

    selected_key = args.text_key
    if not selected_key:
        if "article_text" in available_keys:
            selected_key = "article_text"
        elif "text" in available_keys:
            selected_key = "text"
        elif available_keys:
            selected_key = available_keys[0]

    if not selected_key:
        raise ValueError("Unable to determine text key automatically. Use --text-key.")

    if available_keys and selected_key not in available_keys:
        raise ValueError(
            f"Selected text key '{selected_key}' not found among detected keys: {', '.join(available_keys)}"
        )

    raw_docs = normalize_docs(records, selected_key)
    print(f"Loaded {len(records)} rows from {input_path}")
    print(f"Using key '{selected_key}' -> {len(raw_docs)} documents")

    if not raw_docs:
        raise ValueError("No non-empty documents found for selected text key.")

    tokenizer = RegexMatchTokenizer()
    pipelines = build_pipelines()
    selected_pipeline_names = parse_pipeline_selection(args.pipelines)

    for name in selected_pipeline_names:
        p_start = time.perf_counter()
        vocab = process_pipeline(name, pipelines[name], raw_docs, tokenizer, progress_every=args.progress_every)
        out_path = output_dir / f"vocab_{name}.txt"
        write_start = time.perf_counter()
        with out_path.open("w", encoding="utf-8") as f:
            write_weighted_vocab(vocab, f)
        write_elapsed = time.perf_counter() - write_start
        total_elapsed = time.perf_counter() - p_start
        print(f"[{name}] wrote {out_path} in {write_elapsed:.1f}s")
        print(f"[{name}] done | terms={len(vocab)} | total={total_elapsed:.1f}s")

    if ("baseline" in selected_pipeline_names) and not args.no_compat_vocab:
        compat_path = output_dir / "vocab.txt"
        baseline_vocab = process_pipeline(
            "baseline_compat",
            pipelines["baseline"],
            raw_docs,
            tokenizer,
            progress_every=args.progress_every,
        )
        with compat_path.open("w", encoding="utf-8") as f:
            write_weighted_vocab(baseline_vocab, f)
        print(f"[baseline_compat] wrote {compat_path}")

    print(f"Run finished in {time.perf_counter() - run_start:.1f}s")

