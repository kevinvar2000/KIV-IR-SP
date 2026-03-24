import json
from pathlib import Path
from typing import Iterable
from collections import Counter

try:
    from .tokenizer import Tokenizer, RegexMatchTokenizer
    from .preprocess import PreprocessingPipeline
except ImportError:
    from tokenizer import Tokenizer, RegexMatchTokenizer
    from preprocess import PreprocessingPipeline


class Document:
    def __init__(self, text: str):
        self.text = text
        self.tokens = None
        self.vocab = None

    def tokenize(self, tokenizer: Tokenizer = None):
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


def write_jsonl_records(records: Iterable[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        for record in records:
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    for idx, row in enumerate(records, start=1):
        if not isinstance(row, dict):
            continue
        value = row.get(text_key)
        if isinstance(value, str) and value.strip():
            doc_id = row.get("doc_id") or row.get("id") or row.get("url") or f"doc_{idx}"
            docs.append(
                {
                    "doc_id": str(doc_id),
                    "url": row.get("url"),
                    "text": value,
                }
            )
    return docs
