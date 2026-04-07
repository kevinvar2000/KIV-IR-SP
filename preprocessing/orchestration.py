"""Utilities to select preprocessing pipelines and process documents."""

import time
import json
from collections import Counter
from pathlib import Path

from .config import PIPELINE_NAMES
from .dataset import Document, build_vocabulary
from .tokenizer import Tokenizer
from .preprocess import PreprocessingPipeline


def parse_pipeline_selection(raw_selection: list[str]) -> list[str]:
    """Parse comma-separated pipeline selection into validated unique names."""
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


def process_pipeline(
    pipeline_name: str,
    pipeline: PreprocessingPipeline,
    raw_docs: list[dict],
    tokenizer: Tokenizer,
    progress_every: int = 1000,
) -> tuple[dict, list[dict]]:
    """Process documents in memory and return vocabulary with normalized records."""
    total_docs = len(raw_docs)
    print(f"[{pipeline_name}] start | docs={total_docs}")

    t0 = time.perf_counter()
    documents: list[Document] = []
    normalized_docs: list[dict] = []

    # Tokenize + preprocess in one pass to reduce peak memory and show progress.
    for i, raw in enumerate(raw_docs, start=1):
        doc = Document(raw["text"]).tokenize(tokenizer).preprocess(pipeline)
        documents.append(doc)

        processed_tokens = [token.processed_form for token in doc.tokens if token.processed_form]
        normalized_docs.append(
            {
                "doc_id": raw["doc_id"],
                "url": raw.get("url"),
                "tokens": processed_tokens,
                "normalized_text": " ".join(processed_tokens),
            }
        )

        if i % progress_every == 0 or i == total_docs:
            elapsed = time.perf_counter() - t0
            print(f"[{pipeline_name}] processed {i}/{total_docs} docs in {elapsed:.1f}s")

    t_vocab_start = time.perf_counter()
    vocab = build_vocabulary(documents)
    t_vocab = time.perf_counter() - t_vocab_start
    print(f"[{pipeline_name}] vocab built | terms={len(vocab)} | {t_vocab:.1f}s")

    return vocab, normalized_docs


def process_pipeline_to_files(
    pipeline_name: str,
    pipeline: PreprocessingPipeline,
    raw_docs: list[dict],
    tokenizer: Tokenizer,
    docs_output_path: Path,
    progress_every: int = 1000,
) -> Counter:
    """Process docs in one pass, write JSONL incrementally, and build vocabulary incrementally."""
    total_docs = len(raw_docs)
    print(f"[{pipeline_name}] start | docs={total_docs}")

    t0 = time.perf_counter()
    vocab: Counter = Counter()
    docs_output_path.parent.mkdir(parents=True, exist_ok=True)

    with docs_output_path.open("w", encoding="utf-8") as out_file:
        for i, raw in enumerate(raw_docs, start=1):
            doc = Document(raw["text"]).tokenize(tokenizer).preprocess(pipeline)
            processed_tokens = [token.processed_form for token in doc.tokens if token.processed_form]
            vocab.update(processed_tokens)

            normalized_record = {
                "doc_id": raw["doc_id"],
                "url": raw.get("url"),
                "tokens": processed_tokens,
                "normalized_text": " ".join(processed_tokens),
            }
            out_file.write(json.dumps(normalized_record, ensure_ascii=False) + "\n")

            if i % progress_every == 0 or i == total_docs:
                elapsed = time.perf_counter() - t0
                print(f"[{pipeline_name}] processed {i}/{total_docs} docs in {elapsed:.1f}s")

    print(f"[{pipeline_name}] vocab built | terms={len(vocab)}")
    return vocab
