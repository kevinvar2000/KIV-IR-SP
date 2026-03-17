import json
import time
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

    run_start = time.perf_counter()
    with open("eval_data_cs/documents.json", 'r', encoding="utf-8") as f:
        raw_docs = json.load(f)
    print(f"Loaded {len(raw_docs)} documents from eval_data_cs/documents.json")

    tokenizer = RegexMatchTokenizer()

    pipelines: dict[str, PreprocessingPipeline] = {
        # 1) baseline – lowercase, remove junk, stopwords, short tokens
        "baseline": PreprocessingPipeline([*BASE_STEPS]),
        # 2) + stemming (simplemma greedy)
        "stemming": PreprocessingPipeline([*BASE_STEPS, StemmingPreprocessor()]),
        # 3) + lemmatization (simplemma)
        "lemmatization": PreprocessingPipeline([*BASE_STEPS, LemmatizationPreprocessor()]),
        # 4) stemming + remove diacritics
        "stemming_no_diacritics": PreprocessingPipeline([
            *BASE_STEPS, StemmingPreprocessor(), RemoveDiacriticsPreprocessor(),
        ]),
        # 5) lemmatization + remove diacritics
        "lemmatization_no_diacritics": PreprocessingPipeline([
            *BASE_STEPS, LemmatizationPreprocessor(), RemoveDiacriticsPreprocessor(),
        ]),
    }

    for name, pipeline in pipelines.items():
        p_start = time.perf_counter()
        vocab = process_pipeline(name, pipeline, raw_docs, tokenizer)
        out_path = f"vocab_{name}.txt"
        write_start = time.perf_counter()
        with open(out_path, "w", encoding="utf-8") as f:
            write_weighted_vocab(vocab, f)
        write_elapsed = time.perf_counter() - write_start
        total_elapsed = time.perf_counter() - p_start
        print(f"[{name}] wrote {out_path} in {write_elapsed:.1f}s")
        print(f"[{name}] done | terms={len(vocab)} | total={total_elapsed:.1f}s")

    # Keep backward-compatible vocab.txt from the baseline pipeline.
    baseline_vocab = process_pipeline("baseline_compat", pipelines["baseline"], raw_docs, tokenizer)
    with open("vocab.txt", "w", encoding="utf-8") as f:
        write_weighted_vocab(baseline_vocab, f)
    print(f"Run finished in {time.perf_counter() - run_start:.1f}s")

